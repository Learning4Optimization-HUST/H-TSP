import os
import time

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from fast_pytorch_kmeans import KMeans
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

import models
import utils


class GroupState:

    def __init__(self, group_size, x):
        # x.shape = [B, N, 2]
        self.batch_size = x.size(0)
        self.group_size = group_size
        self.device = x.device

        self.selected_count = 0
        # current_node.shape = [B, G]
        self.current_node = None
        # selected_node_list.shape = [B, G, selected_count]
        self.selected_node_list = torch.zeros(x.size(0),
                                              group_size,
                                              0,
                                              device=x.device).long()
        # ninf_mask.shape = [B, G, N]
        self.ninf_mask = torch.zeros(x.size(0),
                                     group_size,
                                     x.size(1),
                                     device=x.device)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        batch_idx_mat = torch.arange(self.batch_size)[:, None].expand(
            self.batch_size, self.group_size).to(self.device)
        group_idx_mat = torch.arange(self.group_size)[None, :].expand(
            self.batch_size, self.group_size).to(self.device)
        self.ninf_mask[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf


class Env:

    def __init__(
        self,
        x,
    ):
        self.x = x
        self.batch_size = self.B = x.size(0)
        self.graph_size = self.N = x.size(1)
        self.node_dim = self.C = x.size(2)
        self.group_size = self.G = None
        self.group_state = None

    def reset(self, group_size):
        self.group_size = group_size
        self.group_state = GroupState(group_size=group_size, x=self.x)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = (self.group_state.selected_count == self.graph_size)
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
        # ordered_seq.shape = [B, G, N, C]
        shp = (self.B, self.group_size, self.N, self.C)
        gathering_index = self.group_state.selected_node_list.unsqueeze(
            3).expand(*shp)
        seq_expanded = self.x[:, None, :, :].expand(*shp)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        # segment_lengths.size = [B, G, N]
        segment_lengths = ((ordered_seq - rolled_seq)**2).sum(3).sqrt()

        group_travel_distances = segment_lengths.sum(2)
        return group_travel_distances


class LSTSPCentroidDataset(Dataset):

    def __init__(self,
                 lstsp_size,
                 n_clusters,
                 node_dim=2,
                 num_samples=100000,
                 data_distribution='uniform'):
        super(LSTSPCentroidDataset, self).__init__()
        self.lstsp_size = lstsp_size
        self.n_clusters = n_clusters
        self.node_dim = node_dim
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        kemans = KMeans(n_clusters=self.n_clusters, max_iter=100)
        data = torch.rand(self.lstsp_size, self.node_dim)
        kemans.fit(data)
        return kemans.centroids


class LoopSolver(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        if cfg.node_dim > 2:
            assert 'noAug' in cfg.val_type, \
                "High-dimension TSP doesn't support augmentation"
        if cfg.encoder_type == 'mha':
            self.encoder = models.MHAEncoder(
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                embedding_dim=cfg.embedding_dim,
                input_dim=cfg.node_dim,
                add_init_projection=cfg.add_init_projection)
        elif cfg.encoder_type == 'mlp':
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(cfg.node_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, cfg.embedding_dim),
                torch.nn.ReLU(),
            )
        else:
            if cfg.gnn_framework == 'pyg':
                self.encoder = models.PYG_DeepGCNEncoder(
                    embedding_dim=cfg.embedding_dim,
                    input_e_dim=1,
                    input_h_dim=cfg.node_dim,
                    n_layers=cfg.n_layers,
                    n_neighbors=cfg.n_encoding_neighbors,
                    conv_type=cfg.pyg_conv_type)
            else:
                self.encoder = models.DGL_ResGatedGraphEncoder(
                    embedding_dim=cfg.embedding_dim,
                    input_e_dim=1,
                    input_h_dim=cfg.node_dim,
                    n_layers=cfg.n_layers,
                    n_neighbors=cfg.n_encoding_neighbors)
            assert cfg.precision == 32, 'GNN can only handle float32 or float64'
        self.decoder = models.LoopDecoder(
            embedding_dim=cfg.embedding_dim,
            n_heads=cfg.n_heads,
            tanh_clipping=cfg.tanh_clipping,
            n_decoding_neighbors=cfg.n_decoding_neighbors)
        self.cfg = cfg
        self.save_hyperparameters(cfg)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.cfg.learning_rate *
                                      len(self.cfg.gpus),
                                      weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=2,
                                                       gamma=0.99)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def forward(self, batch, val_type=None, return_pi=False):
        val_type = val_type or self.cfg.val_type
        if val_type == 'x8Aug_nTraj':
            batch = utils.augment_xy_data_by_8_fold(batch)

        B, N, _ = batch.shape
        G = 1 if val_type == 'noAug_1Traj' else N

        env = Env(batch)
        s, r, d = env.reset(group_size=G)
        embeddings = self.encoder(batch)
        self.decoder.reset(batch, embeddings, s.ninf_mask)
        first_action = torch.arange(G, device=self.device,
                                    dtype=torch.long)[None, :].expand(B, G)
        pi = first_action[..., None]
        s, r, d = env.step(first_action)

        while not d:
            action_probs = self.decoder(s.current_node)
            action = action_probs.argmax(dim=2)
            pi = torch.cat([pi, action[..., None]], dim=-1)
            s, r, d = env.step(action)

        if val_type == 'noAug_1Traj':
            max_reward = r
            best_pi = pi
        elif val_type == 'noAug_nTraj':
            max_reward, idx_dim_1 = r.max(dim=1)
            idx_dim_1 = idx_dim_1.reshape(B, 1, 1)
            best_pi = pi.gather(1, idx_dim_1.repeat(1, 1, N))
        else:
            B = round(B / 8)
            reward = r.reshape(8, B, G)
            max_reward, idx_dim_2 = reward.max(dim=2)
            max_reward, idx_dim_0 = max_reward.max(dim=0)
            pi = pi.reshape(8, B, G, N)
            idx_dim_0 = idx_dim_0.reshape(1, B, 1, 1)
            idx_dim_2 = idx_dim_2.reshape(8, B, 1, 1).gather(0, idx_dim_0)
            best_pi = pi.gather(0, idx_dim_0.repeat(1, 1, G, N))
            best_pi = best_pi.gather(2, idx_dim_2.repeat(1, 1, 1, N))

        if return_pi:
            return -max_reward, best_pi.squeeze()
        return -max_reward

    def train_dataloader(self):
        self.train_graph_size = self.cfg.n_clusters
        dataset = LSTSPCentroidDataset(
            lstsp_size=self.cfg.lstsp_size,
            n_clusters=self.cfg.n_clusters,
            node_dim=self.cfg.node_dim,
            num_samples=self.cfg.epoch_size,
            data_distribution=self.cfg.data_distribution)
        return DataLoader(dataset,
                          num_workers=os.cpu_count(),
                          batch_size=self.cfg.train_batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        dataset = LSTSPCentroidDataset(
            lstsp_size=self.cfg.graph_size,
            n_clusters=self.cfg.n_clusters,
            node_dim=self.cfg.node_dim,
            num_samples=self.cfg.val_size,
            data_distribution=self.cfg.data_distribution)
        return DataLoader(dataset,
                          batch_size=self.cfg.val_batch_size,
                          num_workers=os.cpu_count(),
                          pin_memory=True)

    def training_step(self, batch, _):
        B, N, _ = batch.shape
        G = self.cfg.group_size
        batch_idx_range = torch.arange(B)[:, None].expand(B, G)
        group_idx_range = torch.arange(G)[None, :].expand(B, G)
        env = Env(batch)
        s, r, d = env.reset(group_size=G)
        embeddings = self.encoder(batch)
        # return torch.nn.functional.mse_loss(embeddings, torch.rand_like(embeddings))
        self.decoder.reset(batch, embeddings, s.ninf_mask)
        group_prob_list = torch.zeros(B, G, 0, device=self.device)
        while not d:
            last_node_info = None
            if s.current_node is None:
                first_action = torch.randperm(
                    N, device=self.device)[None, :G].expand(B, G)
                s, r, d = env.step(first_action)
                continue
            else:
                last_node = s.current_node
            action_probs = self.decoder(last_node)
            action = action_probs.reshape(
                B * G, -1).multinomial(1).squeeze(dim=1).reshape(B, G)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while self.decoder.group_ninf_mask[batch_idx_range, group_idx_range,
                                               action].bool().any():
                action = action_probs.reshape(
                    B * G, -1).multinomial(1).squeeze(dim=1).reshape(B, G)
            chosen_action_prob = action_probs[batch_idx_range, group_idx_range,
                                              action].reshape(B, G)
            group_prob_list = torch.cat(
                (group_prob_list, chosen_action_prob[:, :, None]), dim=2)
            s, r, d = env.step(action)
        # Note that when G == 1, we can only use the PG without baseline so far
        advantage = r - r.mean(dim=1, keepdim=True) if G != 1 else r
        log_prob = group_prob_list.log().sum(dim=2)
        loss = (-advantage * log_prob).mean()
        length = -r.max(dim=1)[0].mean().clone().detach().item()
        self.log(
            name='length',
            value=length,
            prog_bar=True,
            # sync_dist=True
        )
        return {'loss': loss, 'length': length}

    def training_epoch_end(self, outputs):
        outputs = torch.as_tensor([item["length"] for item in outputs])
        self.train_length_mean = outputs.mean().item()
        self.train_length_std = outputs.std().item()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            max_reward = self.forward(batch)
        return max_reward

    def validation_epoch_end(self, outputs):
        self.val_length_mean = torch.cat(outputs).mean().item()
        self.val_length_std = torch.cat(outputs).std().item()

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "train_graph_size": self.train_graph_size,
                "train_length": self.train_length_mean,
                "val_length": self.val_length_mean,
            },
            # sync_dist=True,
            on_epoch=True,
            on_step=False)
        self.print(
            f'\nEpoch {self.current_epoch}: '
            f'train_graph_size={self.train_graph_size}, ',
            'train_performance={:.02f}±{:.02f}, '.format(
                self.train_length_mean, self.train_length_std),
            'val_performance={:.02f}±{:.02f}, '.format(self.val_length_mean,
                                                       self.val_length_std))


@hydra.main(config_name="config")
def run(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    cfg.run_name = cfg.run_name or cfg.default_run_name
    if cfg.save_dir is None:
        root_dir = os.getcwd(),
    elif os.path.isabs(cfg.save_dir):
        root_dir = cfg.save_dir
    else:
        root_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.save_dir)
    root_dir = os.path.join(root_dir, f'{cfg.run_name}')

    # build  LoopSolver
    loop_solver = LoopSolver(cfg)

    # build trainer
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        gpus=cfg.gpus,
        #  accelerator='ddp',
        #  sync_batchnorm=True,
        precision=cfg.precision,
        max_epochs=cfg.total_epoch,
        reload_dataloaders_every_epoch=True,
        num_sanity_val_steps=0)

    # wandb logger
    if cfg.wandb:
        os.makedirs(os.path.join(os.path.abspath(root_dir), 'wandb'))
        trainer.logger = WandbLogger(name=cfg.run_name,
                                     save_dir=root_dir,
                                     project=cfg.wandb_project,
                                     log_model=True,
                                     save_code=True,
                                     group=time.strftime(
                                         "%Y%m%d", time.localtime()),
                                     tags=cfg.default_run_name.split('-')[:-1])

    # training and save ckpt
    trainer.fit(loop_solver)
    trainer.save_checkpoint(
        os.path.join(hydra.utils.get_original_cwd(), 'pretrained_models',
                     'loop_solver_checkpoint.ckpt'))


if __name__ == "__main__":
    run()
