import os
import time

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader, Dataset

import os.path
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import models
import utils


class GroupState:
    def __init__(self, group_size, x, source, target):
        # x.shape = [B, N, 2]
        self.batch_size = x.size(0)
        self.graph_size = x.size(1)
        self.node_dim = x.size(2)
        self.group_size = group_size
        self.device = x.device
        # source.shape = target.shape = [B, G]
        self.source = source
        self.target = target

        self.selected_count = 0
        # current_node.shape = [B, G]
        self.current_node = None
        # selected_node_list.shape = [B, G, selected_count]
        self.selected_node_list = torch.zeros(
            x.size(0), group_size, 0, device=x.device
        ).long()
        # ninf_mask.shape = [B, G, N]
        self.ninf_mask = torch.zeros(x.size(0), group_size, x.size(1), device=x.device)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.selected_count += 1
        self.__move_to(selected_idx_mat)
        next_selected_idx_mat = self.__connect_source_target_city(selected_idx_mat)
        if (selected_idx_mat != next_selected_idx_mat).any():
            self.__move_to(next_selected_idx_mat)

    def __move_to(self, selected_idx_mat):
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected_idx_mat[:, :, None]), dim=2
        )
        self.mask(selected_idx_mat)

    def __connect_source_target_city(self, selected_idx_mat):
        source_idx = torch.where(selected_idx_mat == self.source)
        target_idx = torch.where(selected_idx_mat == self.target)
        next_selected_idx_mat = selected_idx_mat.clone()
        next_selected_idx_mat[source_idx] = self.target[source_idx]
        next_selected_idx_mat[target_idx] = self.source[target_idx]
        return next_selected_idx_mat

    def mask(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.ninf_mask.scatter_(
            dim=-1, index=selected_idx_mat[:, :, None], value=-torch.inf
        )


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

    def reset(self, group_size, source, target):
        self.group_size = group_size
        self.group_state = GroupState(
            group_size=group_size, x=self.x, source=source, target=target
        )
        self.fixed_edge_length = self._get_edge_length(source, target)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.selected_count == (self.graph_size - 1)
        if done:
            reward = -self._get_path_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_edge_length(self, source, target):
        idx_shp = (self.batch_size, self.group_size, 1, self.node_dim)
        coord_shp = (self.batch_size, self.group_size, self.graph_size, self.node_dim)
        source_idx = source[..., None, None].expand(*idx_shp)
        target_idx = target[..., None, None].expand(*idx_shp)
        fixed_edge_idx = torch.cat([source_idx, target_idx], dim=2)
        seq_expanded = self.x[:, None, :, :].expand(*coord_shp)
        ordered_seq = seq_expanded.gather(dim=2, index=fixed_edge_idx)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        delta = (ordered_seq - rolled_seq)[:, :, :-1, :]
        edge_length = (delta**2).sum(3).sqrt().sum(2)
        return edge_length

    def _get_path_distance(self) -> torch.Tensor:
        # selected_node_list.shape = [B, G, selected_count]
        interval = (
            torch.tensor([-1], device=self.x.device)
            .long()
            .expand(self.B, self.group_size)
        )
        selected_node_list = torch.cat(
            (self.group_state.selected_node_list, interval[:, :, None]),
            dim=2,
        ).flatten()
        unique_selected_node_list = selected_node_list.unique_consecutive()
        assert unique_selected_node_list.shape[0] == (
            self.B * self.group_size * (self.N + 1)
        ), unique_selected_node_list.shape
        unique_selected_node_list = unique_selected_node_list.view(
            [self.B, self.group_size, -1]
        )[..., :-1]
        shp = (self.B, self.group_size, self.N, self.C)
        gathering_index = unique_selected_node_list.unsqueeze(3).expand(*shp)
        seq_expanded = self.x[:, None, :, :].expand(*shp)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        delta = ordered_seq - rolled_seq
        tour_distances = (delta**2).sum(3).sqrt().sum(2)
        # minus the length of the fixed edge
        path_distances = tour_distances - self.fixed_edge_length
        return path_distances


class ClusterDataset(Dataset):
    def __init__(
        self, size=50, node_dim=2, num_samples=100000, data_distribution="uniform"
    ):
        super(ClusterDataset, self).__init__()
        if data_distribution == "uniform":
            self.data = torch.rand(num_samples, size, node_dim)
        elif data_distribution == "normal":
            self.data = torch.randn(num_samples, size, node_dim)
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class PathSolver(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        if cfg.node_dim > 2:
            assert (
                "noAug" in cfg.val_type
            ), "High-dimension TSP doesn't support augmentation"
        if cfg.encoder_type == "mha":
            self.encoder = models.MHAEncoder(
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                embedding_dim=cfg.embedding_dim,
                input_dim=cfg.node_dim,
                add_init_projection=cfg.add_init_projection,
            )
        else:
            if cfg.gnn_framework == "pyg":
                self.encoder = models.PYG_DeepGCNEncoder(
                    embedding_dim=cfg.embedding_dim,
                    input_e_dim=1,
                    input_h_dim=cfg.node_dim,
                    n_layers=cfg.n_layers,
                    n_neighbors=cfg.n_encoding_neighbors,
                    conv_type=cfg.pyg_conv_type,
                )
            else:
                self.encoder = models.DGL_ResGatedGraphEncoder(
                    embedding_dim=cfg.embedding_dim,
                    input_e_dim=1,
                    input_h_dim=cfg.node_dim,
                    n_layers=cfg.n_layers,
                    n_neighbors=cfg.n_encoding_neighbors,
                )
            assert cfg.precision == 32, "GNN can only handle float32 or float64"
        self.decoder = models.PathDecoder(
            embedding_dim=cfg.embedding_dim,
            n_heads=cfg.n_heads,
            tanh_clipping=cfg.tanh_clipping,
            n_decoding_neighbors=cfg.n_decoding_neighbors,
        )
        self.cfg = cfg
        self.save_hyperparameters(cfg)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate * len(self.cfg.gpus),
            weight_decay=self.cfg.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.99
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

    def forward(
        self,
        batch,
        source_nodes,
        target_nodes,
        val_type="x8Aug_2Traj",
        return_pi=False,
        group_size=2,
        greedy=True,
    ):
        val_type = val_type or self.cfg.val_type
        if "x8Aug" in val_type:
            batch = utils.augment_xy_data_by_8_fold(batch)
            source_nodes = torch.repeat_interleave(source_nodes, 8, dim=0)
            target_nodes = torch.repeat_interleave(target_nodes, 8, dim=0)

        B, N, _ = batch.shape
        # only support 1Traj or 2Traj
        # G = int(val_type[-5])
        # support larger Traj by sampling
        G = group_size
        assert G <= self.cfg.graph_size
        batch_idx_range = torch.arange(B)[:, None].expand(B, G)
        group_idx_range = torch.arange(G)[None, :].expand(B, G)

        source_action = source_nodes.view(B, 1).expand(B, G)
        target_action = target_nodes.view(B, 1).expand(B, G)

        env = Env(batch)
        s, r, d = env.reset(group_size=G, source=source_action, target=target_action)
        embeddings = self.encoder(batch)

        first_action = torch.randperm(N, device=self.device)[None, :G].expand(B, G)
        s, r, d = env.step(first_action)
        self.decoder.reset(
            batch, embeddings, s.ninf_mask, source_action, target_action, first_action
        )
        for _ in range(N - 2):
            action_probs = self.decoder(s.current_node)
            if greedy:
                action = action_probs.argmax(dim=2)
            else:
                action = (
                    action_probs.reshape(B * G, -1)
                    .multinomial(1)
                    .squeeze(dim=1)
                    .reshape(B, G)
                )
            #     # Check if sampling went OK, can go wrong due to bug on GPU
            #     # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            #     while self.decoder.group_ninf_mask[batch_idx_range, group_idx_range,
            #                                     action].bool().any():
            #         action = action_probs.reshape(
            #             B * G, -1).multinomial(1).squeeze(dim=1).reshape(B, G)
            s, r, d = env.step(action)
        interval = torch.tensor([-1], device=self.device).long().expand(B, G)
        selected_node_list = torch.cat(
            (s.selected_node_list, interval[:, :, None]),
            dim=2,
        ).flatten()
        unique_selected_node_list = selected_node_list.unique_consecutive()
        assert unique_selected_node_list.shape[0] == (
            B * G * (N + 1)
        ), unique_selected_node_list.shape
        pi = unique_selected_node_list.view([B, G, -1])[..., :-1]

        if val_type == "noAug_1Traj":
            max_reward = r
            best_pi = pi
        elif val_type == "noAug_nTraj":
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
            best_pi = best_pi.squeeze()
            return -max_reward, best_pi
        return -max_reward

    def train_dataloader(self):
        graph_size, random_range = self.cfg.graph_size, self.cfg.random_range
        graph_size_range = []
        for size in range(graph_size - random_range, graph_size + random_range + 1):
            if size % 10 == 0:
                graph_size_range.append(size)
        random_graph_size = np.random.choice(graph_size_range)
        self.train_graph_size = random_graph_size
        self.group_size = min(random_graph_size, self.cfg.group_size)
        dataset = ClusterDataset(
            size=random_graph_size,
            node_dim=self.cfg.node_dim,
            num_samples=self.cfg.epoch_size,
            data_distribution=self.cfg.data_distribution,
        )
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            batch_size=self.cfg.train_batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = ClusterDataset(
            size=self.cfg.graph_size,
            node_dim=self.cfg.node_dim,
            num_samples=self.cfg.val_size,
            data_distribution=self.cfg.data_distribution,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.val_batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def training_step(self, batch, _):
        B, N, _ = batch.shape
        G = self.group_size
        assert G <= self.cfg.graph_size
        batch_idx_range = torch.arange(B, device=self.device)[:, None].expand(B, G)
        group_idx_range = torch.arange(G, device=self.device)[None, :].expand(B, G)
        group_prob_list = torch.zeros(B, G, 0, device=self.device)

        source_nodes, target_nodes, _ = torch.split(
            tensor=torch.argsort(torch.rand(B, N, device=self.device)),
            split_size_or_sections=[1, 1, N - 2],
            dim=-1,
        )

        # we need manually set source node in env
        source_action = source_nodes.view(B, 1).expand(B, G)
        target_action = target_nodes.view(B, 1).expand(B, G)

        env = Env(batch)
        s, r, d = env.reset(group_size=G, source=source_action, target=target_action)
        embeddings = self.encoder(batch)

        first_action = torch.randperm(N, device=self.device)[None, :G].expand(B, G)
        s, r, d = env.step(first_action)

        self.decoder.reset(
            batch, embeddings, s.ninf_mask, source_action, target_action, first_action
        )
        for _ in range(N - 2):
            action_probs = self.decoder(s.current_node)
            action = (
                action_probs.reshape(B * G, -1)
                .multinomial(1)
                .squeeze(dim=1)
                .reshape(B, G)
            )
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            # while self.decoder.group_ninf_mask[batch_idx_range, group_idx_range,
            #                                    action].bool().any():
            #     action = action_probs.reshape(
            #         B * G, -1).multinomial(1).squeeze(dim=1).reshape(B, G)
            chosen_action_prob = action_probs[
                batch_idx_range, group_idx_range, action
            ].reshape(B, G)
            group_prob_list = torch.cat(
                (group_prob_list, chosen_action_prob[:, :, None]), dim=2
            )
            s, r, d = env.step(action)

        eps = torch.finfo(r.dtype).eps
        # Note that when G == 1, we can only use the PG without baseline so far
        if self.cfg.norm_reward:
            advantage = (
                (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-5)
                if G != 1
                else r
            )
        else:
            advantage = r - r.mean(dim=1, keepdim=True) if G != 1 else r
        log_prob = group_prob_list.log().sum(dim=2)
        entropy = -(log_prob.exp() * log_prob).mean()
        loss = (-advantage * log_prob).mean()
        length = -r.max(dim=1)[0].mean().clone().detach().item()
        self.log_dict(
            {
                "loss": loss,
                "advantage": advantage.mean(),
                "log_prob": log_prob.mean(),
                "entropy": entropy,
                "r_min": r.min(dim=1).values.mean(),
                "r_max": r.max(dim=1).values.mean(),
                "r_std": r.std(dim=1).mean(),
                "r_mean": r.mean(),
            },
        )
        self.log(
            name="length",
            value=length,
            prog_bar=True,
            # sync_dist=True
        )
        return {"loss": loss, "length": length}

    def training_epoch_end(self, outputs):
        outputs = torch.as_tensor([item["length"] for item in outputs])
        self.train_length_mean = outputs.mean().item()
        self.train_length_std = outputs.std().item()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            B, N, _ = batch.shape
            source_nodes = torch.zeros(B, 1, device=self.device).long()
            target_nodes = torch.ones(B, 1, device=self.device).long() * (N - 1)
            max_reward = self.forward(
                batch, source_nodes, target_nodes, group_size=self.group_size
            )
        return max_reward

    def validation_step_end(self, batch_parts):
        return batch_parts

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.log("result", torch.cat(outputs).mean())

    def validation_epoch_end(self, outputs):
        self.val_length_mean = torch.cat(outputs).mean().item()
        self.val_length_std = torch.cat(outputs).std().item()

    def on_train_epoch_end(self):
        self.print(f"memory used: {torch.cuda.max_memory_allocated() /1024 / 1024}")
        self.log_dict(
            {
                "train_graph_size": self.train_graph_size,
                "train_length": self.train_length_mean,
                "val_length": self.val_length_mean,
            },
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )
        self.print(
            f"\nEpoch {self.current_epoch}: "
            f"train_graph_size={self.train_graph_size}, ",
            "train_performance={:.02f}±{:.02f}, ".format(
                self.train_length_mean, self.train_length_std
            ),
            "val_performance={:.02f}±{:.02f}, ".format(
                self.val_length_mean, self.val_length_std
            ),
        )


@hydra.main(config_path="./", config_name="config", version_base="1.1")
def run(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    cfg.run_name = cfg.run_name or cfg.default_run_name

    if cfg.save_dir is None:
        root_dir = (os.getcwd(),)
    elif os.path.isabs(cfg.save_dir):
        root_dir = cfg.save_dir
    else:
        root_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.save_dir)
    root_dir = os.path.join(root_dir, f"{cfg.run_name}")

    # build  PathSolver
    path_solver = PathSolver(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_length",
        dirpath=os.path.join(root_dir, "checkpoints"),
        filename=cfg.encoder_type + "{epoch:02d}-{val_length:.2f}",
        save_last=True,
        save_top_k=3,
        mode="min",
        every_n_val_epochs=2,
    )

    # set up loggers
    tb_logger = TensorBoardLogger("logs")
    loggers = [tb_logger]
    # wandb logger
    if cfg.wandb:
        if os.getenv("NODE_RANK") == 0:
            os.makedirs(os.path.join(os.path.abspath(root_dir), "wandb"))
        wandb_logger = WandbLogger(
            name=cfg.run_name,
            save_dir=root_dir,
            project=cfg.wandb_project,
            log_model=True,
            save_code=True,
            group=time.strftime("%Y%m%d", time.localtime()),
            tags=cfg.default_run_name.split("-")[:-1],
        )
        wandb_logger.log_hyperparams(cfg)
        wandb_logger.watch(path_solver)
        loggers.append(wandb_logger)

    # auto resumed training
    last_ckpt_path = os.path.join(checkpoint_callback.dirpath, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        resume = last_ckpt_path
    else:
        resume = None

    # build trainer
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        gpus=cfg.gpus,
        strategy=DDPPlugin(find_unused_parameters=False),
        # detect_anomaly=True,
        #  sync_batchnorm=True,
        precision=cfg.precision,
        max_epochs=cfg.total_epoch,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        resume_from_checkpoint=resume,
        logger=loggers,
        callbacks=[checkpoint_callback],
    )

    # training and save ckpt
    trainer.fit(path_solver)
    trainer.save_checkpoint(
        os.path.join(root_dir, "pretrained_models", "path_solver_checkpoint.ckpt")
    )


if __name__ == "__main__":
    run()
