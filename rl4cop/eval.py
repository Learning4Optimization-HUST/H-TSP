import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from fast_pytorch_kmeans import KMeans
from tqdm import tqdm

from train_loop_solver import LoopSolver
from train_path_solver import PathSolver
from utils import TimeStat


class LSTSPSolver:
    def __init__(
        self,
        graph_size: int,
        n_clusters: int,
        loop_splver: pl.LightningModule,
        path_solver: pl.LightningModule,
    ):
        self.graph_size = graph_size
        self.n_clusters = n_clusters
        self.loop_splver = loop_splver
        self.path_solver = path_solver

    def __call__(self, instance, return_pi=False):
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=100)
        labels = kmeans.fit_predict(instance)
        centroids = kmeans.centroids
        clusters = [instance[labels == i] for i in range(self.n_clusters)]
        # get loop
        _, loop_pi = self.loop_splver(
            centroids[None], val_type="x8Aug_nTraj", return_pi=True
        )

        # get path
        source_node_indices = [None] * self.n_clusters
        target_node_indices = [None] * self.n_clusters
        bridge_distances = [None] * self.n_clusters
        for i in range(self.n_clusters):
            # maybe need remove selected node from `next_cluster` in last iteration
            i_prime = (i + 1) % self.n_clusters
            prev_cluster = clusters[loop_pi[i]]
            next_cluster = clusters[loop_pi[i_prime]]
            pairwised_distances = torch.cdist(prev_cluster, next_cluster)
            bridge_idx = [
                pairwised_distances.min(1).values.argmin(),
                pairwised_distances.min(0).values.argmin(),
            ]
            source_node_indices[i_prime] = bridge_idx[1][None]
            target_node_indices[i] = bridge_idx[0][None]
            prev_xy = prev_cluster[None, bridge_idx[0], :]
            next_xy = next_cluster[None, bridge_idx[1], :]
            bridge_distances[i] = pairwised_distances.min().item()

        path_distances = [None] * self.n_clusters
        path_tours = [None] * self.n_clusters
        for i in range(self.n_clusters):
            c = loop_pi[i]
            x = clusters[c]
            x1_min, _ = x[..., 0].min(dim=-1, keepdim=True)
            x2_min, _ = x[..., 1].min(dim=-1, keepdim=True)
            x1_max, _ = x[..., 0].max(dim=-1, keepdim=True)
            x2_max, _ = x[..., 1].max(dim=-1, keepdim=True)
            s = 1.0 / torch.maximum(x1_max - x1_min, x2_max - x2_min)
            x_new = torch.empty_like(x)
            x_new[..., 0] = s * (x[..., 0] - x1_min)
            x_new[..., 1] = s * (x[..., 1] - x2_min)
            length, path_tours[i] = self.path_solver(
                x_new[None],
                source_nodes=source_node_indices[i],
                target_nodes=target_node_indices[i],
                val_type="x8Aug_2Traj",
                return_pi=True,
            )
            d = x[path_tours[i]][None, :, :]
            path_distances[i] = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1).item()

        # calculate whole solution's length
        total_length = sum(bridge_distances) + sum(path_distances)
        if return_pi:
            return total_length, [loop_pi, path_tours]
        else:
            return total_length


@hydra.main(config_name="config", version_base="1.1")
def run(cfg) -> None:
    pl.seed_everything(1234)
    if os.path.isabs(cfg.pretrained_model_path):
        pretrained_model_path = cfg.pretrained_model_path
    else:
        pretrained_model_path = os.path.join(
            hydra.utils.get_original_cwd(), cfg.pretrained_model_path
        )
    path_solver = (
        PathSolver.load_from_checkpoint(
            f"{pretrained_model_path}/path_solver_checkpoint.ckpt"
        )
        .cuda()
        .eval()
    )
    loop_splver = (
        LoopSolver.load_from_checkpoint(
            f"{pretrained_model_path}/loop_solver_checkpoint.ckpt"
        )
        .cuda()
        .eval()
    )

    lstsp_solver = LSTSPSolver(
        cfg.lstsp_size, cfg.n_clusters, loop_splver=loop_splver, path_solver=path_solver
    )
    optima_dict = {"1000": 23.1182, "10000": 71.7778}
    optima = optima_dict[str(cfg.lstsp_size)]
    timer = TimeStat(cfg.n_lstsp_instances)
    instances = torch.rand(cfg.n_lstsp_instances, cfg.lstsp_size, 2).cuda()
    results = []
    iterations = tqdm(range(cfg.n_lstsp_instances))
    for i in iterations:
        with timer:
            total_length, whole_solution = lstsp_solver(instances[i], return_pi=True)
            results.append(total_length)
            current_mean_value = np.mean(results)
            current_gap = (current_mean_value - optima) / optima
            iterations.set_description(
                f"length_mean={current_mean_value:.4f}, gap={current_gap:.4%}"
            )
    print(np.mean(results), timer.mean * cfg.n_lstsp_instances, timer.mean)


if __name__ == "__main__":
    run()
