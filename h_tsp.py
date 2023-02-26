from argparse import ArgumentError
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from numba import jit
from omegaconf import DictConfig
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset

import rl_models as models
import rl_utils as utils
from rl4cop import train_path_solver
from tsp_opt_solver import lkh_solver

DISTANCE_SCALE = 1000

# ignore numba deprecation warning
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


class LowLevelSolver(ABC):
    @abstractmethod
    def solve(self, x: np.ndarray, fragment: List[int]):
        pass


class LKHSolver(LowLevelSolver):
    def solve(
        self, data: np.ndarray, fragment: np.ndarray, runs=2
    ) -> Tuple[List[List[int]], np.ndarray]:
        node_pos = np.take_along_axis(data, fragment[..., None], axis=1)
        assert node_pos.ndim == 3
        samples = []
        for fragment_pos in node_pos:
            samples.append(utils.nodes_to_sample(fragment_pos))

        with mp.Pool(processes=10) as pool:
            results = [
                pool.apply_async(lkh_solver, (sample, True, runs)) for sample in samples
            ]
            pool.close()
            pool.join()

        paths, lengths = [], []
        for res in results:
            route, dist = res.get()
            paths.append(route)
            lengths.append(dist)
        lengths = np.array(lengths)
        out_paths = []
        for i, path in enumerate(paths):
            assert path[0] == 0
            new_path = [fragment[i][j] for j in path[:-1]]
            out_paths.append(new_path)

        return out_paths, lengths


class GreedySolver(LowLevelSolver):
    def solve(self, x: np.ndarray, fragment: List[int]) -> Tuple[List[int], float]:
        if len(fragment) == 0:
            return []
        fragment_pos = x.take(fragment, axis=0)
        frag_len = len(fragment) - 1
        dist = cdist(fragment_pos, fragment_pos)
        dist_search = dist.copy()
        np.fill_diagonal(dist_search, np.inf)
        dist_search[:, frag_len] = np.inf
        dist_search[:, 0] = np.inf
        greedy_tour = [0]
        city_last = 0
        length = 0.0
        while len(greedy_tour) < frag_len:
            city_next = dist_search[city_last].argmin()
            length += dist[city_last, city_next]
            greedy_tour.append(city_next)
            dist_search[:, city_next] = np.inf
            city_last = city_next
        greedy_tour.append(frag_len)
        length += dist[city_last, frag_len]
        new_path = [fragment[i] for i in greedy_tour]

        return new_path, length


class FarthestInsertSolver(LowLevelSolver):
    def solve(self, x: np.ndarray, fragment: List[int]) -> Tuple[List[int], float]:
        if len(fragment) == 0:
            return []
        fragment_pos = x.take(fragment, axis=0)
        frag_len = len(fragment)

        available_nodes = set(range(frag_len))
        positions = fragment_pos
        tour = np.array([0, frag_len - 1])

        nodes_arr = np.ma.masked_array(list(available_nodes))
        best_distances = np.ma.masked_array(
            cdist(positions[nodes_arr], positions[tour], "euclidean").min(axis=1)
        )

        # We want the most distant node, so we get the max
        index_to_remove = best_distances.argmax()
        next_id = nodes_arr[index_to_remove]

        # Add the most distant point, as well as the first point to close the tour, we'll be inserting from here
        tour = np.insert(tour, 1, next_id)
        available_nodes.remove(0)
        available_nodes.remove(frag_len - 1)
        available_nodes.remove(next_id)
        nodes_arr[index_to_remove] = np.ma.masked
        best_distances[index_to_remove] = np.ma.masked

        # Takes two arrays of points and returns the array of distances
        def dist_arr(x1, x2):
            return np.sqrt(((x1 - x2) ** 2).sum(axis=1))

        # This is our selection method we will be using, it will give us the index in the masked array of the selected node,
        # the city id of the selected node, and the updated distance array.
        def get_next_insertion_node(nodes, positions, prev_id, best_distances):
            best_distances = np.minimum(
                cdist(
                    positions[nodes], positions[prev_id].reshape(-1, 2), "euclidean"
                ).min(axis=1),
                best_distances,
            )
            max_index = best_distances.argmax()
            return max_index, nodes[max_index], best_distances

        while len(available_nodes) > 0:
            index_to_remove, next_id, best_distances = get_next_insertion_node(
                nodes_arr, positions, next_id, best_distances
            )

            # Finding the insertion point
            c_ik = cdist(positions[tour[:-1]], positions[next_id].reshape(-1, 2))
            c_jk = cdist(positions[tour[1:]], positions[next_id].reshape(-1, 2))
            c_ij = dist_arr(positions[tour[:-1]], positions[tour[1:]]).reshape(-1, 1)
            i = (c_ik + c_jk - c_ij).argmin()

            tour = np.insert(tour, i + 1, next_id)

            available_nodes.remove(next_id)
            nodes_arr[index_to_remove] = np.ma.masked
            best_distances[index_to_remove] = np.ma.masked

        new_path = [fragment[i] for i in tour]

        return new_path, None


class RLSolver(LowLevelSolver):
    def __init__(self, low_level_model: nn.Module, sample_size: int = 200) -> None:
        self._solver_model = low_level_model
        self._sample_size = max(sample_size, 2)

    def solve(
        self,
        data: torch.Tensor,
        fragment: torch.Tensor,
        frag_buffer: utils.FragmengBuffer,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        node_pos = torch.gather(
            input=data, index=fragment[..., None].expand(-1, -1, data.shape[-1]), dim=1
        )
        assert node_pos.ndim == 3
        device = self._solver_model.device
        B = node_pos.shape[0]
        N = node_pos.shape[1]
        x = node_pos.to(device=device)
        x1_min = x[..., 0].min(dim=-1, keepdim=True)[0]
        x2_min = x[..., 1].min(dim=-1, keepdim=True)[0]
        x1_max = x[..., 0].max(dim=-1, keepdim=True)[0]
        x2_max = x[..., 1].max(dim=-1, keepdim=True)[0]
        s = 0.9 / torch.maximum(x1_max - x1_min, x2_max - x2_min)
        x_new = torch.empty_like(x)
        x_new[..., 0] = s * (x[..., 0] - x1_min) + 0.05
        x_new[..., 1] = s * (x[..., 1] - x2_min) + 0.05
        frag_buffer.update_buffer(x_new.cpu())
        source_nodes = torch.tensor([[0]], device=device).expand(B, 1)
        target_nodes = torch.tensor([[N - 1]], device=device).expand(B, 1)
        with amp.autocast(enabled=False) as a, torch.no_grad() as b, utils.evaluating(
            self._solver_model
        ) as solver:
            lengths, paths = solver(
                x_new.float(),
                source_nodes=source_nodes,
                target_nodes=target_nodes,
                val_type="x8Aug_2Traj",
                return_pi=True,
                group_size=self._sample_size,
            )
        lengths = (lengths / s.squeeze()).detach().cpu().numpy()

        # Flip and construct tour on original graph
        out_paths = []
        paths = paths.detach().cpu().numpy()
        fragment = fragment.cpu().numpy()
        if B == 1:
            paths = [paths]
        for i, path in enumerate(paths):
            # make sure path start with `0` and end with `N-1`
            zero_idx = np.nonzero(path == 0)[0][0]
            path = np.roll(path, -zero_idx)
            assert path[0] == 0, f"{zero_idx=}, {path=}"
            if path[-1] != N - 1:
                path = np.roll(path, -1)
                path = np.flip(path)
            assert path[-1] == N - 1, f"{path=}"
            assert np.unique(path).shape[0] == path.shape[0] == N
            assert path.shape[0] == fragment[i].shape[0] == N
            new_path = [fragment[i][j] for j in path]
            out_paths.append(new_path)

        return out_paths, lengths


class LargeState:
    def __init__(
        self,
        x: torch.Tensor,
        k: int,
        init_tour: List[int],
        dist_matrix: torch.Tensor = None,
        knn_neighbor: torch.Tensor = None,
    ) -> None:
        # for now only support single graph
        assert x.ndim == 2
        assert isinstance(x, torch.Tensor)
        self.x = x
        self.device = x.device
        self.graph_size = x.shape[0]
        # k for k-Nearest-Neighbor
        self.k = k

        if dist_matrix is None:
            self.dist_matrix = dist_matrix = torch.cdist(
                x.type(torch.float64) * DISTANCE_SCALE,
                x.type(torch.float64) * DISTANCE_SCALE,
            ).type(torch.float32)
        else:
            assert isinstance(dist_matrix, torch.Tensor)
            self.dist_matrix = dist_matrix

        if knn_neighbor is None:
            self.knn_neighbor = self.dist_matrix.topk(
                k=self.k + 1, largest=False
            ).indices[:, 1:]
        else:
            assert isinstance(knn_neighbor, torch.Tensor)
            self.knn_neighbor = knn_neighbor

        self.numpy_knn_neighbor = self.knn_neighbor.cpu().numpy()
        assert len(init_tour) == 2 or len(init_tour) == 0
        # mask.shape = [N]
        self.selected_mask = torch.zeros(
            x.shape[0], dtype=torch.bool, device=self.device
        )
        self.available_mask = torch.zeros(
            x.shape[0], dtype=torch.bool, device=self.device
        )
        self.neighbor_coord = torch.zeros(
            (x.shape[0], 4), dtype=torch.float32, device=self.device
        )

        # start with a 2-city-tour or empty tour
        self.current_tour = init_tour.copy()
        self.current_num_cities = len(self.current_tour)
        # to make final return equals to tour length
        self.current_tour_len = (
            utils.get_tour_distance(init_tour, self.dist_matrix) / DISTANCE_SCALE
        )
        # self.current_tour_len = 0.0
        utils.update_neighbor_coord_(self.neighbor_coord, self.current_tour, self.x)
        self.mask(self.current_tour)

    def move_to(self, new_path: List[int]) -> None:
        """state transition given current state (partial tour) and action (new path)"""
        if self.current_num_cities == 0:
            self.current_tour = new_path
        else:
            start_idx = self.current_tour.index(new_path[0])
            end_idx = self.current_tour.index(new_path[-1])
            assert start_idx != end_idx, new_path
            # assuming we always choose a fragment from left to right
            # replace the old part in current tour with the new path, there is a city-overlap between them
            if end_idx > start_idx:
                self.current_tour = (
                    self.current_tour[:start_idx]
                    + new_path
                    + self.current_tour[end_idx + 1 :]
                )
            else:
                self.current_tour = (
                    self.current_tour[end_idx + 1 : start_idx] + new_path
                )
        # make sure no duplicate cities
        assert np.unique(self.current_tour).shape[0] == len(
            self.current_tour
        ), f"{self.current_tour}"
        # update info of current tour
        if len(self.current_tour) < self.graph_size:
            assert (
                len(self.current_tour) > self.current_num_cities
            ), f"{len(self.current_tour)}-{self.current_num_cities}-{len(new_path)}"
        else:
            assert (
                len(self.current_tour) == self.graph_size
            ), f"{len(self.current_tour)}-{self.graph_size}"

        self.current_num_cities = len(self.current_tour)
        self.current_tour_len = utils.get_tour_distance(
            self.current_tour, self.dist_matrix
        )
        utils.update_neighbor_coord_(self.neighbor_coord, self.current_tour, self.x)
        # update two masks
        self.mask(new_path)

    def mask(self, new_path: List[int]) -> None:
        """update mask status w.r.t new path"""
        self.selected_mask[new_path] = True
        # update mask of available starting cities for k-NN process
        self.available_mask[new_path] = True
        avaliable_cities = torch.where(self.available_mask == True)
        available_status = torch.logical_not(
            self.selected_mask[self.knn_neighbor[avaliable_cities]].all(dim=1)
        )
        self.available_mask[avaliable_cities] = available_status

    def get_nearest_cluster_city_idx(
        self, old_cities: List[int], new_cities: List[int]
    ) -> int:
        """find the index of the old city nearest to given new cities"""
        assert old_cities
        assert new_cities
        city_idx = utils.get_nearest_cluster_city_idx(
            self.dist_matrix, old_cities, new_cities
        )

        return city_idx

    def get_nearest_old_city_idx(self, predict_coord: torch.Tensor) -> int:
        """find the index of the new city nearest to given coordinates"""
        assert predict_coord.shape == (2,)
        city_idx = utils.get_nearest_city_idx(self.x, predict_coord, self.selected_mask)

        return city_idx

    def get_nearest_new_city_idx(self, predict_coord: torch.Tensor) -> int:
        """find the index of the new city nearest to given coordinates"""
        assert predict_coord.shape == (2,)
        city_idx = utils.get_nearest_city_idx(
            self.x, predict_coord, ~self.selected_mask
        )

        return city_idx

    def get_nearest_new_city_coord(self, predict_coord: torch.Tensor) -> int:
        """find the coordinate of the available city nearest to given coordinates"""

        city_idx = self.get_nearest_new_city_idx(predict_coord)
        city_coord = self.x[city_idx]
        assert city_coord.shape == (2,)

        return city_coord

    def get_nearest_avail_city_idx(self, predict_coord: torch.Tensor) -> int:
        """find the index of the available city nearest to given coordinates"""
        assert predict_coord.shape == (2,)
        city_idx = utils.get_nearest_city_idx(
            self.x, predict_coord, self.available_mask
        )

        return city_idx

    def get_nearest_avail_city_coord(self, predict_coord: torch.Tensor) -> int:
        """find the coordinate of the available city nearest to given coordinates"""

        city_idx = self.get_nearest_avail_city_idx(predict_coord)
        city_coord = self.x[city_idx]
        assert city_coord.shape == (2,)

        return city_coord

    def to_tensor(self) -> torch.Tensor:
        """concatenate `x`, `selected_mask` to produce an state"""
        available_mask = self.available_mask
        return torch.cat(
            [
                self.x,
                self.selected_mask[:, None],
                available_mask[:, None],
                self.neighbor_coord,
            ],
            dim=1,
        )

    def to_device_(self, device: Optional[Union[torch.device, str]] = None):
        self.x = self.x.to(device)
        self.device = self.x.device
        self.dist_matrix = self.dist_matrix.to(device)
        self.knn_neighbor = self.knn_neighbor.to(device)
        self.selected_mask = self.selected_mask.to(device)
        self.available_mask = self.available_mask.to(device)
        self.neighbor_coord = self.neighbor_coord.to(device)

    @classmethod
    def from_numpy(cls, numpy_state: np.ndarray):
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"Large Scale TSP state, num_cities: {self.x.shape[0]},"
            f"current_tour: {len(self.current_tour)} - {self.current_tour_len}"
        )


class Env:
    def __init__(
        self,
        k: int,
        frag_len: int = 200,
        max_new_nodes: int = 160,
        max_improvement_step: int = 5,
    ):
        self.k = k
        # length of the fragment sent to low-level agent
        self.frag_len = frag_len
        self.max_new_nodes = max_new_nodes
        self.max_improvement_step = max_improvement_step

    def reset(self, x: torch.Tensor, no_depot=False) -> Tuple[LargeState, float, bool]:
        self.x = x.type(torch.float32)
        self.device = x.device
        self.graph_size = self.N = x.shape[0]
        self.node_dim = self.C = x.shape[1]
        # for numerical stability
        dist_matrix = torch.cdist(
            x.type(torch.float64) * DISTANCE_SCALE,
            x.type(torch.float64) * DISTANCE_SCALE,
        ).type(torch.float32)
        dist_matrix.fill_diagonal_(float("inf"))
        knn_neighbor = dist_matrix.topk(k=self.k, largest=False).indices
        if no_depot:
            init_tour = []
        else:
            depot = 0
            init_tour = [depot, knn_neighbor[depot][0].item()]
        self.state = LargeState(
            x=self.x,
            k=self.k,
            init_tour=init_tour,
            dist_matrix=dist_matrix,
            knn_neighbor=knn_neighbor,
        )
        self._improvement_steps = 0

        return self.state.to_tensor()

    @staticmethod
    @jit(nopython=True)
    def append_selected(
        neighbor: np.ndarray, selected_set: Set[int], selected_mask: np.ndarray
    ):
        result = []
        for n in neighbor:
            if n not in selected_set:
                if not selected_mask[n]:
                    result.append(n)
        return result

    def get_fragment_knn(
        self, predict_coord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # early return if already done
        if self.done:
            return []
        assert predict_coord.ndim == 1
        assert predict_coord.min() >= 0
        assert predict_coord.min() <= 1
        if not self._construction_done:
            # choose starting city based on the model predicted coordinate
            nearest_new_city = self.state.get_nearest_new_city_idx(predict_coord)
            if self.state.current_num_cities != 0:
                start_city = self.state.get_nearest_old_city_idx(
                    self.x[nearest_new_city]
                )
                new_city_start = self.state.get_nearest_new_city_idx(self.x[start_city])
            else:
                start_city = None
                new_city_start = nearest_new_city
            # search for new cities with k-NN heuristic
            max_cities = max(
                self.max_new_nodes, self.frag_len - self.state.current_num_cities
            )
            new_cities = []
            knn_deque = []
            selected_set = set()
            knn_deque.append(new_city_start)
            selected_set.add(new_city_start)
            selected_mask = self.state.selected_mask.cpu().numpy()
            while len(knn_deque) != 0 and len(new_cities) < max_cities:
                node = knn_deque.pop(0)
                new_cities.append(node)
                res = self.append_selected(
                    self.state.numpy_knn_neighbor[node], selected_set, selected_mask
                )
                knn_deque.extend(res)
                selected_set.update(res)
        else:
            # refine a old fragment
            assert not self._improvement_done
            nearest_old_city = self.state.get_nearest_old_city_idx(predict_coord)
            new_cities = []
        # extend fragment to `frag_len` with some old cities
        if self.state.current_num_cities != 0:
            fragment = self._extend_fragment(start_city, new_cities)
        else:
            fragment = new_cities

        for i in fragment:
            if i in new_cities:
                assert not self.state.selected_mask[i]
            else:
                assert self.state.selected_mask[i]

        return (
            torch.tensor(fragment, device=self.device),
            torch.tensor(new_cities, device=self.device),
            self.unscale_action(self.x[start_city]),
        )

    def step(
        self,
        predict_coord: np.ndarray,
        solver: LowLevelSolver,
        greedy_reward: bool = False,
        average_reward: bool = False,
        float_available_status=False,
    ) -> Tuple[LargeState, float, bool]:
        # raise exception if already done
        if self.done:
            raise RuntimeError("Environment has terminated!")

        # get fragment
        predict_coord = self.scale_action(predict_coord)
        fragment, _ = self.get_fragment_knn(predict_coord)
        # solve subproblem
        if isinstance(solver, RLSolver) or isinstance(solver, LKHSolver):
            new_paths, _ = solver.solve(self.x[None, ...], fragment[None, ...])
            new_path = new_paths[0]
        else:
            new_path = solver.solve(self.x, fragment)
        return self._step(
            new_path, greedy_reward, average_reward, float_available_status
        )

    def _step(
        self,
        new_path: List[int],
        greedy_reward: bool = False,
        average_reward: bool = False,
    ) -> Tuple[LargeState, float, bool]:
        """step function for VecEnv"""
        # raise exception if already done
        if self.done:
            raise RuntimeError("Environment has terminated!")
        if self._construction_done and not self._improvement_done:
            self._improvement_steps += 1
        # record old states
        old_len = self.state.current_tour_len
        old_num_cities = self.state.current_num_cities
        # update state
        self.state.move_to(new_path)
        self.state.current_tour_len = (
            utils.get_tour_distance(self.state.current_tour, self.state.dist_matrix)
            / DISTANCE_SCALE
        )
        if greedy_reward:
            len_rl = (
                utils.get_tour_distance(new_path, self.state.dist_matrix)
                - self.state.dist_matrix[new_path[0], new_path[-1]]
            ) / DISTANCE_SCALE
            _, len_greedy = GreedySolver().solve(self.x, new_path)
            reward = len_greedy - len_rl
        else:
            reward = old_len - self.state.current_tour_len
        if average_reward:
            added_num_cities = self.state.current_num_cities - old_num_cities
            reward /= added_num_cities

        return self.state.to_tensor(), reward, self.done

    @property
    def _construction_done(self):
        return self.state.current_num_cities == self.graph_size

    @property
    def _improvement_done(self):
        return self._improvement_steps >= self.max_improvement_step

    @property
    def done(self):
        return self._construction_done and self._improvement_done

    @staticmethod
    def scale_action(a):
        """scale action from [-1, 1] to [0, 1]"""
        return a * 0.5 + 0.5

    @staticmethod
    def unscale_action(a):
        """unscale action from [0, 1] to [-1, 1]"""
        return a * 2 - 1

    def random_action(self):
        action = torch.randn(size=(self.node_dim,), device=self.device)
        return self.unscale_action(action)

    def heuristic_action(self):
        non_selected_idx = torch.where(self.state.selected_mask == False)[0]
        random_choice = np.random.randint(0, non_selected_idx.shape[0])
        new_idx = non_selected_idx[random_choice]
        action = self.x[new_idx]
        return self.unscale_action(action)

    def _extend_fragment(self, nearest_city: int, new_cities: List[int]):
        nearest_idx = self.state.current_tour.index(nearest_city)
        total_extend_len = self.frag_len - len(new_cities)
        assert total_extend_len > 0, total_extend_len
        offset = nearest_idx - (total_extend_len // 2)
        reorder_tour = (
            self.state.current_tour[offset:] + self.state.current_tour[:offset]
        )
        assert len(reorder_tour) == len(self.state.current_tour)

        fragment = (
            reorder_tour[: (total_extend_len // 2)]
            + new_cities
            + reorder_tour[(total_extend_len // 2) : total_extend_len]
        )

        assert len(fragment) == total_extend_len + len(
            new_cities
        ), f"{len(fragment)}-{total_extend_len}-{len(new_cities)}"

        assert len(fragment) == self.frag_len, len(fragment)
        assert np.unique(fragment).shape[0] == len(fragment), np.unique(fragment).shape

        return fragment

    def to_device_(self, device: Optional[Union[torch.device, str]] = None):
        self.x = self.x.to(device)
        self.device = self.x.device
        self.state.to_device_(device)

    def debug_check_tour_len(self):
        assert self.done, self.done
        l = 0
        tour = self.state.current_tour
        dist_matrix = self.state.dist_matrix.cpu().squeeze().numpy()
        assert len(tour) == self.graph_size, f"{len(tour)=}, {self.graph_size=}"
        assert np.unique(tour).shape[0] == self.graph_size, f"{np.unique(tour).shape=}"
        for i in range(self.graph_size - 1):
            l += dist_matrix[tour[i], tour[i + 1]]
        l += dist_matrix[tour[-1], tour[0]]
        l /= DISTANCE_SCALE
        assert (
            abs(l - self.state.current_tour_len) < 1e-3
        ), f"{l=}, {self.state.current_tour_len=}"


class VecEnv:
    def __init__(
        self,
        k: int,
        frag_len: int = 200,
        max_new_nodes: int = 160,
        max_improvement_step: int = 5,
        auto_reset=False,
        no_depot=False,
    ):
        self.k = k
        self.frag_len = frag_len
        self.max_new_nodes = max_new_nodes
        self.max_improvement_step = max_improvement_step
        self.auto_reset = auto_reset
        self.no_depot = no_depot

    def reset(self, x: torch.Tensor):
        self.x = x
        self.device = x.device
        self.batch_size = self.B = x.shape[0]
        self.graph_size = self.N = x.shape[1]
        self.node_dim = self.C = x.shape[2]
        self.envs = []
        for _ in range(self.batch_size):
            self.envs.append(
                Env(
                    self.k, self.frag_len, self.max_new_nodes, self.max_improvement_step
                )
            )

        states = [self.envs[i].reset(x[i], self.no_depot) for i in range(self.B)]
        self.states = states

        return torch.stack(states).type(torch.float32)

    @property
    def done(self):
        return all(e.done for e in self.envs)

    def step(
        self,
        predict_coords: torch.Tensor,
        solver: LowLevelSolver,
        greedy_reward: bool = False,
        average_reward: bool = False,
        float_available_status=False,
        tsp_data: Callable = None,
        frag_buffer: utils.FragmengBuffer = None,
        log: Callable = None,
    ):
        if self.done:
            raise RuntimeError("Environment has terminated!")
        assert predict_coords.ndim == 2
        assert predict_coords.shape[0] == self.batch_size
        actions = [Env.scale_action(coord) for coord in predict_coords]

        # get active indecies
        if not self.auto_reset:
            active_idx = [i for i in range(self.B) if not self.envs[i].done]
            active_envs = [self.envs[i] for i in active_idx]
            active_acts = [actions[i] for i in active_idx]
        else:
            active_idx = list(range(self.B))
            active_envs = self.envs
            active_acts = actions

        # get fragment of cities with model predicted coordinate
        fragments = []
        new_cities = []
        start_cities = []
        for env, a in zip(active_envs, active_acts):
            frag, new_city, start_city = env.get_fragment_knn(
                a,
            )
            fragments.append(frag)
            new_cities.append(new_city)
            start_cities.append(start_city)

        # solve the fragment of cities to get a new path
        if isinstance(solver, RLSolver):
            new_paths, _ = solver.solve(
                self.x[active_idx], torch.stack(fragments), frag_buffer
            )
        elif isinstance(solver, LKHSolver):
            np_fragments = [frag.cpu().numpy() for frag in fragments]
            new_paths, _ = solver.solve(
                self.x[active_idx].cpu().numpy(), np.stack(np_fragments)
            )
        else:
            new_paths = []
            for env, frag in zip(self.envs, fragments):
                res = solver.solve(env.x, frag)
                new_paths.append(res)
        # env update states and rewards given the new path
        outputs = []
        for env, p in zip(active_envs, new_paths):
            res = env._step(p, greedy_reward, average_reward)
            outputs.append(res)

        if not self.auto_reset:
            states = [None] * self.B
            rewards = [0.0] * self.B
            dones = [False] * self.B
            count = 0
            for i in range(self.B):
                if i in active_idx:
                    states[i] = outputs[count][0]
                    rewards[i] = outputs[count][1]
                    dones[i] = outputs[count][2]
                    count += 1
            assert count == len(active_idx), f"{count}!={len(active_idx)}"
        else:
            states = [output[0] for output in outputs]
            rewards = [output[1] for output in outputs]
            dones = [output[2] for output in outputs]
            for i in range(self.B):
                env = self.envs[i]
                if env.done:
                    log(
                        "explore/tour_length",
                        env.state.current_tour_len.item(),
                        on_step=True,
                    )
                    data = tsp_data().squeeze().to(self.device)
                    state = env.reset(data, self.no_depot)
                    self.x[i] = data
                    states[i] = state

        for i in active_idx:
            self.states[i] = states[i]

        return (
            torch.stack(self.states).type(torch.float32),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
            {
                "fragments": fragments,
                "new_cities": new_cities,
                "start_city": torch.stack(
                    start_cities,
                ),
            },
        )

    def random_action(self):
        return Env.unscale_action(torch.randn(size=(self.batch_size, self.node_dim)))

    def heuristic_action(self):
        return torch.stack(
            [
                env.heuristic_action() if not env.done else env.random_action()
                for env in self.envs
            ],
            dim=0,
        )

    def to_device_(self, device: Optional[Union[torch.device, str]] = None):
        for env in self.envs:
            env.to_device_(device)


def readDataFile(filePath):
    """
    read validation dataset from "https://github.com/Spider-scnu/TSP"
    """
    res = []
    with open(filePath, "r") as fp:
        datas = fp.readlines()
        for data in datas:
            data = [float(i) for i in data.split("o")[0].split()]
            loc_x = torch.FloatTensor(data[::2])
            loc_y = torch.FloatTensor(data[1::2])
            data = torch.stack([loc_x, loc_y], dim=1)
            res.append(data)
    res = torch.stack(res, dim=0)
    return res


class TSPDataset(Dataset):
    def __init__(
        self,
        size=50,
        node_dim=2,
        num_samples=100000,
        data_distribution="uniform",
        data_path=None,
    ):
        super(TSPDataset, self).__init__()
        if data_distribution == "uniform":
            self.data = torch.rand(num_samples, size, node_dim)
        elif data_distribution == "normal":
            self.data = torch.randn(num_samples, size, node_dim)
        self.size = num_samples
        if not data_path is None:
            self.data = readDataFile(data_path)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class DummyDataset(Dataset):
    """Generate a dummy dataset.
    Example:
        >>> from pl_bolts.datasets import DummyDataset
        >>> from torch.utils.data import DataLoader
        >>> # mnist dims
        >>> ds = DummyDataset((1, 28, 28), (1, ))
        >>> dl = DataLoader(ds, batch_size=7)
        >>> # get first batch
        >>> batch = next(iter(dl))
        >>> x, y = batch
        >>> x.size()
        torch.Size([7, 1, 28, 28])
        >>> y.size()
        torch.Size([7, 1])
    """

    def __init__(self, *shapes, num_samples: int = 10000):
        """
        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset
        """
        super().__init__()
        self.shapes = shapes
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        sample = []
        for shape in self.shapes:
            spl = torch.rand(*shape)
            sample.append(spl)
        return sample


def update_buffer(buffer, traj_list):
    cur_items = list(map(list, zip(*traj_list)))
    cur_items = [torch.cat(item, dim=0) for item in cur_items]
    buffer[:] = cur_items

    steps, r_exp = len(buffer[1]), buffer[1].mean().item()

    return steps, r_exp


class HTSP_PPO(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.low_level_model = train_path_solver.PathSolver.load_from_checkpoint(
            cfg.low_level_load_path
        )
        self.low_level_solver = RLSolver(
            self.low_level_model, cfg.low_level_sample_size
        )
        # self.low_level_solver = LKHSolver()

        if cfg.encoder_type == "pixel":
            self.encoder = models.IMPALAEncoder(
                input_dim=cfg.input_dim,
                embedding_dim=cfg.embedding_dim,
            )
            self.encoder_target = models.IMPALAEncoder(
                input_dim=cfg.input_dim,
                embedding_dim=cfg.embedding_dim,
            )
        else:
            raise TypeError(f"Encoder type {cfg.encoder_type} not supported!")
        self.actor = models.ActorPPO(
            state_dim=cfg.embedding_dim,
            mid_dim=cfg.acotr_mid_dim,
            action_dim=cfg.nb_actions,
            init_a_std_log=cfg.init_a_std_log,
        )
        self.critic = models.CriticPPO(
            state_dim=cfg.embedding_dim,
            mid_dim=cfg.hidden_dim,
            _action_dim=cfg.nb_actions,
        )
        self.critic_target = models.CriticPPO(
            state_dim=cfg.embedding_dim,
            mid_dim=cfg.hidden_dim,
            _action_dim=cfg.nb_actions,
        )
        utils.hard_update(self.critic_target, self.critic)
        utils.hard_update(self.encoder_target, self.encoder)

        self.mse_loss = nn.MSELoss()
        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.MSELoss()
        self.lambda_entropy = cfg.lambda_entropy

        self.cfg = cfg
        self.save_hyperparameters(cfg)

        # Memory
        self.memory = []
        self.traj_list = [
            [[] for _ in range(cfg.experience_items)] for _ in range(cfg.env_num)
        ]
        self.frag_buffer = utils.FragmengBuffer(
            cfg.low_level_buffer_size, cfg.frag_len, cfg.node_dim
        )
        self.val_frag_buffer = utils.FragmengBuffer(
            cfg.low_level_buffer_size, cfg.frag_len, cfg.node_dim
        )

        # Environment
        self.env_maker = lambda auto_reset=False, no_depot=False: VecEnv(
            k=cfg.k,
            frag_len=cfg.frag_len,
            max_new_nodes=cfg.max_new_nodes,
            max_improvement_step=cfg.max_improvement_step,
            auto_reset=auto_reset,
            no_depot=no_depot,
        )
        self.env = self.env_maker(True, self.cfg.no_depot)
        self.states = None
        self.__tsp_data_epoch = None
        self.__tsp_used_times = 0
        # Optimizer
        self.automatic_optimization = False

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if "target" not in name:
                module.train(mode)
        return self

    def configure_optimizers(self):
        optim_actor = torch.optim.AdamW(
            [
                {"params": self.actor.parameters(), "lr": self.cfg.lr_actor},
                {"params": self.encoder.parameters(), "lr": self.cfg.lr_enc},
            ],
            weight_decay=self.cfg.weight_decay,
        )

        optim_critic = torch.optim.AdamW(
            [
                {"params": self.critic.parameters(), "lr": self.cfg.lr_critic},
                {"params": self.encoder.parameters(), "lr": self.cfg.lr_enc},
            ],
            weight_decay=self.cfg.weight_decay,
        )

        optim_low_level = torch.optim.AdamW(
            params=self.low_level_model.parameters(),
            lr=self.cfg.low_level_lr,
            weight_decay=self.cfg.weight_decay,
        )

        sche_actor = torch.optim.lr_scheduler.StepLR(optim_actor, step_size=2, gamma=1)
        sche_critic = torch.optim.lr_scheduler.StepLR(
            optim_critic, step_size=2, gamma=1
        )
        sche_low_level = torch.optim.lr_scheduler.StepLR(
            optim_low_level, step_size=2, gamma=1
        )

        return [optim_actor, optim_critic, optim_low_level], [
            sche_actor,
            sche_critic,
            sche_low_level,
        ]

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action given a state.

        :param state: a state in a shape (state_dim, ).
        :return: a action in a shape (action_dim, ) where each action is clipped into range(-1, 1).
        """
        with torch.no_grad():
            s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
            graph_feat = self.encoder(s_tensor)
            a_tensor = self.actor(graph_feat)
        action = a_tensor.detach().cpu().numpy()
        return action.tanh()

    def select_actions(self, state: torch.Tensor) -> tuple:
        """
        Select actions given an array of states.

        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        with torch.no_grad():
            state = state.to(self.device)
            graph_feat = self.encoder(state)
            action, noise = self.actor.get_action(graph_feat)
        return action.detach(), noise.detach()

    def get_critic_value(
        self, state: torch.Tensor, is_target: bool = False
    ) -> torch.Tensor:
        if is_target:
            graph_feat = self.encoder_target(state)
            value = self.critic_target(graph_feat)
        else:
            graph_feat = self.encoder(state)
            value = self.critic(graph_feat)
        return value

    @torch.no_grad()
    def explore_vec_env(self, target_steps: int) -> list:
        traj_list = []
        experience = []
        # initialize env
        if self.states is None:
            tsp_data = torch.cat(
                [self.tsp_data() for _ in range(self.cfg.env_num)], dim=0
            ).to(self.device)
            self.states = self.env.reset(tsp_data)

        self.env.to_device_(self.device)
        step = 0
        ten_s = self.states
        last_done = torch.zeros(self.cfg.env_num, dtype=torch.int, device=self.device)
        while step < target_steps:
            ten_a, ten_n = self.select_actions(ten_s)
            ten_s_next, ten_rewards, ten_dones, info = self.env.step(
                ten_a.tanh(),
                self.low_level_solver,
                self.cfg.greedy_reward,
                self.cfg.average_reward,
                self.cfg.float_available_status,
                self.tsp_data,
                self.frag_buffer,
                self.log,
            )
            traj_list.append(
                (
                    ten_s.clone(),
                    ten_rewards.clone(),
                    ten_dones.clone(),
                    ten_a,
                    ten_n,
                    info["start_city"],
                )
            )
            ten_s = ten_s_next

            step += 1
            last_done[torch.where(ten_dones)[0]] = step  # behind `step+=1`

        self.states = ten_s

        buf_srdan = list(map(list, zip(*traj_list)))
        assert len(buf_srdan) == self.cfg.experience_items
        del traj_list[:]

        buf_srdan[0] = torch.stack(buf_srdan[0])
        buf_srdan[1] = (torch.stack(buf_srdan[1]) * self.cfg.reward_scale).unsqueeze(2)
        buf_srdan[2] = ((1 - torch.stack(buf_srdan[2])) * self.cfg.gamma).unsqueeze(2)
        buf_srdan[3] = torch.stack(buf_srdan[3])
        buf_srdan[4] = torch.stack(buf_srdan[4])
        buf_srdan[5] = torch.stack(buf_srdan[5])

        experience.append(self.splice_trajectory(buf_srdan, last_done)[0])
        self.env.to_device_("cpu")
        return experience

    def get_reward_sum_gae(
        self, buf_len, ten_reward, ten_mask, ten_value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=ten_reward.device
        )  # old policy value
        buf_adv_v = torch.empty(
            buf_len, dtype=torch.float32, device=ten_reward.device
        )  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        ten_bool = torch.not_equal(ten_mask, 0).float()
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_adv_v[i] = ten_reward[i] + ten_bool[i] * (pre_adv_v - ten_value[i])
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.cfg.lambda_gae_adv
        return buf_r_sum, buf_adv_v

    def splice_trajectory(self, buf_srdan, last_done):
        out_srdan = []
        for j in range(self.cfg.experience_items):
            cur_items = []
            buf_items = buf_srdan.pop(0)  # buf_srdan[j]

            for env_i in range(self.cfg.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_items.append(pre_item)

                cur_items.append(buf_items[:last_step, env_i])
                self.traj_list[env_i][j] = buf_items[last_step:, env_i]
                if j == 2:
                    assert buf_items[last_step - 1, env_i] == 0.0, (
                        buf_items[last_step - 1, env_i],
                        last_step,
                        env_i,
                    )

            out_srdan.append(torch.vstack(cur_items).detach().cpu())

        del buf_srdan
        return [
            out_srdan,
        ]

    def _tsp_dataloader(self, tsp_batch_size) -> DataLoader:
        dataset = TSPDataset(
            size=self.cfg.graph_size,
            node_dim=self.cfg.node_dim,
            num_samples=self.cfg.epoch_size,
            data_distribution=self.cfg.data_distribution,
        )
        return DataLoader(
            dataset,
            batch_size=tsp_batch_size,
            num_workers=0,
            pin_memory=True,
        )

    def tsp_data(self):
        # get one tsp graph at epoch start
        if self.__tsp_data_epoch is None or self.__tsp_data_epoch != self.current_epoch:
            self._stored_data = torch.rand(1, self.cfg.graph_size, self.cfg.node_dim)
            self._stored_data = utils.augment_xy_data_by_8_fold(self._stored_data)
            self.__tsp_data_epoch = self.current_epoch
            self.__tsp_used_times = 0

        # iterate over the 8 augmentations
        return_data = self._stored_data[
            self.__tsp_used_times : self.__tsp_used_times + 1
        ]
        self.__tsp_used_times += 1

        if self.__tsp_used_times >= 8:
            self.__tsp_used_times = 0

        return return_data.detach().clone()

    def train_dataloader(self) -> DataLoader:
        dataset = DummyDataset(((1,)), num_samples=self.cfg.epoch_size)
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = TSPDataset(
            size=self.cfg.graph_size,
            node_dim=self.cfg.node_dim,
            num_samples=self.cfg.val_size,
            data_distribution=self.cfg.data_distribution,
            data_path=self.cfg.val_data_path,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.val_batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        graph_feat = self.encoder(states)
        action = self.actor(graph_feat)

        return action

    def on_train_start(self) -> None:
        # to prevent exactly the same TSP graph on different card
        pl.seed_everything(self.cfg.seed + self.global_rank)

    def on_train_epoch_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        sche_actor, sche_critic, sche_low_level = self.lr_schedulers()
        sche_actor.step()
        sche_critic.step()
        sche_low_level.step()
        self.log(
            "train/lr",
            sche_actor.get_last_lr()[0],
        )

    def low_level_training(self, optim_low_level: torch.optim.Optimizer) -> None:
        self.low_level_model.group_size = self.low_level_model.cfg.group_size
        self.low_level_model.cfg.fine_tune = True
        self.low_level_model.cfg.norm_reward = True
        self.low_level_model.cfg.knn_decode = False
        self.trainer._callback_connector.attach_model_logging_functions(
            self.low_level_model
        )
        for _ in range(self.cfg.low_level_update_time):
            data = self.frag_buffer.sample_batch(self.cfg.low_level_batch_size).to(
                self.device
            )
            outputs = self.low_level_model.training_step(data, _)
            self.optim_update(optim_low_level, outputs["loss"])
            self.log_dict(
                {
                    "low_level/loss": outputs["loss"].detach().item(),
                    "low_level/length": outputs["length"],
                },
                on_epoch=True,
            )

    def training_step(
        self,
        batch: torch.Tensor,
        _,
    ) -> OrderedDict:
        # interact with environment
        # train_start = time.time()
        traj = self.explore_vec_env(self.cfg.target_steps)
        steps, r_exp = update_buffer(self.memory, traj)
        self.log_dict({"train/env_steps": steps, "train/env_reward": r_exp})
        # env_time = time.time()-train_start

        # update agent
        optim_actor, optim_critic, optim_low_level = self.optimizers(
            use_pl_optimizer=True
        )
        if self.cfg.low_level_training:
            self.low_level_training(optim_low_level)
        # buf_time = time.time() - train_start
        log_obj_critic = []
        log_prob_ratio = []
        log_obj_aux = []
        log_obj_actor = []
        log_obj_surrogate = []
        log_obj_entropy = []
        log_value = []
        log_approx_kl_divs = []
        log_exp_var = []
        log_mmd = []
        obj_critic = None
        obj_actor = None
        buf_len = self.memory[0].shape[0]
        buf_state, buf_reward, buf_mask, buf_action, buf_noise, buf_action_target = [
            ten.to(self.device) for ten in self.memory
        ]
        del self.memory[:]
        for i in range(self.cfg.repeat_times):
            with torch.no_grad():
                """get buf_r_sum, buf_logprob"""
                bs = (
                    self.cfg.train_batch_size * 4
                )  # set a smaller 'BatchSize' when out of GPU memory.

                buf_value = [
                    self.get_critic_value(buf_state[i : i + bs], is_target=True)
                    for i in range(0, buf_len, bs)
                ]
                buf_value = torch.cat(buf_value, dim=0)
                buf_logprob = self.actor.get_old_logprob(buf_action, buf_noise)
                # get advantage
                buf_r_sum, buf_adv_v = self.get_reward_sum_gae(
                    buf_len, buf_reward, buf_mask, buf_value
                )  # detach()
                raw_adv = buf_adv_v.clone()
                buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                    self.cfg.lambda_a_value / (buf_adv_v.std() + 1e-5)
                )
                # buf_adv_v: buffer data of adv_v value
                last_step_idx = torch.where(buf_mask == 0.0)[0]
                first_step_index = (last_step_idx + 1) % buf_len
                episodic_return = buf_r_sum[first_step_index]
                self.log_dict(
                    {
                        "train_data/num_episode": episodic_return.shape[0],
                        "train_data/return_min": episodic_return.min(),
                        "train_data/return_max": episodic_return.max(),
                        "train_data/return_std": episodic_return.std(),
                        "train_data/return_mean": episodic_return.mean(),
                        "train_data/adv_min": raw_adv.min(),
                        "train_data/adv_max": raw_adv.max(),
                        "train_data/adv_mean": raw_adv.mean(),
                        "train_data/adv_std": raw_adv.std(),
                        "train_data/logprob_mean": buf_logprob.mean(),
                        "train_data/logprob_std": buf_logprob.std(),
                    },
                    on_epoch=True,
                    on_step=False,
                )
                del (
                    raw_adv,
                    last_step_idx,
                    first_step_index,
                    episodic_return,
                )
            update_times = int(buf_len / self.cfg.train_batch_size)
            for update_i in range(1, update_times + 1):
                indices = torch.randint(
                    buf_len,
                    size=(self.cfg.train_batch_size,),
                    requires_grad=False,
                    device=self.device,
                )

                state = buf_state[indices]
                r_sum = buf_r_sum[indices]
                adv_v = buf_adv_v[indices]
                action = buf_action[indices]
                logprob = buf_logprob[indices]
                value_pred = buf_value[indices]
                action_target = buf_action_target[indices]

                """PPO: Surrogate objective of Trust Region"""
                graph_feat = self.encoder(state)
                new_logprob, obj_entropy = self.actor.get_logprob_entropy(
                    graph_feat, action
                )  # it is obj_actor
                new_action = self.actor(graph_feat)
                obj_aux = self.criterion(new_action, action_target)
                with torch.no_grad():
                    mmd = utils.MMD(action, new_action)
                    log_mmd.append(mmd.item())
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(
                    1 - self.cfg.ratio_clip, 1 + self.cfg.ratio_clip
                )
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = (
                    obj_surrogate
                    + obj_entropy * self.lambda_entropy
                    + obj_aux * self.cfg.lambda_aux
                )
                self.optim_update(optim_actor, obj_actor)
                # update lambda_entropy
                self.lambda_entropy = (
                    self.lambda_entropy
                    + self.cfg.target_entropy_beta
                    * (self.cfg.target_entropy + obj_entropy.detach().item())
                )
                self.lambda_entropy = max(self.lambda_entropy, 0.0)

                with torch.no_grad():
                    log_ratio = new_logprob - logprob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    log_approx_kl_divs.append(approx_kl_div)

                explained_var = utils.explained_variance(
                    value_pred.cpu().numpy().flatten(), r_sum.cpu().numpy().flatten()
                )
                log_exp_var.append(explained_var.mean())

                # critic network predicts the reward_sum (Q value) of state
                value = self.get_critic_value(state, is_target=False).squeeze(1)
                obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
                self.optim_update(optim_critic, obj_critic)
                log_prob_ratio.append(ratio.mean().detach().item())
                log_obj_aux.append(obj_aux.detach().item())
                log_obj_surrogate.append(obj_surrogate.detach().item())
                log_obj_entropy.append(obj_entropy.detach().item())
                log_obj_actor.append(obj_actor.detach().item())
                log_obj_critic.append(obj_critic.detach().item())
                log_value.append(value.mean().detach().item())

            utils.soft_update(self.critic_target, self.critic, self.cfg.tau)
            utils.soft_update(self.encoder_target, self.encoder, self.cfg.tau)

        a_std_log = getattr(self.actor, "a_std_log", torch.zeros(1)).mean()
        # train_time = time.time() - train_start
        self.log_dict(
            {
                "loss/value": np.mean(log_obj_critic),
                "loss/aux": np.mean(log_obj_aux),
                "loss/policy": np.mean(log_obj_actor),
                "loss/entropy": np.mean(log_obj_entropy),
                "loss/prob_ratio": np.mean(log_prob_ratio),
                "loss/surrogate": np.mean(log_obj_surrogate),
                "loss/mmd": np.mean(log_mmd),
                "train/approx_kl_divs": np.mean(log_approx_kl_divs),
                "train/explained_var": np.mean(log_exp_var),
                "train/value": np.mean(log_value),
                "train/a_std_log": a_std_log,
                "train/a_std": a_std_log.exp(),
                "train/update_times": update_times,
                "train/lambda_entropy": self.lambda_entropy,
            }
        )

        # print(f"\n[Rank {self.local_rank}] Env time: {env_time}, buffer time: {buf_time}, train time: {train_time}\n")

        return OrderedDict({})

    def validation_step(self, batch, _):
        steps = 0
        wandb_logger = (
            self.logger.experiment[1] if len(self.logger.experiment) > 1 else None
        )
        val_env = self.env_maker(no_depot=self.cfg.no_depot)
        states = val_env.reset(batch)
        log_rewards = []
        while not val_env.done:
            with torch.no_grad():
                actions = self(states).detach().squeeze()
            states, rewards, dones, info = val_env.step(
                actions, self.low_level_solver, frag_buffer=self.val_frag_buffer
            )
            log_rewards.append(rewards)
            if steps % self.cfg.log_fig_freq == 0 and wandb_logger:
                if not dones[0]:
                    fig_tour = utils.visualize_route(
                        val_env.envs[0].state.current_tour,
                        val_env.envs[0].state.x.cpu().numpy(),
                        Env.scale_action(actions[0].cpu().numpy()),
                        info["fragments"][0].cpu().numpy(),
                        info["new_cities"][0].cpu().numpy(),
                    )
                    wandb_logger.log(
                        {
                            f"val/tour_{steps:02d}": wandb.Image(fig_tour),
                            # f"val/reward_{steps:02d}": rewards[0],
                        }
                    )
                    plt.close()
                    steps += 1
                else:
                    fig_tour = utils.visualize_route(
                        val_env.envs[0].state.current_tour,
                        val_env.envs[0].state.x.cpu().numpy(),
                        Env.scale_action(actions[0].cpu().numpy()),
                        info["fragments"][0].cpu().numpy(),
                        info["new_cities"][0].cpu().numpy(),
                    )
                    wandb_logger.log(
                        {
                            f"val/tour_{steps:02d}": wandb.Image(fig_tour),
                            # f"val/reward_{steps:02d}": rewards[0],
                        }
                    )
                    wandb_logger.log(
                        {
                            f"val/tour_last": wandb.Image(fig_tour),
                        }
                    )
                    plt.close()
                    wandb_logger = None

        average_len = np.mean([e.state.current_tour_len.item() for e in val_env.envs])
        discounted_rewards = torch.zeros_like(log_rewards[0])
        for r in log_rewards[::-1]:
            discounted_rewards = discounted_rewards * self.cfg.gamma + r

        self.print(f"Validation on {batch.shape[0]} graphs, mean length: {average_len}")
        self.log_dict(
            {
                "val/mean_tour_length": average_len,
                "val/discounted_rewards": discounted_rewards.mean(),
            },
            on_epoch=True,
        )
        del val_env

        return {}

    def test_step(self, *args, **kwargs):
        return self.validation_step(self, *args, **kwargs)

    def optim_update(
        self, optimizer: torch.optim.Optimizer, objective: torch.Tensor
    ) -> None:  # [ElegantRL 2021.11.11]
        """minimize the optimization objective via update the network parameters

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        self.manual_backward(objective)
        for param_group in optimizer.param_groups:
            if self.cfg.grad_method == "norm":
                nn.utils.clip_grad_norm_(
                    parameters=param_group["params"], max_norm=self.cfg.grad_clip
                )
            elif self.cfg.grad_method == "value":
                nn.utils.clip_grad_value_(
                    parameters=param_group["params"], clip_value=self.cfg.grad_clip
                )
            else:
                raise ArgumentError(
                    self.cfg.grad_method, "can only be 'norm' or 'value'"
                )
        optimizer.step()
