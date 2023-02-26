from contextlib import contextmanager
from typing import Dict, List

import abc
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape = [B, N, 2]
    # x,y shape = [B, N, 1]

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    # data_augmented.shape = [8*B, N, 2]
    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)

    return data_augmented


def nodes_to_sample(node_pos: np.ndarray, max_demand: int = 9) -> Dict:
    sample = {}
    sample["capacity"] = max_demand * len(node_pos)
    sample["depot"] = node_pos[0]
    sample["customers"] = []

    for i in range(1, len(node_pos)):
        sample["customers"].append({"position": node_pos[i], "demand": 1})

    return sample


@torch.jit.script
def get_nearest_city_idx(
    x: torch.Tensor, predict_coord: torch.Tensor, mask: torch.Tensor
) -> int:
    """find the city nearest to given coordinates, return its coordinates"""
    assert x.ndim == 2
    assert predict_coord.shape == (2,)
    assert mask.ndim == 1
    masked_cities = torch.where(mask == torch.tensor(True))[0]
    dist_matrix = torch.cdist(
        x[masked_cities].type(torch.float64), predict_coord[None, :].type(torch.float64)
    ).type(
        x.dtype
    )  # shape=[len(available_idx), 1]
    nearest_city = masked_cities[dist_matrix.argmin()]

    return nearest_city.item()


@torch.jit.script
def get_nearest_city_coord(
    x: torch.Tensor, predict_coord: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """find the city nearest to given coordinates, return its coordinates"""
    nearest_city = get_nearest_city_idx(x, predict_coord, mask)
    nearest_coord = x[nearest_city]

    return nearest_coord


def get_centroid_coord(x: np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, f"{x.ndim}"
    assert selected_mask.ndim == 1, f"{selected_mask.ndim}"
    assert x.shape[0] == selected_mask.shape[0], f"{x.shape}-{selected_mask.shape}"

    centroid_coord = x[selected_mask].mean(axis=0)
    assert centroid_coord.shape == (2,)

    return centroid_coord


@torch.jit.script
def get_tour_distance(tour: List[int], dist_matrix: torch.Tensor):
    assert dist_matrix.dim() == 2
    num_nodes = dist_matrix.shape[0]
    assert num_nodes == dist_matrix.shape[1]
    _tour = torch.tensor(tour, dtype=torch.int64, device=dist_matrix.device)
    _tour_offset = torch.tensor(
        tour[1:] + tour[0:1], dtype=torch.int64, device=dist_matrix.device
    )
    edge_list = torch.stack([_tour, _tour_offset], dim=1)
    l = (
        dist_matrix.gather(index=edge_list[:, 0][:, None].expand(-1, num_nodes), dim=0)
        .gather(index=edge_list[:, 1][:, None], dim=1)
        .sum()
    )
    return l


def get_nearest_cluster_city_idx(
    dist_matrix: torch.Tensor, old_cities: List[int], new_cities: List[int]
) -> int:
    assert dist_matrix.ndim == 2
    num_nodes = dist_matrix.shape[-1]
    len_old = len(old_cities)
    old_idx = torch.tensor(old_cities, device=dist_matrix.device)
    new_idx = torch.tensor(new_cities, device=dist_matrix.device)
    cluster_matrix = dist_matrix.gather(
        index=old_idx[:, None].expand(-1, num_nodes), dim=0
    ).gather(index=new_idx[None, :].expand(len_old, -1), dim=1)

    min_idx = torch.argmin(cluster_matrix).item()

    old_id = min_idx // new_idx.shape[0]
    old_id = old_idx[old_id]

    return old_id.item()


def scale_spatial_feat(data: torch.Tensor):
    """Scale coordinates and distance value from [0, 1] to [-1, 1]"""
    assert data.dim() == 2
    assert data.shape[-1] == 2 or data.shape[-1] == 4
    assert data.min() >= 0
    assert data.max() <= 1

    return data * 2 - 1


@torch.jit.script
def update_neighbor_coord_(
    neighbor_coord: torch.Tensor, current_tour: List[int], nodes_coord: torch.Tensor
) -> None:
    assert nodes_coord.ndim == 2, nodes_coord.shape
    num_nodes = nodes_coord.shape[0]
    node_dim = nodes_coord.shape[1]
    num_edges = len(current_tour)
    tour = torch.tensor(current_tour, dtype=torch.int64, device=nodes_coord.device)
    curr_pre_next = torch.stack(
        [tour, torch.roll(tour, -1), torch.roll(tour, 1)], dim=1
    )
    assert curr_pre_next.shape[1] == 3, curr_pre_next.shape
    pre_coord = nodes_coord.gather(
        index=curr_pre_next[:, 1][:, None].expand(-1, node_dim), dim=0
    )
    next_coord = nodes_coord.gather(
        index=curr_pre_next[:, 2][:, None].expand(-1, node_dim), dim=0
    )
    pre_next_coord = torch.cat([pre_coord, next_coord], dim=-1)
    neighbor_coord.scatter_(
        index=curr_pre_next[:, 0][:, None].expand(-1, 2 * node_dim),
        src=pre_next_coord,
        dim=0,
    )


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def heatmap(data, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if cbar_kw is None:
        cbar_kw = {}

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    # ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    # ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def generate_q_heatmap(state, model, RES=20):
    with torch.no_grad() as a, evaluating(model) as model:
        graph_feat = model.encoder(
            torch.tensor(state.to_numpy()[None, :], device=model.device)
        )
        RES = 10
        heat_map = np.zeros((RES, RES))
        for i in range(RES):
            for j in range(RES):
                q_v = model.critic(
                    [
                        graph_feat,
                        torch.tensor([[i / RES, j / RES]], device=graph_feat.device),
                    ]
                )
                heat_map[i, j] = q_v.item()

    return heat_map


def visualize_route(
    route: List[int],
    node_pos: np.ndarray,
    predict_coord,
    fragment: np.ndarray,
    newcity: np.ndarray,
) -> None:
    G = nx.DiGraph()
    depotG = nx.DiGraph()
    depotG.add_node(0)
    actionG = nx.DiGraph()
    actionG.add_node(0)
    pos = node_pos
    edge_list = []
    node_colors = []
    cmap = plt.get_cmap("Set1").reversed()
    to_hex = matplotlib.colors.to_hex
    for i in range(0, len(pos)):
        G.add_node(i)
        if i in newcity:
            node_colors.append(to_hex(cmap.colors[4]))
        elif i in fragment:
            node_colors.append(to_hex(cmap.colors[2]))
        elif i in route:
            node_colors.append(to_hex(cmap.colors[1]))
        else:
            node_colors.append(to_hex(cmap.colors[0]))

    route = route + route[0:1]
    for idx in range(len(route) - 1):
        edge_list.append((route[idx], route[idx + 1]))
    fig = plt.figure(figsize=(10, 10))
    plt.axis("on")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
    nx.draw_networkx_nodes(
        depotG,
        pos[0:1],
        node_color=to_hex(cmap.colors[-1]),
        node_size=100,
        node_shape="p",
    )
    nx.draw_networkx_nodes(
        actionG,
        predict_coord[None, :],
        node_color=to_hex(cmap.colors[-2]),
        node_size=100,
        node_shape="*",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        width=2,
        arrows=False,
        arrowsize=1,
        edgelist=edge_list,
    )

    return fig


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


class FragmengBuffer:
    def __init__(self, max_len: int, frag_len: int, node_dim: int = 2) -> None:
        self.max_len = max_len
        self.frag_len = frag_len
        self.node_dim = node_dim
        self.frag_buffer = torch.empty((max_len, frag_len, node_dim))
        self.if_full = False
        self.now_len = 0
        self.next_idx = 0

    def update_buffer(self, fragments: torch.Tensor) -> None:
        size = fragments.shape[0]
        assert fragments.shape[1] == self.frag_len
        assert fragments.shape[2] == self.node_dim
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            self.frag_buffer[self.next_idx : self.max_len] = fragments[
                : self.max_len - self.next_idx
            ]
            self.if_full = True
            next_idx = next_idx - self.max_len
            self.frag_buffer[0:next_idx] = fragments[-next_idx:]
        else:
            self.frag_buffer[self.next_idx : next_idx] = fragments
        self.next_idx = next_idx
        self.update_now_len()

    def update_now_len(self) -> None:
        self.now_len = self.max_len if self.if_full else self.next_idx

    def sample_batch(self, batch_size) -> tuple:
        indices = np.random.randint(self.now_len - 1, size=batch_size)
        return self.frag_buffer[indices]


def MMD(x: torch.Tensor, y: torch.Tensor, kernel: str = "multiscale"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.
       Borrowed From https://www.kaggle.com/onurtunali/maximum-mean-discrepancy

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


class MeanStd(metaclass=abc.ABCMeta):
    """Abstract base class that keeps track of mean and standard deviation.
    Modified from https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/policy_gradient/modules/running_statistics.py
    """

    @abc.abstractmethod
    def init(self, size):
        """Initializes normalization variables.
        Args:
          size: Integer with the dimensionality of the tracked tensor.
        """
        raise NotImplementedError("`init` is not implemented.")

    def normalize(self, x):
        """Normalizes target values x using past target statistics.
        Args:
          x: <float32>[(...), size] tensor.
        Returns:
          <float32>[(...), size] normalized tensor.
        """
        mean, std = self.get_mean_std()
        return (x - mean) / std

    def unnormalize(self, x):
        """Unnormalizes a corrected prediction x using past target statistics.
        Args:
          x: <float32>[(...), size] tensor.
        Returns:
          <float32>[(...), size] unnormalized tensor.
        """
        mean, std = self.get_mean_std()
        return std * x + mean

    @abc.abstractmethod
    def update(self, data):
        """Updates normalization statistics.
        Args:
          data: <float32>[(...), size].
        """
        raise NotImplementedError("`update` is not implemented.")

    @abc.abstractmethod
    def get_mean_std(self):
        """Returns mean and standard deviation for current statistics."""
        raise NotImplementedError("`get_mean_std` is not implemented.")


class EMAMeanStd(MeanStd):
    """Tracks mean and standard deviation using an exponential moving average.
    This works by keeping track of the first and second non-centralized moments
    using an exponential average of the global batch means of these moments, i.e.,
        new_1st_moment = (1-beta)*old_1st_moment + beta*mean(data)
        new_2nd_moment = (1-beta)*old_2nd_moment + beta*mean(data**2).
    Initially, mean and standard deviation are set to zero and one respectively.
    """

    def __init__(self, beta=1e-2, std_min_value=1e-6, std_max_value=1e6):
        """Creates a EMAMeanVariance.
        Args:
          beta: Float that determines how fast parameters are updated via the
            formula `new_parameters = (1-beta)* old_parameters + beta*batch_mean`.
          std_min_value: Float with the minimum value for the standard deviation.
          std_max_value: Float with the maximum value for the standard deviation.
        """
        super().__init__()
        self._beta = beta
        self._std_min_value = std_min_value
        self._std_max_value = std_max_value
        self.first_moment = None
        self.second_moment = None

    def init(self, size):
        """Initializes normalization variables.
        Args:
          size: Integer with the dimensionality of the tracked tensor.
        """
        self.first_moment = torch.zeros(size=[size], dtype=torch.float32)
        self.second_moment = torch.ones(size=[size], dtype=torch.float32)

    def update(self, data: torch.Tensor):
        """Updates normalization statistics.
        Args:
          data: <float32>[(...), size].
        """
        # Reduce tensors along all the dimensions except the last ones.
        reduce_dims = list(range(data.dim()))[:-1]
        batch_first_moment = torch.mean(data, dim=reduce_dims)
        batch_second_moment = torch.mean(data**2, dim=reduce_dims)

        # Updates the tracked moments. We do this by computing the difference to the
        # the current value as that allows us to use mean aggregation to make it
        # work with replicated tensors (e.g., when using multiple TPU cores), i.e.,
        #     new_moment = old_moment + beta*mean(data - old_moment)
        # where the mean is a mean across different replica and within the
        # mini-batches of each replica.
        first_moment_diff = self._beta * (batch_first_moment - self.first_moment)
        second_moment_diff = self._beta * (batch_second_moment - self.second_moment)

        # The following two assign_adds will average their arguments across
        # different replicas as the underlying variables have
        # `aggregation=tf.VariableAggregation.MEAN` set.
        self.first_moment.add_(first_moment_diff)
        self.second_moment.add_(second_moment_diff)

    def get_mean_std(self):
        """Returns mean and standard deviation for current statistics."""
        std = torch.sqrt(self.second_moment - self.first_moment**2)
        std = torch.clip(std, self._std_min_value, self._std_max_value)
        # Multiplication with one converts the variable to a tensor with the value
        # at the time this function is called. This is important if the python
        # reference is passed around and the variables are changed in the meantime.
        return self.first_moment * 1.0, std


def merge_summed_variances(v1, v2, mu1, mu2, merged_mean, n1, n2):
    """Computes the (summed) variance of a combined series.
    Args:
      v1: summed variance of the first series.
      v2: summed variance of the second series.
      mu1: mean of the first series.
      mu2: mean of the second series.
      merged_mean: mean for the combined series.
      n1: Number of datapoints in the first series.
      n2: Number of datapoints in the second series.
    Returns:
      The summed variance for the combined series.
    """
    return (
        v1
        + n1 * torch.square(mu1 - merged_mean)
        + v2
        + n2 * torch.square(mu2 - merged_mean)
    )


def merge_means(mu1, mu2, n1, n2):
    """Merges means. Requires n1 + n2 > 0."""
    total = n1 + n2
    return (n1 * mu1 + n2 * mu2) / total


class AverageMeanStd(MeanStd):
    """Tracks mean and standard deviation across all past samples.
    This works by updating the mean and the sum of past variances with Welford's
    algorithm using batches (see https://stackoverflow.com/questions/56402955/
    whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates).
    One limitation of this class is that it uses float32 to aggregate statistics,
    which leads to inaccuracies after 7M batch due to limited float precision (see
    b/160686691 for details). Use TwoLevelAverageMeanStd to work around that.

    Modified to torch version.
    Attributes:
      observation_count: float32 tf.Variable with observation counts.
      update_count: int32 tf.Variable representing the number of times update() or
        merge() have been called.
      mean: float32 tf.Variable with mean.
      summed_variance: float32 tf.Variable with summed variance of all samples.
    """

    def __init__(self, std_min_value=1e-6, std_max_value=1e6):
        """Creates a AverageMeanStd.
        Args:
          std_min_value: Float with the minimum value for the standard deviation.
          std_max_value: Float with the maximum value for the standard deviation.
        """
        super().__init__()
        self._std_min_value = std_min_value
        self._std_max_value = std_max_value
        self.observation_count = None
        self.update_count = None
        self.mean = None
        self.summed_variance = None

    def init(self, size):
        """Initializes normalization variables.
        Args:
          size: Integer with the dimensionality of the tracked tensor.
        """
        self.observation_count = torch.zeros(size=[size], dtype=torch.float32)

        self.update_count = torch.zeros(size=[], dtype=torch.float32)
        self.mean = torch.zeros(size=[size], dtype=torch.float32)
        self.summed_variance = torch.zeros(size=[size], dtype=torch.float32)

    def update(self, data: torch.Tensor):
        """Updates normalization statistics.
        Args:
          data: <float32>[(...), size].
        """
        # Reduce tensors along all the dimensions except the last ones.
        reduce_dims = list(range(data.dim()))[:-1]

        # Update the observations counts.
        count = torch.ones_like(data, dtype=torch.int32)
        aggregated_count = torch.sum(count, dim=reduce_dims)
        # SUM across replicas.
        self.observation_count.add_(aggregated_count.to(dtype=torch.float32))
        self.update_count.add_(1)
        # Update the mean.
        diff_to_old_mean = data - self.mean
        mean_update = torch.sum(diff_to_old_mean, dim=reduce_dims)
        mean_update /= self.observation_count.to(dtype=torch.float32)
        self.mean.add_(mean_update)

        # Update the variance.
        diff_to_new_mean = data - self.mean
        variance_update = diff_to_old_mean * diff_to_new_mean
        variance_update = torch.sum(variance_update, dim=reduce_dims)
        self.summed_variance.add_(variance_update)

    def get_mean_std(self):
        """Returns mean and standard deviation for current statistics."""
        # The following clipping guarantees an initial variance of one.
        minval = torch.tensor(self._std_min_value * self._std_min_value)
        eff_var = torch.maximum(minval, self.summed_variance)
        eff_count = self.observation_count.to(dtype=torch.float32)
        eff_count = torch.maximum(minval, eff_count)
        std = torch.sqrt(eff_var / eff_count)
        std = torch.clip(std, self._std_min_value, self._std_max_value)
        # Multiplication with one converts the variable to a tensor with the value
        # at the time this function is called. This is important if the python
        # reference is passed around and the variables are changed in the meantime.
        return self.mean * 1.0, std
