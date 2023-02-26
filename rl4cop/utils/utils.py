import math
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg


def multi_head_attention(q, k, v, mask=None):
    # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
    # k,v shape = (B, n_heads, N, key_dim)
    # mask.shape = (B, group, N)

    B, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_heads, n, N)
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))

    if mask is not None:
        score += mask[:, None, :, :].expand_as(score)

    shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
    attn = torch.matmul(
        F.softmax(score.float(), dim=3, dtype=torch.float32).type_as(score), v
    ).transpose(1, 2)
    return attn.reshape(*shp)


def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)


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


def make_graphs(batch, n_neighbors=25, gnn_framework="dgl"):
    graph_list = []
    topk_distances, indices = torch.topk(
        torch.cdist(batch, batch), k=n_neighbors + 1, largest=False
    )
    for i in range(batch.size(0)):
        x = batch[i]
        u = indices[i, :, 1:].flatten()
        v = indices[i, :, 0:1].repeat(1, n_neighbors).flatten()
        if gnn_framework == "dgl":
            g = dgl.graph((u, v))
            g.ndata["feat"] = x
            g.edata["feat"] = topk_distances[i, :, 1:].reshape(-1, 1)
        else:
            edge_index = torch.stack([u, v])
            edge_attr = topk_distances[i, :, 1:].reshape(-1, 1)
            g = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_list.append(g)

    if gnn_framework == "dgl":
        return dgl.batch(graph_list).to(batch.device)
    else:
        return pyg.data.Batch.from_data_list(graph_list).to(batch.device)


def make_dgl_graphs(batch, n_neighbors=25):
    graph_list = []
    topk_distances, indices = torch.topk(
        torch.cdist(batch, batch), k=n_neighbors + 1, largest=False
    )
    for i in range(batch.size(0)):
        x = batch[i]
        u = indices[i, :, 1:].flatten()
        v = indices[i, :, 0:1].repeat(1, n_neighbors).flatten()
        g = dgl.graph((u, v))
        g.ndata["feat"] = x
        g.edata["feat"] = topk_distances[i, :, 1:].view(-1, 1)
        graph_list.append(g)
    return dgl.batch(graph_list).to(batch.device)


def make_pyg_graphs(batch, n_neighbors=25):
    graph_list = []
    topk_distances, indices = torch.topk(
        torch.cdist(batch, batch), k=n_neighbors + 1, largest=False
    )
    for i in range(batch.size(0)):
        x = batch[i]
        u = indices[i, :, 1:].flatten()
        v = indices[i, :, 0:1].repeat(1, n_neighbors).flatten()
        edge_index = torch.stack([u, v])
        edge_attr = topk_distances[i, :, 1:].view(-1, 1)
        g = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_list.append(g)
    return pyg.data.Batch.from_data_list(graph_list).to(batch.device)


class TimeStat(object):
    """A time stat for logging the elapsed time of code running
    Example:
        time_stat = TimeStat()
        with time_stat:
            // some code
        print(time_stat.mean)
    """

    def __init__(self, window_size=1):
        self.time_samples = WindowStat(window_size)
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        time_delta = time.time() - self._start_time
        self.time_samples.add(time_delta)

    @property
    def mean(self):
        return self.time_samples.mean

    @property
    def min(self):
        return self.time_samples.min

    @property
    def max(self):
        return self.time_samples.max


class WindowStat(object):
    """Tool to maintain statistical data in a window."""

    def __init__(self, window_size):
        self.items = [None] * window_size
        self.idx = 0
        self.count = 0

    def add(self, obj):
        self.items[self.idx] = obj
        self.idx += 1
        self.count += 1
        self.idx %= len(self.items)

    @property
    def mean(self):
        if self.count > 0:
            return np.mean(self.items[: self.count])
        else:
            return None

    @property
    def min(self):
        if self.count > 0:
            return np.min(self.items[: self.count])
        else:
            return None

    @property
    def max(self):
        if self.count > 0:
            return np.max(self.items[: self.count])
        else:
            return None


def bilinear_interpolate_torch(im, x, y, align=False):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    if align:
        x0 = x0.float() + 0.5
        x1 = x1.float() + 0.5
        y0 = y0.float() + 0.5
        y1 = y1.float() + 0.5

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = (
        torch.t((torch.t(Ia) * wa))
        + torch.t(torch.t(Ib) * wb)
        + torch.t(torch.t(Ic) * wc)
        + torch.t(torch.t(Id) * wd)
    )
    return ans


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module, a=1, mode="fan_in", nonlinearity="leaky_relu", distribution="uniform"
    )
