from __future__ import absolute_import

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class IMPALACNN(nn.ModuleDict):
    """
    CNN from paper:
        "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
        https://arxiv.org/abs/1802.01561

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, input_channels, output_dim, depths=None, input_shape=None):
        if depths is None:
            depths = [16, 32, 32]
        if input_shape is None:
            input_shape = [100, 100]
        super().__init__()
        self.input_channels = input_channels
        input_shape = torch.as_tensor(input_shape)

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        for num_ch in depths:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.flatten = nn.Flatten()

        flatten_dim = self._get_flatten_dim(input_shape)
        self.fc = nn.Linear(flatten_dim, output_dim)

    def _get_flatten_dim(self, input_shape):
        input_sample = torch.empty(
            [
                1,
                self.input_channels,
                input_shape[0].int().item(),
                input_shape[1].int().item(),
            ]
        )
        x = input_sample
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(self.flatten(x))
        x = F.relu(self.fc(x))

        return x


class IMPALAEncoder(nn.ModuleDict):
    """
    CNN from paper:
        "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
        https://arxiv.org/abs/1802.01561

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim,
        bev_range=None,
        bev_pixel_size=None,
        depths=None,
    ):
        if bev_range is None:
            bev_range = [0.0, 0.0, 1.0, 1.0]
        if bev_pixel_size is None:
            bev_pixel_size = [0.01, 0.01]
        if depths is None:
            depths = [16, 32, 32]
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        super().__init__()
        self.bev_range = torch.tensor(bev_range)
        self.bev_pixel_size = torch.tensor(bev_pixel_size)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.input_channels = embedding_dim // 2
        bev_grid_shape = (self.bev_range[2:] - self.bev_range[:2]) / self.bev_pixel_size
        self.input_transform = nn.Sequential(
            nn.Conv1d(
                self.input_dim + 4,
                self.input_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm1d(
                self.input_channels, eps=1e-3, momentum=0.01, affine=True
            ),
            nn.ReLU(inplace=True),
        )
        self.cnn = IMPALACNN(
            self.input_channels, self.embedding_dim, depths, bev_grid_shape
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # get input feature, whose shape is [B*N, input_dim]
        # size of 2nd dim changed from 2 to 8, add three features
        B, N, _ = batch.shape
        H = self.input_channels
        bev_range = self.bev_range.to(batch.device)
        bev_pixel_size = self.bev_pixel_size.to(batch.device)
        bev_grid_shape = (bev_range[2:] - bev_range[:2]) / bev_pixel_size
        bev_idxs = (batch[..., 0:2].view(-1, 2) - bev_range[0:2]) / bev_pixel_size
        bev_coords = torch.floor(bev_idxs).int()
        # flatten, points.shape = [B*N, 1+input_dim]
        batch_idxs = (
            torch.arange(0, B).view(1, B).repeat(N, 1).t().flatten().to(batch.device)
        )
        points = torch.cat([batch_idxs[:, None], batch.flatten(0, 1)], dim=1)
        points_xy = points[:, [1, 2]]
        extra_feat = points[:, 3:]
        assert points.shape[1] == self.input_dim + 1
        bev_scale_xy = bev_grid_shape[0] * bev_grid_shape[1]
        bev_scale_y = bev_grid_shape[1]
        bev_merge_coords = (
            points[:, 0].int() * bev_scale_xy
            + bev_coords[:, 0] * bev_scale_y
            + bev_coords[:, 1]
        )
        bev_unq_coords, bev_unq_inv, bev_unq_cnt = torch.unique(
            bev_merge_coords, return_inverse=True, return_counts=True, dim=0
        )
        bev_f_center = points_xy - (
            (bev_coords.to(points_xy.dtype) + 0.5) * bev_pixel_size + bev_range[[0, 1]]
        )
        bev_f_mean = torch_scatter.scatter_mean(points_xy, bev_unq_inv, dim=0)
        bev_f_cluster = points_xy - bev_f_mean[bev_unq_inv, :]
        bev_f_cluster = bev_f_cluster[:, [0, 1]]
        # distance = torch.sqrt(torch.sum(points**2, dim=1, keepdim=True))
        mvf_input = torch.cat(
            [points_xy, bev_f_center, bev_f_cluster, extra_feat], dim=1
        ).contiguous()  # [B * N, input_dim + 4]

        # get pseudo image and inital transformation,
        # whose shape is [B, H, h, w]
        mvf_input = mvf_input.view(B, N, -1)  # [B, N, C]
        mvf_input = mvf_input.transpose(1, 2)  # [B, C, N]
        pt_fea_in = self.input_transform(mvf_input)
        pt_fea_bev = pt_fea_in.transpose(1, 2).flatten(0, 1)  # [B*N, input_channels]
        bev_fea_in = torch_scatter.scatter_max(pt_fea_bev, bev_unq_inv, dim=0)[0]
        pixel_coords = torch.stack(
            (
                torch.div(bev_unq_coords, bev_scale_xy, rounding_mode="floor"),
                torch.div(
                    bev_unq_coords % bev_scale_xy, bev_scale_y, rounding_mode="floor"
                ),
                torch.div(bev_unq_coords % bev_scale_y, 1, rounding_mode="floor"),
                bev_unq_coords % 1,
            ),
            dim=1,
        )
        pixel_coords = pixel_coords[:, [0, 3, 2, 1]]

        # forward image
        batch_bev_features = []
        for batch_idx in range(B):
            feature = torch.zeros(
                H,
                bev_scale_xy.int().item(),
                dtype=bev_fea_in.dtype,
                device=bev_fea_in.device,
            )

            batch_mask = pixel_coords[:, 0] == batch_idx
            this_coords = pixel_coords[batch_mask, :]
            indices = (
                this_coords[:, 1]
                + this_coords[:, 2] * bev_grid_shape[0]
                + this_coords[:, 3]
            )
            indices = indices.type(torch.long)
            feature[:, indices] = bev_fea_in[batch_mask, :].t()
            batch_bev_features.append(feature)
        batch_bev_features = torch.stack(batch_bev_features, 0)
        batch_bev_features = batch_bev_features.view(
            B, H, bev_grid_shape[1].int().item(), bev_grid_shape[0].int().item()
        )
        batch_bev_features = batch_bev_features.permute(0, 1, 3, 2)

        return self.cnn(batch_bev_features)


class ActorPPO(nn.Module):
    """
    Actor class for **PPO** with stochastic, learnable, **state-independent** log standard deviation.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, state_dim, mid_dim, action_dim, init_a_std_log=-0.5):
        super().__init__()

        nn_middle = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn_middle,
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
        )

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(
            torch.ones((1, action_dim)).mul_(init_a_std_log), requires_grad=True
        )  # calculated from action space
        self.register_parameter("a_std_log", self.a_std_log)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        self.reset_parameter()

    def reset_parameter(self):
        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_norm(module)
        # rescale last layer
        last_layer = self.net[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        layer_norm(last_layer, 0.01)

    def forward(self, state):
        """
        The forward function.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        """
        The forward function with Gaussian noise.

        :param state[np.array]: the input state.
        :return: the action and added noise.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        """
        Compute the log of probability with current network.

        :param state[np.array]: the input state.
        :param action[float]: the action.
        :return: the log of probability and entropy.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        dist = torch.distributions.Normal(a_avg, a_std)
        logprob = dist.log_prob(action).sum(1)
        dist_entropy = -dist.entropy().mean()
        del dist

        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        """
        Compute the log of probability with old network.

        :param _action[float]: the action.
        :param noise[float]: the added noise when exploring.
        :return: the log of probability with old network.
        """
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

    def get_old_logprob_act(self, old_action, old_noise, action):
        """
        Compute the log of probability with out new noise.

        :param _action[float]: the action.
        :param noise[float]: the added noise when exploring.
        :return: the log of probability with old network.
        """
        a_std = self.a_std_log.exp()
        noise = (old_action - action) / a_std - old_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class CriticPPO(nn.Module):
    """
    The Critic class for **PPO**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, state_dim, mid_dim, _action_dim):
        super().__init__()

        nn_middle = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn_middle,
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, 1),
        )
        # layer_norm(self.net[-1], std=0.5)  # output layer for advantage value
        self.reset_parameter()

    def reset_parameter(self):
        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_norm(module)

    def forward(self, state):
        """
        The forward function to ouput the value of the state.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state)  # advantage value
