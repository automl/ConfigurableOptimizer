from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.darts.core.operations import DEVICE
from confopt.searchspace.darts.core.operations import OPS as DARTS_OPS

OPS = DARTS_OPS | {
    "noise": lambda C, stride, affine: NoiseOp(stride, 0.0, 1.0),  # noqa: ARG005
}


class NoiseOp(nn.Module):
    def __init__(self, stride: int, mean: float, std: float) -> None:
        super().__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_new = x[:, :, :: self.stride, :: self.stride] if self.stride != 1 else x

        noise = torch.randn_like(x_new) * self.std + self.mean
        noise = noise.to(DEVICE)
        return noise

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        pass


def drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.bernoulli(keep_prob * torch.ones(x.size(0), 1, 1, 1))
        x.div_(keep_prob)
        x.mul_(mask)
    return x
