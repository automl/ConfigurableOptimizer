from __future__ import annotations

import torch
from torch import nn

from confopt.utils.channel_shuffle import channel_shuffle


class PartialConnector(nn.Module):
    def __init__(
        self,
        ops: list[nn.Module],
        is_reduction_cell: bool = False,
        k: int = 4,
    ) -> None:
        super().__init__()
        self.k = k
        self.is_reduction_cell = is_reduction_cell
        for op in ops:
            op.change_channel_size(k)  # type: ignore
        self.ops = ops

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        assert len(alphas) == len(
            self.ops
        ), "Number of operations and architectural weights do not match"

        dim_2 = x.shape[1]  # channel number of the input to the layer
        xtemp = x[:, : dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k :, :, :]
        temp1 = sum(op(xtemp) * alpha for op, alpha in zip(self.ops, alphas))
        states = torch.cat([temp1, xtemp2], dim=1)
        states = channel_shuffle(states, self.k)
        return states
