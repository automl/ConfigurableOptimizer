from __future__ import annotations

import torch
from torch import nn

from confopt.oneshot.partial_connector import PartialConnector

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
__all__ = ["OperationChoices"]


class OperationChoices(nn.Module):
    def __init__(self, ops: list[nn.Module], is_reduction_cell: bool = False) -> None:
        super().__init__()
        self.ops = ops
        self.is_reduction_cell = is_reduction_cell

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        assert len(alphas) == len(
            self.ops
        ), "Number of operations and architectural weights do not match"
        states = [op(x) * alpha for op, alpha in zip(self.ops, alphas)]
        return sum(states)  # type: ignore


class OperationBlock(nn.Module):
    def __init__(
        self,
        ops: list[nn.Module],
        is_reduction_cell: bool,
        partial_connector: PartialConnector = None,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__()
        if partial_connector:
            for op in ops:
                if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                    op.change_channel_size(partial_connector.k, device)  # type: ignore
        self.ops = ops
        self.partial_connector = partial_connector
        self.is_reduction_cell = is_reduction_cell

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        self.partial_connector.is_reduction_cell = self.is_reduction_cell
        if self.partial_connector:
            return self.partial_connector(x, alphas, self.ops)
        states = [op(x) * alpha for op, alpha in zip(self.ops, alphas)]

        return sum(states)  # type: ignore
