from __future__ import annotations

import torch
from torch import nn

from confopt.oneshot.dropout import Dropout
from confopt.oneshot.partial_connector import PartialConnector
from confopt.oneshot.weightentangler import WeightEntangler

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

    def change_op_channel_size(self, wider: int | None = None) -> None:
        if wider is None or wider == 1:
            return

        for op in self.ops:
            if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                op.change_channel_size(k=1 / wider, device=DEVICE)  # type: ignore


class OperationBlock(nn.Module):
    def __init__(
        self,
        ops: list[nn.Module],
        is_reduction_cell: bool,
        partial_connector: PartialConnector | None = None,
        dropout: Dropout | None = None,
        weight_entangler: WeightEntangler | None = None,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__()
        self.device = device
        if partial_connector:
            for op in ops:
                if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                    op.change_channel_size(
                        partial_connector.k, self.device  # type: ignore
                    )
        self.ops = ops
        self.partial_connector = partial_connector
        self.is_reduction_cell = is_reduction_cell
        self.dropout = dropout
        self.weight_entangler = weight_entangler

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        if self.dropout:
            alphas = self.dropout.apply_mask(alphas)
        if self.partial_connector is not None:
            self.partial_connector.is_reduction_cell = self.is_reduction_cell
        if self.partial_connector:
            return self.partial_connector(x, alphas, self.ops)

        if self.weight_entangler is not None:
            output = self.weight_entangler.forward(x, self.ops, alphas)
        else:
            states = [op(x) * alpha for op, alpha in zip(self.ops, alphas)]
            output = sum(states)

        return output

    def change_op_channel_size(self, wider: int | None = None) -> None:
        if wider is None:
            wider = self.partial_connector.k if self.partial_connector else 1
        if wider == 1:
            return

        for op in self.ops:
            if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                op.change_channel_size(k=1 / wider, device=self.device)  # type: ignore
