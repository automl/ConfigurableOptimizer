from __future__ import annotations

import torch
from torch import nn

from confopt.oneshot.dropout import Dropout
from confopt.oneshot.partial_connector import PartialConnector
from confopt.oneshot.weightentangler import WeightEntangler
from confopt.utils import freeze

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
        states = []
        for op, alpha in zip(self.ops, alphas):
            if hasattr(op, "is_pruned") and op.is_pruned:
                continue
            states.append(op(x) * alpha)

        return sum(states)  # type: ignore

    def change_op_channel_size(self, wider: int | None = None) -> None:
        if wider is None or wider == 1:
            return

        for op in self.ops:
            if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                op.change_channel_size(k=1 / wider, device=DEVICE)  # type: ignore

    def set_ops_to_prune(self, mask: list[torch.Tensor]) -> None:
        assert len(mask) == len(self.ops)
        for op, mask_val in zip(self.ops, mask):
            if not torch.is_nonzero(mask_val):
                freeze(op)
                op.is_pruned = True


class OperationBlock(nn.Module):
    def __init__(
        self,
        ops: list[nn.Module],
        is_reduction_cell: bool,
        partial_connector: PartialConnector | None = None,
        dropout: Dropout | None = None,
        weight_entangler: WeightEntangler | None = None,
        device: torch.device = DEVICE,
        is_argmax_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        if partial_connector:
            for op in ops:
                if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                    op.change_channel_size(
                        partial_connector.k, self.device  # type: ignore
                    )
                    if hasattr(op, "__post__init__"):
                        op.__post__init__()

        self.ops = ops
        self.partial_connector = partial_connector
        self.is_reduction_cell = is_reduction_cell
        self.dropout = dropout
        self.weight_entangler = weight_entangler
        self.is_argmax_sampler = is_argmax_sampler

    def forward_method(
        self, x: torch.Tensor, ops: list[nn.Module], alphas: list[torch.Tensor]
    ) -> torch.Tensor:
        if self.weight_entangler is not None:
            return self.weight_entangler.forward(x, ops, alphas)

        states = []
        if self.is_argmax_sampler:
            argmax = torch.argmax(alphas)

            for i, op in enumerate(ops):
                if hasattr(op, "is_pruned") and op.is_pruned:
                    continue

                if i == argmax:
                    states.append(alphas[i] * op(x))
                else:
                    states.append(alphas[i])
        else:
            for op, alpha in zip(ops, alphas):
                if hasattr(op, "is_pruned") and op.is_pruned:
                    continue
                states.append(op(x) * alpha)

        return sum(states)

    def forward(
        self,
        x: torch.Tensor,
        alphas: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.dropout:
            alphas = self.dropout.apply_mask(alphas)

        if self.partial_connector:
            return self.partial_connector(x, alphas, self.ops, self.forward_method)

        return self.forward_method(x, self.ops, alphas)

    def change_op_channel_size(self, wider: int | None = None) -> None:
        if wider is None:
            wider = self.partial_connector.k if self.partial_connector else 1
        if wider == 1:
            return

        for op in self.ops:
            if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                op.change_channel_size(k=1 / wider, device=self.device)  # type: ignore

    def set_ops_to_prune(self, mask: list[torch.Tensor]) -> None:
        assert len(mask) == len(self.ops)
        for op, mask_val in zip(self.ops, mask):
            if not torch.is_nonzero(mask_val):
                freeze(op)
                op.is_pruned = True
