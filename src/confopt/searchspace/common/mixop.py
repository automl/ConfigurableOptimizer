from __future__ import annotations

from thop import profile as flop_profile
import torch
from torch import nn

from confopt.oneshot.dropout import Dropout
from confopt.oneshot.partial_connector import PartialConnector
from confopt.oneshot.weightentangler import WeightEntangler

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
__all__ = ["OperationChoices"]


class OperationChoices(nn.Module):
    def __init__(
        self,
        ops: list[nn.Module],
        is_reduction_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__()
        self.ops = ops
        self.is_reduction_cell = is_reduction_cell
        self.device = device
        self.flops: list[float] | None = None

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        assert len(alphas) == len(
            self.ops
        ), "Number of operations and architectural weights do not match"
        if self.flops is None:
            self._calculate_flops(x)
        states = []
        for op, alpha in zip(self.ops, alphas):
            states.append(op(x) * alpha)

        return sum(states)  # type: ignore

    def change_op_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
    ) -> None:
        if not k and k == 1:
            return

        for op in self.ops:
            # if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
            op.change_channel_size(
                k=k,
                num_channels_to_add=num_channels_to_add,
                new_cell=new_cell,
                device=self.device,
            )  # type: ignore
            if hasattr(op, "__post__init__"):
                op.__post__init__()
        self.flops = None

    def change_op_stride_size(self, new_stride: int) -> None:
        for op in self.ops:
            op.change_stride_size(new_stride)
        self.flops = None

    def _calculate_flops(self, x: torch.Tensor) -> None:
        input_tensor = torch.randn_like(x)
        self.flops = []
        with torch.no_grad():
            for op in self.ops:
                op_flop, _ = flop_profile(op, inputs=(input_tensor,))
                self.flops.append(op_flop)

        # move buffers created by thop to gpu
        for op in self.ops:
            op.to(DEVICE)

    def flops_forward(self, alphas: torch.Tensor) -> torch.Tensor:
        assert self.flops is not None, (
            "Atleast one forward pass is required to polulate"
            + " flops for this OperationBlock"
        )
        return sum([alpha * op_flop for alpha, op_flop in zip(alphas, self.flops)])


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

        self.ops = ops
        self.partial_connector = partial_connector
        self.is_reduction_cell = is_reduction_cell
        self.dropout = dropout
        self.weight_entangler = weight_entangler
        self.is_argmax_sampler = is_argmax_sampler
        self.flops: list[float] | None = None

    def forward_method(
        self, x: torch.Tensor, ops: list[nn.Module], alphas: list[torch.Tensor]
    ) -> torch.Tensor:
        if self.weight_entangler is not None:
            return self.weight_entangler.forward(x, ops, alphas)

        states = []
        if self.is_argmax_sampler:
            argmax = torch.argmax(alphas)

            for i, op in enumerate(ops):
                if i == argmax:
                    states.append(alphas[i] * op(x))
                else:
                    states.append(alphas[i])
        else:
            for op, alpha in zip(ops, alphas):
                states.append(op(x) * alpha)

        return sum(states)

    def forward(
        self,
        x: torch.Tensor,
        alphas: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.flops is None:
            self._calculate_flops(x)

        if self.dropout:
            alphas = self.dropout.apply_mask(alphas)

        if self.partial_connector:
            return self.partial_connector(x, alphas, self.ops, self.forward_method)

        return self.forward_method(x, self.ops, alphas)

    def change_op_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
    ) -> None:
        if not k and not num_channels_to_add:
            k = self.partial_connector.k if self.partial_connector else 1
        if k and k == 1:
            return

        for op in self.ops:
            # if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
            op.change_channel_size(
                k=k,
                num_channels_to_add=num_channels_to_add,
                new_cell=new_cell,
                device=self.device,
            )  # type: ignore
            if hasattr(op, "__post__init__"):
                op.__post__init__()
        self.flops = None

    def change_op_stride_size(self, new_stride: int) -> None:
        for op in self.ops:
            op.change_stride_size(new_stride)
        self.flops = None

    def _calculate_flops(self, x: torch.Tensor) -> None:
        input_tensor = torch.randn_like(x)
        self.flops = []
        for op in self.ops:
            op_flop, _ = flop_profile(op, inputs=(input_tensor,), verbose=False)
            self.flops.append(op_flop)

        # move buffers created by thop to gpu
        for op in self.ops:
            op.to(DEVICE)

    def get_weighted_flops(self, alphas: torch.Tensor) -> torch.Tensor:
        assert self.flops is not None, (
            "Atleast one forward pass is required to polulate"
            + " flops for this OperationBlock"
        )
        return sum([alpha * op_flop for alpha, op_flop in zip(alphas, self.flops)])
