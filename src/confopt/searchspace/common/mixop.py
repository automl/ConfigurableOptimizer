from __future__ import annotations

from typing import TYPE_CHECKING

from thop import profile as flop_profile
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
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
        use_aux_skip: bool = False,
    ) -> None:
        super().__init__()
        self.ops = ops
        self.is_reduction_cell = is_reduction_cell
        self.device = device
        self.flops: list[float] | None = None
        self.use_aux_skip = use_aux_skip
        self._init_aux_skip_connection()
        self._init_dan()

    def _init_dan(self) -> None:
        C = None
        for op in self.ops:
            # targeting the SepConv Block
            if hasattr(op, "stride") and hasattr(op, "op"):
                for in_op in op.op:
                    if hasattr(in_op, "in_channels"):
                        C = in_op.in_channels
                        break
                break

        self.dan = DynamicAttentionNetwork(
            C=C,  # type: ignore
            num_ops=len(self.ops),
            attention_weight=1,
        )

    def _init_aux_skip_connection(self) -> None:
        stride = 1
        C = None
        affine = False
        if self.is_reduction_cell:
            for op in self.ops:
                # targeting the SepConv Block
                if hasattr(op, "stride"):
                    stride = op.stride
                    if hasattr(op, "op"):
                        for in_op in op.op:
                            if hasattr(in_op, "in_channels"):
                                C = in_op.in_channels
                                break
                        break

        self.aux_skip = AuxiliarySkipConnection(
            stride=stride,
            C_in=C,
            C_out=C,
            affine=affine,
            device=self.device,
        )

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        assert len(alphas) == len(
            self.ops
        ), "Number of operations and architectural weights do not match"
        if self.flops is None:
            self._calculate_flops(x)
        states = []
        for op, alpha in zip(self.ops, alphas):
            states.append(op(x) * alpha)

        if self.use_aux_skip:
            return self.aux_skip(x) + sum(states)

        return sum(states)

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
        aux_skip: nn.Module | None = None,
        dan: nn.Module | None = None,
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
        self.use_aux_skip = aux_skip is not None
        self.aux_skip = aux_skip
        self.dan = dan

    def forward_method(
        self, x: torch.Tensor, ops: list[nn.Module], alphas: list[torch.Tensor]
    ) -> torch.Tensor:
        if self.weight_entangler is not None:
            if self.use_aux_skip:
                return self.aux_skip(x) + self.weight_entangler.forward(  # type: ignore
                    x, ops, alphas
                )
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

        if self.use_aux_skip:
            return self.aux_skip(x) + sum(states)  # type: ignore

        return sum(states)

    def forward(
        self,
        x: torch.Tensor,
        alphas: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.flops is None:
            self._calculate_flops(x)

        if self.dan:
            alphas = alphas + self.dan(x)

        if self.dropout:
            alphas = self.dropout.apply_mask(alphas)

        if self.partial_connector:
            return self.partial_connector(
                x,
                alphas,
                self.ops,
                self.forward_method,
                auxiliary_skip_module=self.aux_skip,
            )

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
        self.flops = []
        if self.partial_connector is not None:
            partial_input_channel = x.shape[1] // self.partial_connector.k
            partial_x = x[:, :partial_input_channel, :, :]
            input_tensor = torch.randn_like(partial_x)
        else:
            input_tensor = torch.randn_like(x)
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


class AuxiliarySkipConnection(nn.Module):
    def __init__(
        self,
        stride: int,
        C_in: int | None = None,
        C_out: int | None = None,
        affine: bool = True,
        device: torch.device = DEVICE,
    ) -> None:
        """Auxilary Skip Connection.

        Args:
            stride (int): stride to apply
            C_in (int): Number of input channels.
            C_out (int): Number of output channels.
            affine (bool): Whether to apply affine transformations in BatchNorm.
            device (torch.device): torch device to use
        """
        super().__init__()
        assert stride in [1, 2], "Stride can only be 1 or 2"
        self.stride = stride

        if self.stride == 2:
            self.C_in = C_in
            self.C_out = C_out
            self.relu = nn.ReLU(inplace=False).to(device)
            self.conv_1 = nn.Conv2d(
                C_in,  # type: ignore
                C_out // 2,  # type: ignore
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            ).to(device)
            self.conv_2 = nn.Conv2d(
                C_in,  # type: ignore
                C_out - C_out // 2,  # type: ignore
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            ).to(device)
            self.bn = nn.BatchNorm2d(C_out, affine=affine).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Auxilary Skip Connection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Skip Connection
        """
        if self.stride == 1:
            return x

        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class DynamicAttentionNetwork(nn.Module):
    def __init__(self, C: int, num_ops: int, attention_weight: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(C, C // 4)
        self.fc2 = nn.Linear(C // 4, num_ops)
        self._attention_weight = attention_weight

    def update_attention_weight(self, attention_weight: float) -> None:
        self._attention_weight = attention_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        out = out.view(x.size(0), -1)  # shape (N, C)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        out = out.mean(dim=0)
        return self._attention_weight * F.softmax(out, dim=-1)
