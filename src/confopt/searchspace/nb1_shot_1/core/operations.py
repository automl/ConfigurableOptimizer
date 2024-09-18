from __future__ import annotations

import torch
from torch import nn

from confopt.oneshot.weightentangler import (
    ConvolutionalWEModule,
    WeightEntanglementSequential,
)
from confopt.searchspace.common import Conv2DLoRA
from confopt.searchspace.darts.core.operations import Pooling
import confopt.utils.reduce_channels as rc

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

OPS = {
    # For nasbench
    "maxpool3x3": lambda C, stride, affine: Pooling(  # noqa: ARG005
        C, stride=stride, mode="max"
    ),
    "conv3x3-bn-relu": lambda C, stride, affine: Conv3x3BnRelu(  # noqa: ARG005
        C, stride
    ),
    "conv1x1-bn-relu": lambda C, stride, affine: Conv1x1BnRelu(  # noqa: ARG005
        C, stride
    ),
}

# Batch Normalization from nasbench
BN_MOMENTUM = 0.997
BN_EPSILON = 1e-5

"""NASBench OPS"""


class ConvBnRelu(nn.Module):
    """Equivalent to conv_bn_relu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L32.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int | tuple[int, int],
        stride: int,
        padding: int = 1,
    ):
        super().__init__()
        self.op = WeightEntanglementSequential(
            # Padding = 1 is for a 3x3 kernel equivalent to tensorflow padding = same
            Conv2DLoRA(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            # affine is equivalent to scale in original tensorflow code
            nn.BatchNorm2d(C_out, affine=True, momentum=BN_MOMENTUM, eps=BN_EPSILON),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the block.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method dynamically changes the number of output channels in the
            ReLUConvBN block.
        """
        self.op[0] = rc.reduce_conv_channels(self.op[0], k, device)
        self.op[1] = rc.reduce_bn_features(self.op[1], k, device)

    def activate_lora(self, r: int) -> None:
        self.op[0].activate_lora(r)

    def deactivate_lora(self) -> None:
        self.op[0].deactivate_lora()

    def toggle_lora(self) -> None:
        self.op[0].toggle_lora()


class Conv3x3BnRelu(ConvBnRelu, ConvolutionalWEModule):
    """Equivalent to Conv3x3BnRelu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L96.
    """

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__(channels, channels, 3, stride=stride, padding=1)

    def mark_entanglement_weights(self) -> None:
        self.op[0].can_entangle_weight = True


class Conv1x1BnRelu(ConvBnRelu, ConvolutionalWEModule):
    """Equivalent to Conv1x1BnRelu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L107.
    """

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__(channels, channels, 1, stride=stride, padding=0)

    def mark_entanglement_weights(self) -> None:
        self.op[0].can_entangle_weight = True


class MaxPool2d(nn.Module):
    def __init__(
        self,
        stride: int | tuple[int, int],
        padding: int = 1,
    ) -> None:
        """Pooling Block Class.

        Args:
            stride (int or tuple[int, int]): Stride for the pooling operation.
            padding (int, optional): Padding for the pooling operation. Defaults to 1.
        """
        super().__init__()
        self.op = nn.MaxPool2d(3, stride=stride, padding=padding)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Pooling block.

        Args:
            inputs (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.
        """
        return self.op(inputs)  # type: ignore

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the Pooling block's batch
        norm features.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """


OLES_OPS = [Pooling, Conv3x3BnRelu, Conv1x1BnRelu]
