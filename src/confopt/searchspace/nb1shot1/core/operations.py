from __future__ import annotations

import torch
from torch import nn

import confopt.utils.reduce_channels as rc

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
OPS = {
    # For nasbench
    "maxpool3x3": lambda C, stride, affine: nn.MaxPool2d(  # noqa: ARG005
        3, stride=stride, padding=1
    ),
    "conv3x3-bn-relu": lambda C, stride, affine: Conv3x3BnRelu(  # noqa: ARG005
        C, stride
    ),
    "conv1x1-bn-relu": lambda C, stride, affine: Conv1x1BnRelu(  # noqa: ARG005
        C, stride
    ),
    # Normal DARTS
    "none": lambda C, stride, affine: Zero(stride),  # noqa: ARG005
    "avg_pool_3x3": lambda C, stride, affine: Pooling(
        C,
        stride,
        "avg",
        affine,
    ),
    "skip_connect": lambda C, stride, affine: (
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine)
    ),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(
        C, C, 5, stride, 2, affine=affine
    ),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(
        C, C, 7, stride, 3, affine=affine
    ),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine
    ),
    "conv_7x1_1x7": lambda C, stride, affine: Conv7x1Conv1x7BN(
        C, stride, affine=affine
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
        kernel_size: int,
        stride: int,
        padding: int = 1,
    ):
        super().__init__()
        self.op = nn.Sequential(
            # Padding = 1 is for a 3x3 kernel equivalent to tensorflow padding = same
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            # affine is equivalent to scale in original tensorflow code
            nn.BatchNorm2d(C_out, affine=True, momentum=BN_MOMENTUM, eps=BN_EPSILON),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        self.op[0] = rc.reduce_conv_channels(self.op[0], k, device)
        self.op[1] = rc.reduce_bn_features(self.op[1], k, device)


class Conv3x3BnRelu(nn.Module):
    """Equivalent to Conv3x3BnRelu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L96.
    """

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.op = ConvBnRelu(
            C_in=channels, C_out=channels, kernel_size=3, stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        self.op.change_channel_size(k, device)


class Conv1x1BnRelu(nn.Module):
    """Equivalent to Conv1x1BnRelu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L107.
    """

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.op = ConvBnRelu(
            C_in=channels,
            C_out=channels,
            kernel_size=1,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        self.op.change_channel_size(k, device)


"""DARTS OPS"""


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)  # type: ignore

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        # TODO: make this change dynamic
        self.op[1] = rc.reduce_conv_channels(self.op[1], k, device)
        self.op[2] = rc.reduce_bn_features(self.op[2], k, device)


class Pooling(nn.Module):
    def __init__(
        self,
        C: int,
        stride: int | tuple[int, int],
        mode: str,
        affine: bool = False,
    ):
        super().__init__()
        if mode == "avg":
            op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            op = nn.MaxPool2d(3, stride=stride, padding=1)  # type: ignore
        else:
            raise ValueError(f"Invalid mode={mode} in POOLING")
        self.op = nn.Sequential(op, nn.BatchNorm2d(C, affine=affine))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.op(inputs)  # type: ignore

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        self.op[1] = rc.reduce_bn_features(self.op[1], k, device)


class DilConv(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        self.op[1] = rc.reduce_conv_channels(self.op[1], k, device)
        self.op[2] = rc.reduce_conv_channels(self.op[2], k, device)
        self.op[3] = rc.reduce_bn_features(self.op[3], k, device)


class SepConv(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        self.op[1] = rc.reduce_conv_channels(self.op[1], k, device)
        self.op[2] = rc.reduce_conv_channels(self.op[2], k, device)
        self.op[3] = rc.reduce_bn_features(self.op[3], k, device)
        self.op[5] = rc.reduce_conv_channels(self.op[5], k, device)
        self.op[6] = rc.reduce_conv_channels(self.op[6], k, device)
        self.op[7] = rc.reduce_bn_features(self.op[7], k, device)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        pass


class Zero(nn.Module):
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        pass


class FactorizedReduce(nn.Module):
    def __init__(self, C_in: int, C_out: int, affine: bool = True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        self.conv_1 = rc.reduce_conv_channels(self.conv_1, k, device)
        self.conv_2 = rc.reduce_conv_channels(self.conv_2, k, device)
        self.bn = rc.reduce_bn_features(self.bn, k, device)


class Conv7x1Conv1x7BN(nn.Module):
    def __init__(
        self,
        C: int,
        stride: int,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(C, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        # TODO: make this change dynamic
        self.op[1] = rc.reduce_conv_channels(self.op[1], k, device)
        self.op[2] = rc.reduce_conv_channels(self.op[2], k, device)
        self.op[3] = rc.reduce_bn_features(self.op[3], k, device)
