##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from __future__ import annotations

import torch
from torch import nn

from confopt.utils.reduce_channels import reduce_bn_features, reduce_conv_channels

__all__ = ["OPS", "ResNetBasicblock", "SearchSpaceNames", "ReLUConvBN"]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
OPS = {
    "none": lambda C_in, C_out, stride, affine, track_running_stats: Zero(  # noqa:
        C_in, C_out, stride  # type: ignore
    ),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: Pooling(
        C_in, C_out, stride, "avg", affine, track_running_stats
    ),
    "max_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: Pooling(
        C_in, C_out, stride, "max", affine, track_running_stats
    ),
    "nor_conv_7x7": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (7, 7),
        (stride, stride),
        (3, 3),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (1, 1),
        (stride, stride),
        (0, 0),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(  # noqa:
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(  # noqa:
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (2, 2),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dil_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (2, 2),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "dil_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (4, 4),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity()
    if stride == 1 and C_in == C_out
    else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}

NAS_BENCH_201 = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
DARTS_SPACE = [
    "none",
    "skip_connect",
    "dua_sepc_3x3",
    "dua_sepc_5x5",
    "dil_sepc_3x3",
    "dil_sepc_5x5",
    "avg_pool_3x3",
    "max_pool_3x3",
]

SearchSpaceNames = {
    "nas-bench-201": NAS_BENCH_201,
    "darts": DARTS_SPACE,
}


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        dilation: int | tuple[int, int],
        affine: bool,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)  # type: ignore

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        # TODO: make this change dynamic
        self.op[1] = reduce_conv_channels(self.op[1], k, device)
        self.op[2] = reduce_bn_features(self.op[2], k, device)


class SepConv(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        dilation: int | tuple[int, int],
        affine: bool,
        track_running_stats: bool = True,
    ):
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
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)  # type: ignore

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        self.op[1] = reduce_conv_channels(self.op[1], k, device)
        self.op[2] = reduce_conv_channels(self.op[2], k, device)
        self.op[3] = reduce_bn_features(self.op[3], k, device)


class DualSepConv(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        dilation: int | tuple[int, int],
        affine: bool,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.op_a = SepConv(
            C_in,
            C_in,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats,
        )
        self.op_b = SepConv(
            C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.op_a(x)
        x = self.op_b(x)
        return x

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        self.op_b.change_channel_size(k, device)
        self.op_b.change_channel_size(k, device)


class ResNetBasicblock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        assert stride == 1 or stride == 2, f"invalid stride {stride}"
        self.conv_a = ReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(  # type: ignore
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None  # type: ignore
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self) -> str:
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        residual = self.downsample(inputs) if self.downsample is not None else inputs
        return residual + basicblock  # type: ignore

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        pass


class Pooling(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int | tuple[int, int],
        mode: str,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)  # type: ignore
        else:
            raise ValueError(f"Invalid mode={mode} in POOLING")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(inputs) if self.preprocess else inputs
        return self.op(x)  # type: ignore

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        if self.preprocess:
            self.preprocess.change_channel_size(k, device)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        pass


class Zero(nn.Module):
    def __init__(self, C_in: int, C_out: int, stride: int):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            return x[:, :, :: self.stride, :: self.stride].mul(0.0)

        shape = list(x.shape)
        shape[1] = self.C_out
        zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
        return zeros

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        pass

    def extra_repr(self) -> str:
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class FactorizedReduce(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int | tuple[int, int],
        affine: bool,
        track_running_stats: bool,
    ):
        super().__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(
                        C_in, C_outs[i], 1, stride=stride, padding=0, bias=not affine
                    )
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride=stride, padding=0, bias=not affine
            )
        else:
            raise ValueError(f"Invalid stride : {stride}")
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        if self.stride == 2:
            for i in range(2):
                self.convs[i] = reduce_conv_channels(self.convs[i], k, device)
        elif self.stride == 1:
            self.conv = reduce_conv_channels(self.conv, k, device)
        else:
            raise ValueError(f"Invalid stride : {self.stride}")

        self.bn = reduce_bn_features(self.bn, k)

    def extra_repr(self) -> str:
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)
