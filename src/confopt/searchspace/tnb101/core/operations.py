from __future__ import annotations

from math import log2

import torch
from torch import nn

from confopt.oneshot.weightentangler import (
    ConvolutionalWEModule,
    WeightEntanglementSequential,
)
from confopt.searchspace.common import Conv2DLoRA
import confopt.utils.change_channel_size as ch

TRANS_NAS_BENCH_101 = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3"]
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# OPS defines operations for micro cell structures
OPS = {
    "none": lambda C_in, C_out, stride, affine, track_running_stats: Zero(  # noqa:
        C_in, C_out, stride  # type: ignore
    ),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (1, 1),
        stride,
        (0, 0),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: (
        Identity()
        if (stride == 1 and C_in == C_out)
        else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats)
    ),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3, 3),
        stride,
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
}


class ReLUConvBN(ConvolutionalWEModule):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        dilation: int | tuple[int, int],
        affine: bool,
        track_running_stats: bool,
        activation: str = "relu",
    ):
        super().__init__()
        if activation == "leaky":
            ops = [nn.LeakyReLU(0.2, False)]
        elif activation == "relu":
            ops = [nn.ReLU(inplace=False)]
        else:
            raise ValueError(f"invalid activation {activation}")
        ops += [
            Conv2DLoRA(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        ]
        self.op = WeightEntanglementSequential(*ops)
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        )

        self.__post__init__()

    def mark_entanglement_weights(self) -> None:
        self.op[1].can_entangle_weight = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)  # type: ignore

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,  # noqa: ARG002
        device: torch.device = DEVICE,
    ) -> None:
        assert (k is not None) or (num_channels_to_add is not None)

        self.op[1], index = ch.change_channel_size_conv(
            self.op[1], k=k, num_channels_to_add=num_channels_to_add, device=device
        )
        self.op[2], _ = ch.change_features_bn(
            self.op[2],
            k=k,
            num_channels_to_add=num_channels_to_add,
            index=index,
            device=device,
        )

    def activate_lora(self, r: int) -> None:
        self.op[1].activate_lora(r)

    def deactivate_lora(self) -> None:
        self.op[1].deactivate_lora()

    def toggle_lora(self) -> None:
        self.op[1].toggle_lora()

    def extra_repr(self) -> str:
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
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
        return x.new_zeros(shape, dtype=x.dtype, device=x.device)[
            :, :, :: self.stride, :: self.stride
        ]

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,  # noqa: ARG002
        device: torch.device = DEVICE,
    ) -> None:
        assert (k is not None) or (num_channels_to_add is not None)
        if k is not None:
            self.C_in = int(self.C_in // k)
            self.C_out = int(self.C_out // k)
        if num_channels_to_add is not None:
            self.C_in += num_channels_to_add
            self.C_out += num_channels_to_add
        self.device = device

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
                    Conv2DLoRA(
                        C_in,
                        C_outs[i],
                        1,
                        stride=stride,
                        padding=0,
                        bias=not affine,
                    )
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = Conv2DLoRA(
                C_in, C_out, 1, stride=stride, padding=0, bias=not affine
            )
        else:
            raise ValueError(f"Invalid stride : {stride}")
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
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

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,  # noqa: ARG002
        device: torch.device = DEVICE,
    ) -> None:
        assert (k is not None) or (num_channels_to_add is not None)
        if k is not None and k > 1:
            if self.stride == 2:
                for i in range(2):
                    self.convs[i] = ch.reduce_conv_channels(
                        self.convs[i], k=k, device=device
                    )
            elif self.stride == 1:
                self.conv = ch.reduce_conv_channels(self.conv, k=k, device=device)
            else:
                raise ValueError(f"Invalid stride: {self.stride}")
            self.bn = ch.reduce_bn_features(self.bn, k)
            return
        if num_channels_to_add is not None:
            num_channels_to_add_C_in = num_channels_to_add
            num_channels_to_add_C_out = num_channels_to_add

        if self.stride == 2:
            if k is not None:
                num_channels_to_add_C_in = int(
                    max(1, self.convs[0].in_channels // int(1 / k - 1))
                )
                num_channels_to_add_C_out = int(
                    max(1, self.convs[0].out_channels // int(1 / k - 1))
                )

            self.convs[0], _ = ch.increase_in_channel_size_conv(
                self.convs[0], num_channels_to_add_C_in
            )
            self.convs[0], index1 = ch.increase_out_channel_size_conv(
                self.convs[0], num_channels_to_add_C_out // 2
            )
            self.convs[1], _ = ch.increase_in_channel_size_conv(
                self.convs[1], num_channels_to_add_C_in
            )
            self.convs[1], index2 = ch.increase_out_channel_size_conv(
                self.convs[1],
                num_channels_to_add_C_out - num_channels_to_add_C_out // 2,
            )
            self.bn, _ = ch.increase_num_features_bn(
                self.bn,
                num_channels_to_add_C_out,
                index=torch.cat([index1, index2]),
            )
        elif self.stride == 1:
            if k is not None:
                num_channels_to_add_C_in = int(max(1, self.conv.in_channels // k))
                num_channels_to_add_C_out = int(max(1, self.conv.out_channels // k))

            self.conv, _ = ch.increase_in_channel_size_conv(
                self.conv, num_channels_to_add_C_in
            )
            self.conv, index = ch.increase_out_channel_size_conv(
                self.conv, num_channels_to_add_C_out
            )
            self.bn, _ = ch.increase_num_features_bn(
                self.bn, num_channels_to_add_C_out, index=index
            )

    def activate_lora(self, r: int) -> None:
        if self.stride == 2:
            for i in range(2):
                self.convs[i].activate_lora(r)
        elif self.stride == 1:
            self.conv.activate_lora(r)
        else:
            raise ValueError(f"Invalid stride : {self.stride}")

    def deactivate_lora(self) -> None:
        if self.stride == 2:
            for i in range(2):
                self.convs[i].deactivate_lora()
        elif self.stride == 1:
            self.conv.deactivate_lora()
        else:
            raise ValueError(f"Invalid stride : {self.stride}")

    def toggle_lora(self) -> None:
        if self.stride == 2:
            for i in range(2):
                self.convs[i].toggle_lora()
        elif self.stride == 1:
            self.conv.toggle_lora()
        else:
            raise ValueError(f"Invalid stride : {self.stride}")

    def extra_repr(self) -> str:
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int] | str,
        activation: nn.Module,
        norm: nn.Module,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel, stride=stride, padding=padding
        )
        self.activation = activation
        if norm:
            if norm == nn.BatchNorm2d:
                self.norm = norm(out_channel)
            else:
                self.norm = norm
                self.conv = norm(self.conv)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm and isinstance(self.norm, nn.BatchNorm2d):
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DeconvLayer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int] | str,
        activation: nn.Module,
        norm: nn.Module,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel,
            stride=stride,
            padding=padding,
            output_padding=1,
        )
        self.activation = activation
        if norm == nn.BatchNorm2d:
            self.norm = norm(out_channel)
        else:
            self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm and isinstance(self.norm, nn.BatchNorm2d):
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Stem(nn.Module):
    """This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, C_in: int = 3, C_out: int = 64):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class StemJigsaw(nn.Module):
    """This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, C_in: int = 3, C_out: int = 64):
        super().__init__(locals())
        self.seq = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, s3, s4, s5 = x.size()
        x = x.reshape(-1, s3, s4, s5)
        return self.seq(x)


class SequentialJigsaw(nn.Module):
    """Implementation of `torch.nn.Sequential` to be used
    as op on edges.
    """

    def __init__(self, *args):  # type: ignore
        super().__init__()
        self.primitives = args
        self.op = nn.Sequential(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, s2, s3, s4 = x.size()
        x = x.reshape(-1, 9, s2, s3, s4)
        enc_out = []
        for i in range(9):
            enc_out.append(x[:, i, :, :, :])
        x = torch.cat(enc_out, dim=1)
        return self.op(x)


class Sequential(nn.Module):
    """Implementation of `torch.nn.Sequential` to be used
    as op on edges.
    """

    def __init__(self, *args):  # type: ignore
        super().__init__()
        self.primitives = args
        self.op = nn.Sequential(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class GenerativeDecoder(nn.Module):
    def __init__(
        self,
        in_dim: tuple[int, int],
        target_dim: tuple[int, int],
        target_num_channel: int = 3,
        norm: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()

        in_channel, in_width = in_dim[0], in_dim[1]
        out_width = target_dim[0]
        num_upsample = int(log2(out_width / in_width))
        assert num_upsample in [
            2,
            3,
            4,
            5,
            6,
        ], f"invalid num_upsample: {num_upsample}"

        self.conv1 = ConvLayer(in_channel, 1024, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv2 = ConvLayer(1024, 1024, 3, 2, 1, nn.LeakyReLU(0.2), norm)

        if num_upsample == 6:
            self.conv3 = DeconvLayer(1024, 512, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv3 = ConvLayer(1024, 512, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv4 = ConvLayer(512, 512, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        if num_upsample >= 5:
            self.conv5 = DeconvLayer(512, 256, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv5 = ConvLayer(512, 256, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv6 = ConvLayer(256, 128, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        if num_upsample >= 4:
            self.conv7 = DeconvLayer(128, 64, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv7 = ConvLayer(128, 64, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv8 = ConvLayer(64, 64, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        if num_upsample >= 3:
            self.conv9 = DeconvLayer(64, 32, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv9 = ConvLayer(64, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv10 = ConvLayer(32, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv11 = DeconvLayer(32, 16, 3, 2, 1, nn.LeakyReLU(0.2), norm)

        self.conv12 = ConvLayer(16, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv13 = DeconvLayer(32, 16, 3, 2, 1, nn.LeakyReLU(0.2), norm)

        self.conv14 = ConvLayer(16, target_num_channel, 3, 1, 1, nn.Tanh(), norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        return x


OLES_OPS = [Zero, Identity, ReLUConvBN]
