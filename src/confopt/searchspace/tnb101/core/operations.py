from __future__ import annotations

from math import log2

import torch
from torch import nn

from confopt.utils.reduce_channels import (
    reduce_bn_features,
    reduce_conv_channels,
)

TRANS_NAS_BENCH_101 = ["none", "nor_conv_1x1", "skip_connect", "nor_conv_3x3"]
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
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity()
    if (stride == 1 and C_in == C_out)
    else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
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


class ReLUConvBN(nn.Module):
    """ReLU-Convolution-BatchNorm Block Class.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        kernel_size (int or tuple[int, int]): Size of the convolutional kernel.
        stride (int or tuple[int, int]): Stride for the convolution operation.
        padding (int or tuple[int, int]): Padding for the convolution operation.
        dilation (int or tuple[int, int]): Dilation rate for the convolution operation.
        affine (bool): Whether to apply affine transformations in BatchNorm.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm.
        activation (str): Activation used in operation. "relu" and "leaky" are defined.
        Default to "relu".

    Attributes:
        ops (nn.Sequential): Sequential block containing Activation, Convolution,
        and BatchNorm operations.
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int or tuple[int, int]): Stride for the convolution operation.
    """

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
            nn.Conv2d(
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
        self.ops = nn.Sequential(*ops)
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ReLUConvBN block.

        Args:
            x (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.
        """
        return self.ops(x)  # type: ignore

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the ReLUConvBN block.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """
        # TODO: make this change dynamic
        self.ops[1] = reduce_conv_channels(self.ops[1], k, device)
        self.ops[2] = reduce_bn_features(self.ops[2], k, device)

    def extra_repr(self) -> str:
        """Create class representation.

        Returns:
            str: Class representation based on C_in, C_out and stride.
        """
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class Identity(nn.Module):
    """Identity Block Class.

    This class represents an identity block, which simply passes the input tensor
    through without any changes.

    Attributes:
        None
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor unchanged.
        """
        return x

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the Identity block
        (no operation performed).

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """


class Zero(nn.Module):
    """Zero Block Class.

    This class represents a block that performs a zero operation on the input
    tensor, adjusting the output tensor's dimensions based on the specified
    parameters.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int): Stride for the zero operation.

    Attributes:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int): Stride for the zero operation.
        is_zero (bool): Flag indicating whether this block performs a zero operation.
    """

    def __init__(self, C_in: int, C_out: int, stride: int):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with zeros, adjusted based on block parameters.
        """
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            return x[:, :, :: self.stride, :: self.stride].mul(0.0)

        shape = list(x.shape)
        shape[1] = self.C_out
        return x.new_zeros(shape, dtype=x.dtype, device=x.device)[
            :, :, :: self.stride, :: self.stride
        ]

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the Zero block.

        This method will not change the functionality, as the Zero block does not
        change the number of channels anyway.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """
        self.C_in = self.C_in // k
        self.C_out = self.C_out // k
        self.device = device

    def extra_repr(self) -> str:
        """Create class representation.

        Returns:
            str: Class representation based on C_in, C_out and stride.
        """
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class FactorizedReduce(nn.Module):
    """Factorized Reduce Block Class.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int or tuple[int, int]): Stride for the factorized reduce operation.
        affine (bool): Whether to apply affine transformations in BatchNorm.
        track_running_stats (bool): Whether to track running statistics in BatchNorm.

    Attributes:
        stride (int or tuple[int, int]): Stride for the factorized reduce operation.
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        relu (nn.ReLU): ReLU activation layer.
        convs (nn.ModuleList): List of Conv2d layers for factorized reduction when
        stride is 2.
        pad (nn.ConstantPad2d): Padding layer used for factorized reduction when stride
        is 2.
        conv (nn.Conv2d): Conv2d layer for factorized reduction when stride is 1.
        bn (nn.BatchNorm2d): BatchNorm layer.
    """

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
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride=stride, padding=0, bias=not affine
            )
        else:
            raise ValueError(f"Invalid stride : {stride}")
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Factorized Reduce block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after factorized reduction.
        """
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def change_channel_size(self, k: int, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the Factorized Reduce
        block.

        This method dynamically changes the number of output channels in the block's
        convolutional layers and BatchNorm.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """
        if self.stride == 2:
            for i in range(2):
                self.convs[i] = reduce_conv_channels(self.convs[i], k, device)
        elif self.stride == 1:
            self.conv = reduce_conv_channels(self.conv, k, device)
        else:
            raise ValueError(f"Invalid stride : {self.stride}")

        self.bn = reduce_bn_features(self.bn, k)

    def extra_repr(self) -> str:
        """Create class representation.

        Returns:
            str: Class representation based on C_in, C_out and stride.
        """
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class ConvLayer(nn.Module):
    """Convolution Class.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel (int or tuple[int, int]): Size of convolutional kernel.
        stride (int or tuple[int, int]): Stride for the convolutional operation.
        padding (int or tuple[int, int] or str): Padding for the convolutional
        operation.
        activation (nn.Module): Activation layer.
        norm (nn.Module, optional): Normalization layer.

    Attributes:
        conv (nn.Conv2d): Convolution layer.
        activation (nn.Module): Activation layer.
        norm (nn.Module): Normalization layer.
    """

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
        """Forward pass through the Convolution, Normalization and Activation..

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        if self.norm and isinstance(self.norm, nn.BatchNorm2d):
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DeconvLayer(nn.Module):
    """Deconvolution Class.

    Reverses the process of convolution. Essentially a Transposed Convolution layer.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel (int or tuple[int, int]): Size of deconvolutional kernel.
        stride (int or tuple[int, int]): Stride for the deconvolutional operation.
        padding (int or tuple[int, int] or str): Padding for the deconvolutional
        operation.
        activation (nn.Module): Activation layer.
        norm (nn.Module): Normalization layer.

    Attributes:
        conv (nn.ConvTranspose2d): Deconvolution layer.
        activation (nn.Module): Activation layer.
        norm (nn.Module): Normalization layer.
    """

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
        """Forward pass through the Deconvolution, Normalization and Activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        if self.norm and isinstance(self.norm, nn.BatchNorm2d):
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Stem(nn.Module):
    """This is used as an initial layer directly after the
    image input.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.

    Attributes:
        seq (nn.Sequential): Contains one Convolution and one Normalisation layer
    """

    def __init__(self, C_in: int = 3, C_out: int = 64):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Stem.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.seq(x)


class StemJigsaw(nn.Module):
    """This is used as an initial layer directly after the
    image input.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.

    Attributes:
        seq (nn.Sequential): Contains one Convolution and one Normalisation layer
    """

    def __init__(self, C_in: int = 3, C_out: int = 64):
        super().__init__(locals())
        self.seq = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Stem.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        _, _, s3, s4, s5 = x.size()
        x = x.reshape(-1, s3, s4, s5)
        return self.seq(x)


class SequentialJigsaw(nn.Module):
    """Implementation of `torch.nn.Sequential` to be used
    as op on edges. Special to the `Jigsaw` task.

    Args:
        *args: Positional arguments to pass to the Sequential Module.
        Have to be nn.Module.

    Attributes:
        seq (nn.Sequential): Contains all given layers.
    """

    def __init__(self, *args):  # type: ignore
        super().__init__()
        self.primitives = args
        self.op = nn.Sequential(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Sequential Module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
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

    Args:
        *args: Positional arguments to pass to the Sequential Module.
        Have to be nn.Module.

    Attributes:
        primitives (list[nn.Modules]): Contains a list of all given modules.
        op (nn.Sequential): Contains all given layers.
    """

    def __init__(self, *args):  # type: ignore
        super().__init__()
        self.primitives = args
        self.op = nn.Sequential(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class GenerativeDecoder(nn.Module):
    """Args:
        in_dim (tuple[int, int]): Number of input channels and width.
        target_dim (tuple[int, int]): Number of output channels and width.
        target_num_channel (int, optional): Number of channels in the output.
        Default is 3.
        norm (nn.Module, optional): Normalization layer. Default is `nn.BatchNorm2d`.

    Attributes:
        conv1 (ConvLayer): ConvLayer, channels: input_dim -> 1024
        conv2 (ConvLayer): ConvLayer, channels: 1024 -> 1024
        conv3 (nn.Module): DeconvLayer or ConvLayer, channels: 1024 -> 512
        conv4 (ConvLayer): ConvLayer, channels: 512 -> 512
        conv5 (nn.Module): DeconvLayer or ConvLayer, channels: 512 -> 256
        conv6 (ConvLayer): ConvLayer, channels: 256 -> 128
        conv7 (nn.Module): DeconvLayer or ConvLayer, channels: 128 -> 64
        conv8 (ConvLayer): ConvLayer, channels: 64 -> 64
        conv9 (nn.Module): DeconvLayer or ConvLayer, channels: 64 -> 32
        conv10 (ConvLayer): ConvLayer, channels: 32 -> 32
        conv11 (DeconvLayer): DeconvLayer, channels: 32 -> 16
        conv12 (ConvLayer): ConvLayer with LeakyReLU as activation layer.
        conv13 (DeconvLayer):
        conv14 (ConvLayer): ConvLayer with Tanh as activation layer.

    """

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
