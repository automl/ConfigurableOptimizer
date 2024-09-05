##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from __future__ import annotations

import torch
from torch import nn

from confopt.oneshot.weightentangler import (
    ConvolutionalWEModule,
    WeightEntanglementSequential,
)
from confopt.searchspace.common import Conv2DLoRA
import confopt.utils.change_channel_size as ch

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
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: (
        Identity()
        if stride == 1 and C_in == C_out
        else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats)
    ),
}

NAS_BENCH_201 = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]
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


class ReLUConvBN(ConvolutionalWEModule):
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
        BatchNorm. Defaults to True.

    Attributes:
        op (nn.Sequential): Sequential block containing ReLU, Convolution, and BatchNorm
        operations.

    Note:
        This class represents a ReLU-Convolution-BatchNorm block commonly used in
        neural network architectures.
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
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        )
        self.stride = stride
        self.op = WeightEntanglementSequential(
            nn.ReLU(inplace=False),
            Conv2DLoRA(
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

        self.__post__init__()

    def mark_entanglement_weights(self) -> None:
        self.op[1].can_entangle_weight = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ReLUConvBN block.

        Args:
            x (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the ReLUConvBN block, applying
            ReLU activation, convolution, and BatchNorm operations to the input tensor.
        """
        return self.op(x)  # type: ignore

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,  # noqa: ARG002
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the ReLUConvBN block.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method dynamically changes the number of output channels in the
            ReLUConvBN block.
        """
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


class SepConv(ConvolutionalWEModule):
    """Separable Convolution-BatchNorm Block Class.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        kernel_size (int or tuple[int, int]): Size of the convolutional kernel.
        stride (int or tuple[int, int]): Stride for the convolution operation.
        padding (int or tuple[int, int]): Padding for the convolution operation.
        dilation (int or tuple[int, int]): Dilation rate for the convolution operation.
        affine (bool): Whether to apply affine transformations in BatchNorm.
        track_running_stats (bool, optional): Whether to track running statistics
        in BatchNorm. Defaults to True.

    Attributes:
        op (nn.Sequential): Sequential block containing ReLU, Depthwise Convolution,
            Pointwise Convolution, and BatchNorm operations.

    Note:
        This class represents a separable convolutional block, commonly used in
        neural network architectures.
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
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        )
        self.stride = stride
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv2DLoRA(
                C_in,
                C_in,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            Conv2DLoRA(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

        self.__post__init__()

    def mark_entanglement_weights(self) -> None:
        self.op[1].can_entangle_weight = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SepConv block.

        Args:
            x (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the SepConv block,
            applying ReLU activation, depthwise convolution, pointwise convolution,
            and BatchNorm operations to the input tensor.
        """
        return self.op(x)  # type: ignore

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,  # noqa: ARG002
        new_cell: bool = False,  # noqa: ARG002
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the SepConv block.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method dynamically changes the number of output channels in the SepConv
            block.
        """
        if k is not None and k > 1:
            self.op[1] = ch.reduce_conv_channels(self.op[1], k=k, device=device)
            self.op[2] = ch.reduce_conv_channels(self.op[2], k=k, device=device)
            self.op[3] = ch.reduce_bn_features(self.op[3], k=k, device=device)
        # for k < 1, in SepConv according to DRNAS there is no wider() function

    def activate_lora(self, r: int) -> None:
        self.op[1].activate_lora(r)
        self.op[2].activate_lora(r)

    def deactivate_lora(self) -> None:
        self.op[1].deactivate_lora()
        self.op[2].deactivate_lora()

    def toggle_lora(self) -> None:
        self.op[1].toggle_lora()
        self.op[2].toggle_lora()


class DualSepConv(nn.Module):
    """Dual Separable Convolution-BatchNorm Block Class.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        kernel_size (int or tuple[int, int]): Size of the convolutional kernel.
        stride (int or tuple[int, int]): Stride for the convolution operation.
        padding (int or tuple[int, int]): Padding for the convolution operation.
        dilation (int or tuple[int, int]): Dilation rate for the convolution operation.
        affine (bool): Whether to apply affine transformations in BatchNorm.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm. Defaults to True.

    Attributes:
        op_a (SepConv): SepConv block for intermediate processing.
        op_b (SepConv): SepConv block for final output.

    Note:
        This class represents a dual separable convolutional block that consists of two
        SepConv blocks.
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
            C_in,
            C_out,
            kernel_size,
            1,
            padding,
            dilation,
            affine,
            track_running_stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DualSepConv block.

        Args:
            x (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the DualSepConv block,
            which consists of two SepConv blocks applied sequentially.
        """
        x = self.op_a(x)
        x = self.op_b(x)
        return x

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the DualSepConv block.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method dynamically changes the number of output channels in both
            SepConv blocks of the DualSepConv block.
        """
        self.op_b.change_channel_size(k, num_channels_to_add, new_cell, device)
        self.op_b.change_channel_size(k, num_channels_to_add, new_cell, device)

    def activate_lora(self, r: int) -> None:
        self.op_a.activate_lora(r)
        self.op_b.activate_lora(r)

    def deactivate_lora(self) -> None:
        self.op_a.deactivate_lora()
        self.op_b.deactivate_lora()

    def toggle_lora(self) -> None:
        self.op_a.toggle_lora()
        self.op_b.toggle_lora()


class ResNetBasicblock(nn.Module):
    """Basic ResNet Block Class.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride for the convolution operation.
        affine (bool, optional): Whether to apply affine transformations in BatchNorm.
        Defaults to True.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm. Defaults to True.

    Attributes:
        conv_a (ReLUConvBN): First convolutional block.
        conv_b (ReLUConvBN): Second convolutional block.
        downsample (nn.Module | None): Downsample block for adjusting input dimensions.
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stride (int): Stride of the block.
        num_conv (int): Number of convolutional layers in the block (fixed at 2).

    Note:
        This class represents a basic building block used in ResNet architectures.
    """

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
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
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
        """Return an informative string representation of the ResNetBasicblock.

        Returns:
            str: A string containing information about the block's input and output
            dimensions, and stride.

        Note:
            This method constructs a string representation of the block.
        """
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNetBasicblock.

        Args:
            inputs (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the ResNetBasicblock,
            applying two convolutional blocks and an optional downsample block.
        """
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        residual = self.downsample(inputs) if self.downsample is not None else inputs
        return residual + basicblock  # type: ignore

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the ResNetBasicblock
        (no operation performed).

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method does not perform any operations, as the number of channels in
            this block is fixed.
        """
        # this function is not implemented in DRNAS for wider() or partial connection


class Pooling(nn.Module):
    """Pooling Block Class.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int or tuple[int, int]): Stride for the pooling operation.
        mode (str): Pooling mode, either "avg" for average pooling or "max" for
        max pooling.
        affine (bool, optional): Whether to apply affine transformations in BatchNorm
        (if preprocess is used). Defaults to True.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm (if preprocess is used). Defaults to True.

    Attributes:
        preprocess (ReLUConvBN | None): Preprocessing block for adjusting
        input dimensions.
        None if C_in equals C_out.
        op (nn.Module): Pooling operation (AvgPool2d or MaxPool2d).

    Note:
        This class represents a pooling block with optional preprocessing for
        input dimension adjustment.
    """

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
        """Forward pass through the Pooling block.

        Args:
            inputs (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the Pooling block,
            optionally applying preprocessing and then pooling based on the specified
            mode.
        """
        x = self.preprocess(inputs) if self.preprocess else inputs
        return self.op(x)  # type: ignore

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the Pooling block's
        preprocessing (if used).

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method dynamically changes the number of output channels in the
            preprocessing block (if used).
        """
        if self.preprocess:
            self.preprocess.change_channel_size(
                k=k,
                num_channels_to_add=num_channels_to_add,
                new_cell=new_cell,
                device=device,
            )


class Identity(nn.Module):
    """Identity Block Class.

    Note:
        This class represents an identity block, which simply passes the input tensor
        through without any changes.

    Attributes:
        None
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Identity block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor unchanged.

        Note:
            This method performs a forward pass through the Identity block, returning
            the input tensor as-is.
        """
        return x

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the Identity block
        (no operation performed).

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method does not perform any operations, as the Identity block does
            not change the number of channels.
        """


class Zero(nn.Module):
    """Zero Block Class.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int): Stride for the zero operation.

    Attributes:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int): Stride for the zero operation.
        is_zero (bool): Flag indicating whether this block performs a zero operation.

    Note:
        This class represents a block that performs a zero operation on the input
        tensor, adjusting the output tensor's dimensions based on the specified
        parameters.
    """

    def __init__(self, C_in: int, C_out: int, stride: int):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Zero block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with zeros, adjusted based on block parameters.

        Note:
            This method performs a forward pass through the Zero block,
            applying a zero operation to the input tensor and adjusting its dimensions
            accordingly.
        """
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            return x[:, :, :: self.stride, :: self.stride].mul(0.0)

        shape = list(x.shape)
        shape[1] = self.C_out
        zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
        return zeros

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,  # noqa: ARG002
        device: torch.device = DEVICE,  # noqa: ARG002
    ) -> None:
        """Change the number of input and output channels in the Zero block
        (no operation performed).

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method does not perform any operations, as the Zero block does not
            change the number of channels.
        """
        assert (k is not None) or (num_channels_to_add is not None)
        if k is not None:
            self.C_in = int(self.C_in // k)
            self.C_out = int(self.C_out // k)
        if num_channels_to_add is not None:
            self.C_in += num_channels_to_add
            self.C_out += num_channels_to_add

    def extra_repr(self) -> str:
        """Return an informative string representation of the Zero block.

        Returns:
            str: A string containing information about the block's input and output
            channel sizes and stride.

        Note:
            This method constructs a human-readable string representation of the block.
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

    Note:
        This class represents a factorized reduction block that adjusts the dimensions
        of the input tensor based on the specified parameters, including stride and
        channel sizes.
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
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    Conv2DLoRA(
                        C_in,
                        C_outs[i],
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=not affine,
                    )
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = Conv2DLoRA(
                C_in, C_out, kernel_size=1, stride=stride, padding=0, bias=not affine
            )
        else:
            raise ValueError(f"Invalid stride: {stride}")
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Factorized Reduce block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after factorized reduction.

        Note:
            This method performs a forward pass through the Factorized Reduce block,
            applying factorized reduction operations based on the specified stride.
        """
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
        """Change the number of input and output channels in the Factorized Reduce
        block.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method dynamically changes the number of output channels in the block's
            convolutional layers and BatchNorm.
        """
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
            raise ValueError(f"Invalid stride: {self.stride}")

    def deactivate_lora(self) -> None:
        if self.stride == 2:
            for i in range(2):
                self.convs[i].deactivate_lora()
        elif self.stride == 1:
            self.conv.deactivate_lora()
        else:
            raise ValueError(f"Invalid stride: {self.stride}")

    def toggle_lora(self) -> None:
        if self.stride == 2:
            for i in range(2):
                self.convs[i].toggle_lora()
        elif self.stride == 1:
            self.conv.toggle_lora()
        else:
            raise ValueError(f"Invalid stride: {self.stride}")

    def extra_repr(self) -> str:
        """Return an informative string representation of the Factorized Reduce block.

        Returns:
            str: A string containing information about the block's input and output
            channel sizes and stride.

        Note:
            This method constructs a human-readable string representation of the block.
        """
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


OLES_OPS = [Zero, Pooling, DualSepConv, SepConv, Identity, ReLUConvBN]
