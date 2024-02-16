from __future__ import annotations

import torch
from torch import nn

from confopt.utils.reduce_channels import (
    reduce_bn_features,
    reduce_conv_channels,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
OPS = {
    "none": lambda C, stride, affine: Zero(stride),  # noqa: ARG005
    "avg_pool_3x3": lambda C, stride, affine: Pooling(C, stride, "avg", affine=affine),
    "max_pool_3x3": lambda C, stride, affine: Pooling(C, stride, "max", affine=affine),
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
        """ReLU-Convolution-BatchNorm Block Class.

        Args:
            C_in (int): Number of input channels.
            C_out (int): Number of output channels.
            kernel_size (int or tuple[int, int]): Size of the convolutional kernel.
            stride (int or tuple[int, int]): Stride for the convolution operation.
            padding (int or tuple[int, int]): Padding for the convolution operation.
            affine (bool): Whether to apply affine transformations in BatchNorm.

        Attributes:
            op (nn.Sequential): Sequential block containing ReLU, Convolution,
            and BatchNorm operations.

        Note:
            This class represents a ReLU-Convolution-BatchNorm block commonly used in
            neural network architectures.
        """
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
        """Forward pass through the ReLUConvBN block.

        Args:
            x (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the ReLUConvBN block, applying
            ReLU activation,
            convolution, and BatchNorm operations to the input tensor.
        """
        return self.op(x)  # type: ignore

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the ReLUConvBN block.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method dynamically changes the number of output channels in the
            ReLUConvBN block.
        """
        self.op[1] = reduce_conv_channels(self.op[1], k, device)
        self.op[2] = reduce_bn_features(self.op[2], k, device)


class Pooling(nn.Module):
    def __init__(
        self,
        C: int,
        stride: int | tuple[int, int],
        mode: str,
        affine: bool = False,
    ) -> None:
        """Pooling Block Class.

        Args:
            C (int): Number of channels.
            stride (int or tuple[int, int]): Stride for the pooling operation.
            mode (str): Pooling mode, either "avg" for average pooling or "max" for
            max pooling.
            affine (bool, optional): Whether to apply affine transformations in
            BatchNorm (if preprocess is used). Defaults to True.

        Attributes:
            op (nn.Sequential): The pooling operation used inside this operation

        Note:
            This class represents a pooling block with optional mode.
        """
        super().__init__()
        if mode == "avg":
            op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            op = nn.MaxPool2d(3, stride=stride, padding=1)  # type: ignore
        else:
            raise ValueError(f"Invalid mode={mode} in POOLING")
        self.op = nn.Sequential(op, nn.BatchNorm2d(C, affine=affine))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Pooling block.

        Args:
            inputs (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the Pooling block
            applying pooling based on the specified mode.
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
        self.op[1] = reduce_bn_features(self.op[1], k, device)


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
        """Dilated Convolution operation.

        This class defines a Dilated Convolution operation, which consists of two
        convolutional layers with different dilation rates. It is commonly used in
        neural network architectures for various tasks.

        Args:
            C_in (int): Number of input channels.
            C_out (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for the convolution.
            padding (int): Padding for the convolution.
            dilation (int): Dilation factor for the convolution operation.
            affine (bool): If True, use affine transformations in Batch Normalization.

        Attributes:
            op (nn.Sequential): Sequential Block containing ReLU, Conv2d and
            BatchNorm2d.
        """
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
        """Perform a forward pass through the Dilated Convolution operation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Dilated Convolution.
        """
        return self.op(x)  # type: ignore

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the DilConv's ops.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """
        self.op[1] = reduce_conv_channels(self.op[1], k, device)
        self.op[2] = reduce_conv_channels(self.op[2], k, device)
        self.op[3] = reduce_bn_features(self.op[3], k, device)


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
        """Separable Convolution-BatchNorm Block Class.

        Args:
            C_in (int): Number of input channels.
            C_out (int): Number of output channels.
            kernel_size (int or tuple[int, int]): Size of the convolutional kernel.
            stride (int or tuple[int, int]): Stride for the convolution operation.
            padding (int or tuple[int, int]): Padding for the convolution operation.
            dilation (int or tuple[int, int]): Dilation rate for the convolution
            operation.
            affine (bool): Whether to apply affine transformations in BatchNorm.

        Attributes:
            op (nn.Sequential): Sequential block containing ReLU, Depthwise Convolution,
                Pointwise Convolution, and BatchNorm operations.

        Note:
            This class represents a separable convolutional block, commonly used in
            neural network architectures.
        """
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
        """Perform a forward pass through the Seperated Convolution operation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Dilated Convolution.
        """
        return self.op(x)  # type: ignore

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the SepConv's ops.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """
        self.op[1] = reduce_conv_channels(self.op[1], k, device)
        self.op[2] = reduce_conv_channels(self.op[2], k, device)
        self.op[3] = reduce_bn_features(self.op[3], k, device)
        self.op[5] = reduce_conv_channels(self.op[5], k, device)
        self.op[6] = reduce_conv_channels(self.op[6], k, device)
        self.op[7] = reduce_bn_features(self.op[7], k, device)


class Identity(nn.Module):
    def __init__(self) -> None:
        """Identity Block Class.

        Note:
            This class represents an identity block, which simply passes the input
            tensor through without any changes.

        Attributes:
            None
        """
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

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the Identity block
        (no operation performed).

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method does not perform any operations, as the Identity block does
            not change the number of channels.
        """


class Zero(nn.Module):
    def __init__(self, stride: int) -> None:
        """Zero Block Class.

        Args:
            stride (int): Stride for the zero operation.

        Attributes:
            stride (int): Stride for the zero operation.

        Note:
            This class represents a block that performs a zero operation on the input
            tensor, adjusting the output tensor's dimensions based on the specified
            parameters.
        """
        super().__init__()
        self.stride = stride

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
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the Zero block
        (no operation performed).

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method does not perform any operations, as the Zero block does not
            change the number of channels.
        """


class FactorizedReduce(nn.Module):
    def __init__(self, C_in: int, C_out: int, affine: bool = True):
        """Factorized Reduce Block Class.

        Args:
            C_in (int): Number of input channels.
            C_out (int): Number of output channels.
            affine (bool): Whether to apply affine transformations in BatchNorm.

        Attributes:
            relu (nn.ReLU): ReLU activation layer.
            conv1 (nn.Conv2d): First Conv2d layer for factorized reduction.
            conv2 (nn.Conv2d): Second Conv2d layer for factorized reduction.
            bn (nn.BatchNorm2d): BatchNorm layer.
        """
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Factorized Reduce block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after factorized reduction.
        """
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Change the number of input and output channels in the Factorized Reduce
        block.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.

        Note:
            This method dynamically changes the number of output channels in the block's
            convolutional layers and BatchNorm.
        """
        self.conv_1 = reduce_conv_channels(self.conv_1, k, device)
        self.conv_2 = reduce_conv_channels(self.conv_2, k, device)
        self.bn = reduce_bn_features(self.bn, k, device)


class Conv7x1Conv1x7BN(nn.Module):
    def __init__(
        self,
        C: int,
        stride: int,
        affine: bool = True,
    ) -> None:
        """Convolution operation using 7x1 and 1x7 kernels with Batch Normalization.

        Args:
            C (int): Number of input channels.
            stride (int): Stride for the convolution.
            affine (bool): If True, use affine transformations in Batch Normalization.

        Note: This class defines a convolution operation that uses two different
        convolutional kernels, 7x1 and 1x7, with Batch Normalization. This operation
        can be useful in neural network architectures for various tasks.
        """
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(C, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the Convolution operation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the convolution operation.

        """
        return self.op(x)

    def change_channel_size(self, k: float, device: torch.device = DEVICE) -> None:
        """Modify the channel sizes of the operation by reducing them to 'k' channels.

        Args:
            k (int): The new number of channels.
            device (torch.device): The target device (default is 'DEVICE').

        Note: This method is used for architectural search and adjusts the number of
        channels in the Convolution operation.
        """
        self.op[1] = reduce_conv_channels(self.op[1], k, device)
        self.op[2] = reduce_conv_channels(self.op[2], k, device)
        self.op[3] = reduce_bn_features(self.op[3], k, device)
