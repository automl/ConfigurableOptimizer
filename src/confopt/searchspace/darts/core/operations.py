from __future__ import annotations

import torch
from torch import nn

from confopt.oneshot.weightentangler import (
    ConvolutionalWEModule,
    WeightEntanglementSequential,
)
from confopt.searchspace.common import Conv2DLoRA
import confopt.utils.change_channel_size as ch

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
        self.C_in = C_in
        self.C_out = C_out
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv2DLoRA(
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
        if k is not None:
            self.C_in = int(k * self.C_in)
            self.C_out = int(k * self.C_out)

    def increase_in_channel_size(
        self, num_channels_to_add: int, device: torch.device = DEVICE
    ) -> None:
        self.op[1], _ = ch.increase_in_channel_size_conv(
            self.op[1], num_channels_to_add, device=device
        )
        self.C_in += num_channels_to_add

    def increase_out_channel_size(
        self, num_channels_to_add: int, device: torch.device = DEVICE
    ) -> None:
        self.op[1], index = ch.increase_out_channel_size_conv(
            self.op[1], num_channels_to_add, device=device
        )
        self.op[2], _ = ch.increase_num_features_bn(
            self.op[2], num_channels_to_add, index=index, device=device
        )
        self.C_out += num_channels_to_add

    def change_stride_size(self, new_stride: int) -> None:
        for op in self.op:
            if hasattr(op, "stride"):
                op.stride = (new_stride, new_stride)

    def activate_lora(self, r: int) -> None:
        self.op[1].activate_lora(r)

    def deactivate_lora(self) -> None:
        self.op[1].deactivate_lora()

    def toggle_lora(self) -> None:
        self.op[1].toggle_lora()


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

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,  # noqa: ARG002
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the Pooling block's batch
        norm features.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """
        self.op[1], _ = ch.change_features_bn(
            self.op[1], k=k, num_channels_to_add=num_channels_to_add, device=device
        )

    def change_stride_size(self, new_stride: int) -> None:
        for op in self.op:
            if hasattr(op, "stride"):
                op.stride = new_stride


class DilConv(ConvolutionalWEModule):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int | tuple[int, int],
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
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        )
        self.stride = stride
        self.op = WeightEntanglementSequential(
            nn.ReLU(inplace=False),
            Conv2DLoRA(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            Conv2DLoRA(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

        self.__post__init__()

    def mark_entanglement_weights(self) -> None:
        self.op[1].can_entangle_weight = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the Dilated Convolution operation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Dilated Convolution.
        """
        return self.op(x)  # type: ignore

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the DilConv's ops.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """
        if k is not None:
            if k >= 1:
                self.op[1] = ch.reduce_conv_channels(self.op[1], k, device)
                self.op[2] = ch.reduce_conv_channels(self.op[2], k, device)
                self.op[3] = ch.reduce_bn_features(self.op[3], k, device)
                return
            num_channels_to_add_in = int(max(1, self.op[1].in_channels * (1 / k - 1)))
            num_channels_to_add_out = int(max(1, self.op[1].out_channels * (1 / k - 1)))
        if num_channels_to_add:
            num_channels_to_add_in = num_channels_to_add
            num_channels_to_add_out = num_channels_to_add
        if new_cell:
            self.op[1], _ = ch.increase_in_channel_size_conv(
                self.op[1], num_channels_to_add_in
            )
        self.op[1], index = ch.increase_out_channel_size_conv(
            self.op[1], num_channels_to_add_out
        )
        self.op[1].conv.groups += num_channels_to_add_in
        self.op[2], _ = ch.increase_in_channel_size_conv(
            self.op[2], num_channels_to_add_in, index=index
        )
        self.op[2], index = ch.increase_out_channel_size_conv(
            self.op[2], num_channels_to_add_out
        )
        self.op[3], _ = ch.increase_num_features_bn(
            self.op[3], num_channels_to_add_out, index=index
        )

    def change_stride_size(self, new_stride: int) -> None:
        self.stride = new_stride
        if hasattr(self.op[1], "conv") and hasattr(self.op[1].conv, "stride"):
            self.op[1].conv.stride = (new_stride, new_stride)
        else:
            self.op[1].stride = (new_stride, new_stride)

    def activate_lora(self, r: int) -> None:
        self.op[1].activate_lora(r)
        self.op[2].activate_lora(r)

    def deactivate_lora(self) -> None:
        self.op[1].deactivate_lora()
        self.op[2].deactivate_lora()

    def toggle_lora(self) -> None:
        self.op[1].toggle_lora()
        self.op[2].toggle_lora()


class SepConv(ConvolutionalWEModule):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int | tuple[int, int],
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
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        )
        self.stride = stride
        self.op = WeightEntanglementSequential(
            nn.ReLU(inplace=False),
            Conv2DLoRA(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            Conv2DLoRA(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            Conv2DLoRA(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            Conv2DLoRA(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

        self.__post__init__()

    def mark_entanglement_weights(self) -> None:
        self.op[1].can_entangle_weight = True
        self.op[5].can_entangle_weight = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the Seperated Convolution operation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Dilated Convolution.
        """
        return self.op(x)  # type: ignore

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        """Change the number of input and output channels in the SepConv's ops.

        Args:
            k (int): The new number of input and output channels would be 1/k of the
            original size.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device, optional): The device to which the operations are
            moved. Defaults to DEVICE.
        """
        if k is not None:
            if k > 1:
                self.op[1] = ch.reduce_conv_channels(self.op[1], k, device)
                self.op[2] = ch.reduce_conv_channels(self.op[2], k, device)
                self.op[3] = ch.reduce_bn_features(self.op[3], k, device)
                self.op[5] = ch.reduce_conv_channels(self.op[5], k, device)
                self.op[6] = ch.reduce_conv_channels(self.op[6], k, device)
                self.op[7] = ch.reduce_bn_features(self.op[7], k, device)
                return
            num_channels_to_add_in = int(max(1, self.op[1].in_channels * (1 / k - 1)))
            num_channels_to_add_out = int(max(1, self.op[1].out_channels * (1 / k - 1)))
        if num_channels_to_add:
            num_channels_to_add_in = num_channels_to_add
            num_channels_to_add_out = num_channels_to_add

        if new_cell:
            self.op[1], _ = ch.increase_in_channel_size_conv(
                self.op[1], num_channels_to_add_in
            )
        self.op[1], index = ch.increase_out_channel_size_conv(
            self.op[1], num_channels_to_add_out
        )
        self.op[1].conv.groups += num_channels_to_add_in
        self.op[2], _ = ch.increase_in_channel_size_conv(
            self.op[2], num_channels_to_add_in, index=index
        )
        self.op[2], index = ch.increase_out_channel_size_conv(
            self.op[2], num_channels_to_add_out
        )
        self.op[3], _ = ch.increase_num_features_bn(
            self.op[3], num_channels_to_add_out, index=index
        )

        if new_cell:
            self.op[5], _ = ch.increase_in_channel_size_conv(
                self.op[5], num_channels_to_add_in
            )
        self.op[5], index = ch.increase_out_channel_size_conv(
            self.op[5], num_channels_to_add_out
        )
        self.op[5].conv.groups += num_channels_to_add_in
        self.op[6], _ = ch.increase_in_channel_size_conv(
            self.op[6], num_channels_to_add_in, index=index
        )
        self.op[6], index = ch.increase_out_channel_size_conv(
            self.op[6], num_channels_to_add_out
        )
        self.op[7], _ = ch.increase_num_features_bn(
            self.op[7], num_channels_to_add_out, index=index
        )

    def change_stride_size(self, new_stride: int) -> None:
        self.stride = new_stride
        if hasattr(self.op[1], "conv") and hasattr(self.op[1].conv, "stride"):
            self.op[1].conv.stride = (new_stride, new_stride)
        elif hasattr(self.op[1], "stride"):
            self.op[1].stride = (new_stride, new_stride)

    def activate_lora(self, r: int) -> None:
        self.op[1].activate_lora(r)
        self.op[2].activate_lora(r)
        self.op[5].activate_lora(r)
        self.op[6].activate_lora(r)

    def deactivate_lora(self) -> None:
        self.op[1].activate_lora()
        self.op[2].activate_lora()
        self.op[5].activate_lora()
        self.op[6].activate_lora()

    def toggle_lora(self) -> None:
        self.op[1].toggle_lora()
        self.op[2].toggle_lora()
        self.op[5].toggle_lora()
        self.op[6].toggle_lora()


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

    def change_stride_size(self, new_stride: int) -> None:
        pass


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

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
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

    def change_stride_size(self, new_stride: int) -> None:
        self.stride = new_stride


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
        # assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = Conv2DLoRA(
            C_in, C_out // 2, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.conv_2 = Conv2DLoRA(
            C_in, C_out - C_out // 2, kernel_size=1, stride=2, padding=0, bias=False
        )
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
        if k is not None:
            if k >= 1:
                self.conv_1 = ch.reduce_conv_channels(self.conv_1, k, device)
                self.conv_2 = ch.reduce_conv_channels(self.conv_2, k, device)
                self.bn = ch.reduce_bn_features(self.bn, k, device)
                self.C_in //= int(k)
                self.C_out //= int(k)
                return
            num_channels_to_add_in = int(max(1, self.C_in * (1 / k - 1)))
            num_channels_to_add_out = int(max(1, self.C_out * (1 / k - 1) // 2))
        if num_channels_to_add:
            num_channels_to_add_in = num_channels_to_add
            num_channels_to_add_out = num_channels_to_add

        self.conv_1, _ = ch.increase_in_channel_size_conv(
            self.conv_1, num_channels_to_add_in
        )
        self.conv_1, index1 = ch.increase_out_channel_size_conv(
            self.conv_1, num_channels_to_add_out
        )
        self.conv_2, _ = ch.increase_in_channel_size_conv(
            self.conv_2, num_channels_to_add_in
        )
        self.conv_2, index2 = ch.increase_out_channel_size_conv(
            self.conv_2, num_channels_to_add_out
        )
        self.bn, _ = ch.increase_num_features_bn(
            self.bn, num_channels_to_add_in, index=torch.cat([index1, index2])
        )
        self.C_in += num_channels_to_add_in
        self.C_out += num_channels_to_add_out * 2

    def increase_in_channel_size(
        self, num_channels_to_add: int, device: torch.device = DEVICE
    ) -> None:
        self.conv_1, _ = ch.increase_in_channel_size_conv(
            self.conv_1, num_channels_to_add, device
        )
        self.conv_2, _ = ch.increase_in_channel_size_conv(
            self.conv_2, num_channels_to_add, device
        )
        self.C_in += num_channels_to_add

    def increase_out_channel_size(
        self, num_channels_to_add: int, device: torch.device = DEVICE
    ) -> None:
        self.conv_1, index1 = ch.increase_out_channel_size_conv(
            self.conv_1, num_channels_to_add - num_channels_to_add // 2, device=device
        )
        self.conv_2, index2 = ch.increase_out_channel_size_conv(
            self.conv_2, num_channels_to_add // 2, device=device
        )
        self.bn, _ = ch.increase_num_features_bn(
            self.bn,
            num_channels_to_add,
            index=torch.cat([index1, index2]),
            device=device,
        )
        self.C_out += num_channels_to_add

    def change_stride_size(self, new_stride: int) -> None:
        pass

    def activate_lora(self, r: int) -> None:
        self.conv_1.activate_lora(r)
        self.conv_2.activate_lora(r)

    def deactivate_lora(self) -> None:
        self.conv_1.deactivate_lora()
        self.conv_2.deactivate_lora()


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

    def change_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
        new_cell: bool = False,
        device: torch.device = DEVICE,
    ) -> None:
        """Modify the channel sizes of the operation by reducing them to 'k' channels.

        Args:
            k (int): The new number of channels.
            num_channels_to_add (int): The number of channels to add to the operation.
            new_cell (bool): Whether change is for creating a new cell.
            device (torch.device): The target device (default is 'DEVICE').

        Note: This method is used for architectural search and adjusts the number of
        channels in the Convolution operation.
        """

    def change_stride_size(self, new_stride: int) -> None:
        self.op[1].stride = (1, new_stride)
        self.op[2].stride = (new_stride, 1)


OLES_OPS = [Zero, Pooling, Identity, SepConv, DilConv]
