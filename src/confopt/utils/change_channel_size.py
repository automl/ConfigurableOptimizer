from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common import Conv2DLoRA

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def change_channel_size_conv(
    conv: Conv2DLoRA,
    k: float | None = None,
    num_channels_to_add: int | None = None,
    index: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> tuple[nn.Conv2d | Conv2DLoRA, torch.Tensor | None]:
    if k is not None and k >= 1:
        return reduce_conv_channels(conv, k=k, device=device), index

    return increase_conv_channels(
        conv,
        k=k,
        num_channels_to_add=num_channels_to_add,
        device=device,
    )


def reduce_conv_channels(
    conv2d_layer: Conv2DLoRA, k: float, device: torch.device = DEVICE
) -> Conv2DLoRA:
    assert isinstance(conv2d_layer, Conv2DLoRA)

    # Get the number of input and output channels of the original conv2d
    in_channels = conv2d_layer.in_channels
    out_channels = conv2d_layer.out_channels

    # Calculate the new number of output channels
    new_in_channels = int(max(1, in_channels // k))
    new_out_channels = int(max(1, out_channels // k))
    # Create a new conv2d layer with the reduced number of channels
    new_groups = new_in_channels if conv2d_layer.conv.groups != 1 else 1
    reduced_conv2d = Conv2DLoRA(
        new_in_channels,
        new_out_channels,
        conv2d_layer.kernel_size,
        stride=conv2d_layer.conv.stride,
        padding=conv2d_layer.conv.padding,
        dilation=conv2d_layer.conv.dilation,
        groups=new_groups,
        bias=conv2d_layer.conv.bias is not None,
    ).to(device)
    if conv2d_layer.r > 0:
        reduced_conv2d.activate_lora(
            r=conv2d_layer.r,
            lora_alpha=conv2d_layer.lora_alpha,
            lora_dropout_rate=conv2d_layer.lora_dropout_p,
            merge_weights=conv2d_layer.merge_weights,
        )

    # Copy the weights and bias of conv2d layer and LoRA layers
    reduced_conv2d.conv.weight.data[
        :new_out_channels, :new_in_channels, :, :
    ] = conv2d_layer.conv.weight.data[:new_out_channels, :new_in_channels, :, :].clone()
    if conv2d_layer.conv.bias is not None:
        reduced_conv2d.conv.bias.data[:new_out_channels] = conv2d_layer.conv.bias.data[
            :new_out_channels
        ].clone()

    if conv2d_layer.r > 0:
        kernel_size = conv2d_layer.kernel_size
        reduced_conv2d.lora_A.data[
            :, : new_in_channels * kernel_size
        ] = conv2d_layer.lora_A.data[:, : new_in_channels * kernel_size].clone()
        reduced_conv2d.lora_B.data[
            : new_out_channels * kernel_size, :
        ] = conv2d_layer.lora_B.data[: new_out_channels * kernel_size, :].clone()

    return reduced_conv2d


def increase_conv_channels(
    conv: Conv2DLoRA,
    k: float | None = None,
    num_channels_to_add: int | None = None,
    device: torch.device = DEVICE,
) -> tuple[Conv2DLoRA, torch.Tensor]:
    assert isinstance(conv, Conv2DLoRA)

    if k is not None:
        num_channels_to_add = conv.in_channels * int(1 / k - 1)
    assert num_channels_to_add
    increased_conv, _ = increase_in_channel_size_conv(conv, num_channels_to_add)
    increased_conv, out_index = increase_out_channel_size_conv(
        increased_conv, num_channels_to_add
    )
    return increased_conv.to(device=device), out_index


def increase_in_channel_size_conv(
    conv: Conv2DLoRA,
    num_channels_to_add: int,
    index: None | torch.Tensor = None,
    device: torch.device = DEVICE,
) -> tuple[Conv2DLoRA, torch.Tensor]:
    assert isinstance(conv, Conv2DLoRA)
    conv_weights = conv.conv.weight
    in_channels = conv_weights.size(1)

    if not torch.is_tensor(index):
        index = torch.randint(low=0, high=in_channels, size=(num_channels_to_add,))

    conv.conv.weight = nn.Parameter(
        torch.cat([conv_weights, conv_weights[:, index, :, :].clone()], dim=1),
        requires_grad=True,
    )
    conv.weight = conv.conv.weight
    conv.in_channels += num_channels_to_add
    conv.conv.in_channels += num_channels_to_add
    if hasattr(conv_weights, "in_index"):
        conv.weight.in_index.append(index)
    else:
        conv.weight.in_index = [index]

    conv.weight.t = "conv"
    if hasattr(conv_weights, "out_index"):
        conv.weight.out_index = conv_weights.out_index
    conv.weight.raw_id = (
        conv_weights.raw_id if hasattr(conv_weights, "raw_id") else id(conv_weights)
    )
    return conv.to(device=device), index


def increase_out_channel_size_conv(
    conv: Conv2DLoRA,
    num_channels_to_add: int,
    index: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> tuple[Conv2DLoRA, torch.Tensor]:
    assert isinstance(conv, Conv2DLoRA)

    conv_weight = conv.weight
    out_channels = conv_weight.size(0)

    if not torch.is_tensor(index):
        index = torch.randint(low=0, high=out_channels, size=(num_channels_to_add,))

    conv.conv.weight = nn.Parameter(
        torch.cat([conv_weight, conv_weight[index, :, :, :].clone()], dim=0),
        requires_grad=True,
    )

    conv.weight = conv.conv.weight
    conv.out_channels += num_channels_to_add
    conv.conv.out_channels += num_channels_to_add
    if hasattr(conv_weight, "out_index"):
        conv.weight.out_index.append(index)
    else:
        conv.weight.out_index = [index]
    conv.weight.t = "conv"
    if hasattr(conv_weight, "in_index"):
        conv.weight.in_index = conv_weight.in_index
    conv.weight.raw_id = (
        conv_weight.raw_id if hasattr(conv_weight, "raw_id") else id(conv_weight)
    )
    return conv.to(device=device), index


def change_features_bn(
    batchnorm_layer: nn.BatchNorm2d,
    k: float | None = None,
    num_channels_to_add: int | None = None,
    index: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> tuple[nn.BatchNorm2d, torch.Tensor | None]:
    if k is not None and k >= 1:
        return reduce_bn_features(batchnorm_layer, k, device), index

    return increase_bn_features(
        batchnorm_layer,
        k=k,
        num_channels_to_add=num_channels_to_add,
        index=index,
        device=device,
    )


def reduce_bn_features(
    batchnorm_layer: nn.BatchNorm2d, k: float, device: torch.device = DEVICE
) -> nn.BatchNorm2d:
    assert isinstance(batchnorm_layer, nn.BatchNorm2d)

    # Get the number of features in the original BatchNorm2d
    num_features = batchnorm_layer.num_features

    # Calculate the new number of features
    new_num_features = int(max(1, num_features // k))

    # Create a new BatchNorm2d layer with the reduced number of features
    reduced_batchnorm = nn.BatchNorm2d(
        new_num_features,
        eps=batchnorm_layer.eps,
        momentum=batchnorm_layer.momentum,
        affine=batchnorm_layer.affine,
        track_running_stats=batchnorm_layer.track_running_stats,
    ).to(device)

    # Copy the weight and bias from the original BatchNorm2d to the new one
    if batchnorm_layer.affine:
        reduced_batchnorm.weight.data[:new_num_features] = batchnorm_layer.weight.data[
            :new_num_features
        ].clone()
        reduced_batchnorm.bias.data[:new_num_features] = batchnorm_layer.bias.data[
            :new_num_features
        ].clone()

    return reduced_batchnorm


def increase_bn_features(
    bn: nn.BatchNorm2d,
    k: float | None = None,
    num_channels_to_add: int | None = None,
    index: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> tuple[nn.BatchNorm2d, torch.Tensor]:
    assert isinstance(bn, nn.BatchNorm2d)

    if k is not None:
        num_channels_to_add = bn.num_features * int(1 / k - 1)
    assert num_channels_to_add
    wider_bn, index = increase_num_features_bn(bn, num_channels_to_add, index, device)

    return wider_bn.to(device), index


def increase_num_features_bn(
    bn: nn.BatchNorm2d,
    num_features_to_add: int,
    index: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> tuple[nn.BatchNorm2d, torch.Tensor]:
    assert isinstance(bn, nn.BatchNorm2d)

    running_mean = bn.running_mean
    running_var = bn.running_var
    if bn.affine:
        weight = bn.weight
        bias = bn.bias
    num_features = bn.num_features
    if index is None:
        index = torch.randint(low=0, high=num_features, size=(num_features_to_add,))
    bn.running_mean = torch.cat([running_mean, running_mean[index].clone()])
    bn.running_var = torch.cat([running_var, running_var[index].clone()])
    if bn.affine:
        bn.weight = nn.Parameter(
            torch.cat([weight, weight[index].clone()], dim=0), requires_grad=True
        )
        bn.bias = nn.Parameter(
            torch.cat([bias, bias[index].clone()], dim=0), requires_grad=True
        )
        if hasattr(bn.weight, "out_index"):
            bn.weight.out_index.append(index)
            bn.bias.out_index.append(index)
        else:
            bn.weight.out_index = [index]
            bn.bias.out_index = [index]
        bn.weight.t = "bn"
        bn.bias.t = "bn"
        bn.weight.raw_id = weight.raw_id if hasattr(weight, "raw_id") else id(weight)
        bn.bias.raw_id = bias.raw_id if hasattr(bias, "raw_id") else id(bias)
    bn.num_features += num_features_to_add
    return bn.to(device=device), index
