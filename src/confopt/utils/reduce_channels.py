from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common import Conv2DLoRA

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def change_channel_size_conv(
    conv2d_layer: nn.Conv2d | Conv2DLoRA, k: float, device: torch.device = DEVICE
) -> nn.Conv2d | Conv2DLoRA:
    if k > 1:
        return increase_conv_channels(conv2d_layer, k, device)
    return reduce_conv_channels(conv2d_layer, k, device)


def increase_conv_channels(
    conv2d_layer: nn.Conv2d | Conv2DLoRA,
    k: float,
    device: torch.device = DEVICE,
) -> nn.Conv2d | Conv2DLoRA:
    if not isinstance(conv2d_layer, (nn.Conv2d, Conv2DLoRA)):
        raise TypeError("Input must be a nn.Conv2d or a LoRA wrapped conv2d layer.")

    # Get the number of input and output channels of the original conv2d
    in_channels = conv2d_layer.in_channels
    out_channels = conv2d_layer.out_channels

    # Calculate the new number of output channels
    new_in_channels = int(max(1, in_channels // k))
    new_out_channels = int(max(1, out_channels // k))
    # Create a new conv2d layer with the increased number of channels
    if isinstance(conv2d_layer, Conv2DLoRA):
        new_groups = new_in_channels if conv2d_layer.conv.groups != 1 else 1
        increased_conv2d = Conv2DLoRA(
            new_in_channels,
            new_out_channels,
            kernel_size=conv2d_layer.kernel_size,
            stride=conv2d_layer.conv.stride,
            padding=conv2d_layer.conv.padding,
            dilation=conv2d_layer.conv.dilation,
            groups=new_groups,
            bias=conv2d_layer.conv.bias is not None,
            r=conv2d_layer.r,
            lora_alpha=conv2d_layer.lora_alpha,
            lora_dropout=conv2d_layer.lora_dropout_p,
            merge_weights=conv2d_layer.merge_weights,
        ).to(device)
    else:
        new_groups = new_in_channels if conv2d_layer.groups != 1 else 1
        increased_conv2d = nn.Conv2d(
            new_in_channels,
            new_out_channels,
            conv2d_layer.kernel_size,
            conv2d_layer.stride,
            conv2d_layer.padding,
            conv2d_layer.dilation,
            new_groups,
            conv2d_layer.bias is not None,
        ).to(device)

    # Copy the weights and biases from the original conv2d to the new one
    increased_conv2d.weight.data[
        :out_channels, :in_channels, :, :
    ] = conv2d_layer.weight.data[:out_channels, :in_channels, :, :].clone()
    if conv2d_layer.bias is not None:
        increased_conv2d.bias.data[:out_channels] = conv2d_layer.bias.data[
            :out_channels
        ].clone()

    return increased_conv2d


def reduce_conv_channels(
    conv2d_layer: nn.Conv2d | Conv2DLoRA, k: float, device: torch.device = DEVICE
) -> nn.Conv2d:
    if not isinstance(conv2d_layer, (nn.Conv2d, Conv2DLoRA)):
        raise TypeError("Input must be a nn.Conv2d or a LoRA wrapped conv2d layer.")

    # Get the number of input and output channels of the original conv2d
    in_channels = conv2d_layer.in_channels
    out_channels = conv2d_layer.out_channels

    # Calculate the new number of output channels
    new_in_channels = int(max(1, in_channels // k))
    new_out_channels = int(max(1, out_channels // k))
    # Create a new conv2d layer with the reduced number of channels
    if isinstance(conv2d_layer, Conv2DLoRA):
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
            r=conv2d_layer.r,
            lora_alpha=conv2d_layer.lora_alpha,
            lora_dropout=conv2d_layer.lora_dropout_p,
            merge_weights=conv2d_layer.merge_weights,
        ).to(device)

        # Copy the weights and bias of conv2d layer and LoRA layers
        reduced_conv2d.conv.weight.data[
            :new_out_channels, :new_in_channels, :, :
        ] = conv2d_layer.conv.weight.data[
            :new_out_channels, :new_in_channels, :, :
        ].clone()
        if conv2d_layer.conv.bias is not None:
            reduced_conv2d.conv.bias.data[
                :new_out_channels
            ] = conv2d_layer.conv.bias.data[:new_out_channels].clone()

        if conv2d_layer.r > 0:
            kernel_size = conv2d_layer.kernel_size
            reduced_conv2d.lora_A.data[
                :, : new_in_channels * kernel_size
            ] = conv2d_layer.lora_A.data[:, : new_in_channels * kernel_size].clone()
            reduced_conv2d.lora_B.data[
                : new_out_channels * kernel_size, :
            ] = conv2d_layer.lora_B.data[: new_out_channels * kernel_size, :].clone()

    else:
        new_groups = new_in_channels if conv2d_layer.groups != 1 else 1
        reduced_conv2d = nn.Conv2d(
            new_in_channels,
            new_out_channels,
            conv2d_layer.kernel_size,
            conv2d_layer.stride,
            conv2d_layer.padding,
            conv2d_layer.dilation,
            new_groups,
            conv2d_layer.bias is not None,
        ).to(device)

        # Copy the weights and biases from the original conv2d to the new one
        reduced_conv2d.weight.data[
            :new_out_channels, :new_in_channels, :, :
        ] = conv2d_layer.weight.data[:new_out_channels, :new_in_channels, :, :].clone()
        if conv2d_layer.bias is not None:
            reduced_conv2d.bias.data[:new_out_channels] = conv2d_layer.bias.data[
                :new_out_channels
            ].clone()

    return reduced_conv2d


def change_features_bn(
    batchnorm_layer: nn.BatchNorm2d, k: float, device: torch.device = DEVICE
) -> nn.BatchNorm2d:
    if k > 1:
        return increase_bn_features(batchnorm_layer, k, device)
    return reduce_bn_features(batchnorm_layer, k, device)


def increase_bn_features(
    batchnorm_layer: nn.BatchNorm2d, k: float, device: torch.device = DEVICE
) -> nn.BatchNorm2d:
    if not isinstance(batchnorm_layer, nn.BatchNorm2d):
        raise TypeError("Input must be a nn.BatchNorm2d layer.")

    # Get the number of features in the original BatchNorm2d
    num_features = batchnorm_layer.num_features

    # Calculate the new number of features
    new_num_features = int(max(1, num_features // k))

    # Create a new BatchNorm2d layer with the increased number of features
    increased_batchnorm = nn.BatchNorm2d(
        new_num_features,
        eps=batchnorm_layer.eps,
        momentum=batchnorm_layer.momentum,
        affine=batchnorm_layer.affine,
        track_running_stats=batchnorm_layer.track_running_stats,
    ).to(device)

    if batchnorm_layer.affine:
        increased_batchnorm.weight.data[:num_features] = batchnorm_layer.weight.data[
            :num_features
        ].clone()
        increased_batchnorm.bias.data[:num_features] = batchnorm_layer.bias.data[
            :num_features
        ].clone()

    return increased_batchnorm


def reduce_bn_features(
    batchnorm_layer: nn.BatchNorm2d, k: float, device: torch.device = DEVICE
) -> nn.BatchNorm2d:
    if not isinstance(batchnorm_layer, nn.BatchNorm2d):
        raise TypeError("Input must be a nn.BatchNorm2d layer.")

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
