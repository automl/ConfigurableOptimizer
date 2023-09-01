from __future__ import annotations

from torch import nn


def reduce_conv_channels(conv2d_layer: nn.Conv2d, k: int) -> nn.Conv2d:
    if not isinstance(conv2d_layer, nn.Conv2d):
        raise ValueError("Input must be a nn.Conv2d layer.")

    # Get the number of input and output channels of the original conv2d
    in_channels = conv2d_layer.in_channels
    out_channels = conv2d_layer.out_channels

    # Calculate the new number of output channels
    new_out_channels = max(1, out_channels // k)

    # Create a new conv2d layer with the reduced number of channels
    reduced_conv2d = nn.Conv2d(
        in_channels,
        new_out_channels,
        conv2d_layer.kernel_size,
        conv2d_layer.stride,
        conv2d_layer.padding,
        conv2d_layer.dilation,
        conv2d_layer.groups,
        conv2d_layer.bias is not None,
    )

    # Copy the weights and biases from the original conv2d to the new one
    reduced_conv2d.weight.data[:new_out_channels, :, :, :] = conv2d_layer.weight.data[
        :new_out_channels, :, :, :
    ].clone()
    if conv2d_layer.bias is not None:
        reduced_conv2d.bias.data[:new_out_channels] = conv2d_layer.bias.data[
            :new_out_channels
        ].clone()

    return reduced_conv2d


def reduce_bn_features(batchnorm_layer: nn.BatchNorm2d, k: int) -> nn.BatchNorm2d:
    if not isinstance(batchnorm_layer, nn.BatchNorm2d):
        raise ValueError("Input must be a nn.BatchNorm2d layer.")

    # Get the number of features in the original BatchNorm2d
    num_features = batchnorm_layer.num_features

    # Calculate the new number of features
    new_num_features = max(1, num_features // k)

    # Create a new BatchNorm2d layer with the reduced number of features
    reduced_batchnorm = nn.BatchNorm2d(
        new_num_features,
        eps=batchnorm_layer.eps,
        momentum=batchnorm_layer.momentum,
        affine=batchnorm_layer.affine,
        track_running_stats=batchnorm_layer.track_running_stats,
    )

    # Copy the weight and bias from the original BatchNorm2d to the new one
    if batchnorm_layer.affine:
        reduced_batchnorm.weight.data[:new_num_features] = batchnorm_layer.weight.data[
            :new_num_features
        ].clone()
        reduced_batchnorm.bias.data[:new_num_features] = batchnorm_layer.bias.data[
            :new_num_features
        ].clone()

    return reduced_batchnorm
