import unittest

import torch
from torch import nn
from typing_extensions import TypeAlias

from confopt.searchspace.common import Conv2DLoRA
from confopt.utils.change_channel_size import (
    change_channel_size_conv,
    increase_bn_features,
    increase_in_channel_size_conv,
    increase_out_channel_size_conv,
    reduce_bn_features,
    reduce_conv_channels,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LRSchedulerType: TypeAlias = torch.optim.lr_scheduler.LRScheduler


class TestChangeChannelSize(unittest.TestCase):

    def test_reduce_conv_channels(self) -> None:
        conv_layer = Conv2DLoRA(10, 20, kernel_size=3)
        k = 2
        reduced_conv = reduce_conv_channels(conv_layer, k=k)

        assert(reduced_conv.in_channels == conv_layer.in_channels // k)
        assert(reduced_conv.out_channels == conv_layer.out_channels // k)
        assert(reduced_conv.conv.in_channels == conv_layer.in_channels // k)
        assert(reduced_conv.conv.out_channels == conv_layer.out_channels // k)
        assert(reduced_conv.weight.size(1) == conv_layer.in_channels // k)
        assert(reduced_conv.weight.size(0) == conv_layer.out_channels // k)
        assert(
            reduced_conv.conv.weight.size(1) == conv_layer.conv.in_channels // k
        )
        assert(
            reduced_conv.conv.weight.size(0) == conv_layer.conv.out_channels // k
        )

    def test_increase_in_channel_size_conv(self) -> None:
        in_channels = 10
        out_channels = 20
        conv_layer = Conv2DLoRA(in_channels, out_channels, kernel_size=3)
        num_channels_to_add = 5

        increased_conv, index = increase_in_channel_size_conv(
            conv_layer, num_channels_to_add
        )

        assert(increased_conv.in_channels == in_channels + num_channels_to_add)
        assert(
            increased_conv.conv.in_channels == in_channels + num_channels_to_add
        )
        assert(
            increased_conv.weight.size(1) == in_channels + num_channels_to_add
        )
        assert(len(index) == num_channels_to_add)

    def test_increase_out_channel_size_conv(self) -> None:
        in_channels = 10
        out_channels = 20
        conv_layer = Conv2DLoRA(in_channels, out_channels, kernel_size=3)
        num_channels_to_add = 5

        increased_conv, index = increase_out_channel_size_conv(
            conv_layer, num_channels_to_add
        )

        assert(
            increased_conv.out_channels == out_channels + num_channels_to_add
        )
        assert(
            increased_conv.conv.out_channels == out_channels + num_channels_to_add
        )
        assert(
            increased_conv.weight.size(0) == out_channels + num_channels_to_add
        )
        assert(len(index) == num_channels_to_add)

    def test_change_channel_size_conv_increase(self) -> None:
        in_channels = 10
        out_channels = 20
        conv_layer = Conv2DLoRA(in_channels, out_channels, kernel_size=3)
        num_channels_to_add = 5

        changed_conv, _ = change_channel_size_conv(
            conv_layer, num_channels_to_add=num_channels_to_add
        )

        assert(changed_conv.in_channels == in_channels + num_channels_to_add)
        assert(changed_conv.out_channels == out_channels + num_channels_to_add)

    def test_change_channel_size_conv_reduce(self) -> None:
        conv_layer = Conv2DLoRA(10, 20, kernel_size=3)
        k = 2

        changed_conv, _ = change_channel_size_conv(conv_layer, k=k)

        assert(changed_conv.in_channels == conv_layer.in_channels // k)
        assert(changed_conv.out_channels == conv_layer.out_channels // k)

    def test_reduce_bn_features(self) -> None:
        bn_layer = nn.BatchNorm2d(10).to(DEVICE)
        k = 2
        reduced_bn_layer = reduce_bn_features(bn_layer, k, device=DEVICE)

        assert isinstance(reduced_bn_layer, nn.BatchNorm2d)
        assert reduced_bn_layer.num_features == bn_layer.num_features // k

    def test_increase_bn_features_k(self) -> None:
        num_features = 10
        bn_layer = nn.BatchNorm2d(num_features).to(DEVICE)
        k = 0.5
        increased_bn_layer, index = increase_bn_features(bn_layer, k=k, device=DEVICE)

        assert increased_bn_layer.num_features == int(num_features * (1 / k))
        assert index.size(0) == num_features * (1 / k - 1)

    def test_increase_bn_features_num_channels(self) -> None:
        num_features = 10
        bn_layer = nn.BatchNorm2d(num_features).to(DEVICE)
        num_channels_to_add = 5
        increased_bn_layer, index = increase_bn_features(
            bn_layer, num_channels_to_add=num_channels_to_add, device=DEVICE
        )

        assert increased_bn_layer.num_features == num_features + num_channels_to_add
        assert index.size(0) == num_channels_to_add