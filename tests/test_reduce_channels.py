import unittest

import torch
from torch import nn

from confopt.utils.reduce_channels import (
    increase_bn_features,
    increase_conv_channels,
    reduce_bn_features,
    reduce_conv_channels,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestReduceChannels(unittest.TestCase):
    def test_reduce_conv_channels(self) -> None:
        original_conv2d = nn.Conv2d(
            in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1
        ).to(DEVICE)
        original_conv2d.weight.data.fill_(0.5)

        reduced_conv2d = reduce_conv_channels(original_conv2d, k=2, device=DEVICE)

        assert reduced_conv2d.in_channels == 3
        assert reduced_conv2d.out_channels == 6

        assert torch.all(
            torch.eq(
                reduced_conv2d.weight[:6, :3, :, :],
                original_conv2d.weight[:6, :3, :, :],
            )
        )
        if original_conv2d.bias is not None:
            assert torch.all(
                torch.eq(reduced_conv2d.bias[:6], original_conv2d.bias[:6])
            )

    def test_reduce_features(self) -> None:
        original_batchnorm = nn.BatchNorm2d(num_features=12).to(DEVICE)
        original_batchnorm.weight.data.fill_(0.5)

        reduced_batchnorm = reduce_bn_features(original_batchnorm, k=3, device=DEVICE)

        assert reduced_batchnorm.num_features == 4

        if original_batchnorm.affine:
            assert torch.all(
                torch.eq(reduced_batchnorm.weight[:4], original_batchnorm.weight[:4])
            )
            assert torch.all(
                torch.eq(reduced_batchnorm.bias[:4], original_batchnorm.bias[:4])
            )

    def test_increase_conv_channels(self) -> None:
        original_conv2d = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        ).to(DEVICE)
        original_conv2d.weight.data.fill_(0.5)

        increased_conv2d = increase_conv_channels(original_conv2d, k=0.5, device=DEVICE)

        assert increased_conv2d.in_channels == 6
        assert increased_conv2d.out_channels == 12

        assert torch.all(
            torch.eq(
                original_conv2d.weight[:6, :3, :, :],
                increased_conv2d.weight[:6, :3, :, :],
            )
        )
        if original_conv2d.bias is not None:
            assert torch.all(
                torch.eq(original_conv2d.bias[:6], increased_conv2d.bias[:6])
            )

    def test_increase_features(self) -> None:
        original_batchnorm = nn.BatchNorm2d(num_features=6).to(DEVICE)
        original_batchnorm.weight.data.fill_(0.5)

        reduced_batchnorm = increase_bn_features(
            original_batchnorm, k=0.5, device=DEVICE
        )

        assert reduced_batchnorm.num_features == 12

        if original_batchnorm.affine:
            assert torch.all(
                torch.eq(reduced_batchnorm.weight[:6], original_batchnorm.weight[:6])
            )
            assert torch.all(
                torch.eq(reduced_batchnorm.bias[:6], original_batchnorm.bias[:6])
            )


if __name__ == "__main__":
    unittest.main()
