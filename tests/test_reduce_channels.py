import unittest

import torch
from torch import nn
from torch import optim

from confopt.utils.reduce_channels import (
    increase_bn_features,
    increase_conv_channels,
    reduce_bn_features,
    reduce_conv_channels,
    increase_bn_features,
    increase_conv_channels,
    change_channel_size_conv,
    change_features_bn,
    configure_optimizer,
    configure_scheduler
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

        increased_conv2d, _ = increase_conv_channels(
            original_conv2d, k=0.5, device=DEVICE
        )

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

        reduced_batchnorm, _ = increase_bn_features(
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

    def test_change_conv_channels(self) -> None:
        in_channels = 6
        out_channels = 12
        conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        ).to(DEVICE)

        k = 2.0
        reduced, _ = change_channel_size_conv(conv2d, k, device=DEVICE)
        assert reduced.in_channels < in_channels
        assert reduced.out_channels < out_channels

        k = 0.5
        increased, _ = change_channel_size_conv(conv2d, k, device=DEVICE)
        assert increased.in_channels > in_channels
        assert increased.out_channels > out_channels

    def test_change_bn_channels(self) -> None:
        num_features = 6
        bn = nn.BatchNorm2d(num_features=num_features).to(DEVICE)

        k = 2.0
        reduced, _ = change_features_bn(bn, k, device=DEVICE)
        assert reduced.num_features < num_features

        k = 0.5
        increased, _ = change_features_bn(bn, k, device=DEVICE)
        assert increased.num_features > num_features

    def test_configure_optimizer(self) -> None:
        lr = 1e-3
        out_channels = 64
        net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Flatten(),
        ).to(DEVICE)
        x = torch.randn(1, 3, 64, 64).to(DEVICE)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        pred = net(x)
        target = torch.ones_like(pred)
        loss = criterion(pred, target)
        with torch.no_grad():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for layer in net:
            if isinstance(layer, nn.Conv2d):
                change_channel_size_conv(layer, k=0.5, device=DEVICE)
            elif isinstance(layer, nn.BatchNorm2d):
                change_features_bn(layer, k=0.5, device=DEVICE)
        
        new_optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        configured_optim = configure_optimizer(optimizer, new_optimizer)

        new_state_dict = configured_optim.state_dict()["state"]
        old_state_dict = optimizer.state_dict()["state"]
        assert new_state_dict[1] == old_state_dict[1]
        assert new_state_dict[2] == old_state_dict[2]
        assert new_state_dict[3] == old_state_dict[3]
        assert not hasattr(new_state_dict[0], "raw_id")

    def test_configure_scheduler(self) -> None:
        lr = 1e-3
        out_channels = 64
        net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Flatten(),
        ).to(DEVICE)
        x = torch.randn(1, 3, 64, 64).to(DEVICE)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        criterion = nn.CrossEntropyLoss()

        pred = net(x)
        target = torch.ones_like(pred)
        loss = criterion(pred, target)
        with torch.no_grad():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        for layer in net:
            if isinstance(layer, nn.Conv2d):
                change_channel_size_conv(layer, k=0.5, device=DEVICE)
            elif isinstance(layer, nn.BatchNorm2d):
                change_features_bn(layer, k=0.5, device=DEVICE)

        new_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        configured_scheduler = configure_scheduler(scheduler, new_scheduler)

        assert configured_scheduler.state_dict()["last_epoch"] == 1

if __name__ == "__main__":
    unittest.main()
