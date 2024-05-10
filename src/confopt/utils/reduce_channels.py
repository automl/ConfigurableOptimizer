from __future__ import annotations

from typing import TypeAlias

import torch
from torch import nn, optim

from confopt.searchspace.common import Conv2DLoRA

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LRSchedulerType: TypeAlias = torch.optim.lr_scheduler.LRScheduler


def change_channel_size_conv(
    conv2d_layer: nn.Conv2d | Conv2DLoRA,
    k: float,
    index: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> tuple[nn.Conv2d | Conv2DLoRA, torch.Tensor | None]:
    if k < 1:
        return increase_conv_channels(conv2d_layer, k, device)
    return reduce_conv_channels(conv2d_layer, k, device), index


def increase_conv_channels(
    conv2d_layer: nn.Conv2d | Conv2DLoRA,
    k: float,
    device: torch.device = DEVICE,
) -> tuple[nn.Conv2d | Conv2DLoRA, torch.Tensor]:
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
        increased_conv2d, _ = in_channel_wider(conv2d_layer, new_in_channels)
        increased_conv2d, out_index = out_channel_wider(
            increased_conv2d, new_out_channels
        )

    return increased_conv2d.to(device=device), out_index


def in_channel_wider(
    module: nn.Conv2d, new_channels: int, index: torch.Tensor | None = None
) -> tuple[nn.Conv2d, torch.Tensor]:
    weight = module.weight
    in_channels = weight.size(1)

    if index is None:
        index = torch.randint(
            low=0, high=in_channels, size=(new_channels - in_channels,)
        )
    module.weight = nn.Parameter(
        torch.cat([weight, weight[:, index, :, :].clone()], dim=1), requires_grad=True
    )

    module.in_channels = new_channels
    module.weight.in_index = index
    module.weight.t = "conv"
    if hasattr(weight, "out_index"):
        module.weight.out_index = weight.out_index
    module.weight.raw_id = weight.raw_id if hasattr(weight, "raw_id") else id(weight)
    return module, index


# bias = 0
def out_channel_wider(
    module: nn.Conv2d, new_channels: int, index: torch.Tensor | None = None
) -> tuple[nn.Conv2d, torch.Tensor]:
    weight = module.weight
    out_channels = weight.size(0)

    if index is None:
        index = torch.randint(
            low=0, high=out_channels, size=(new_channels - out_channels,)
        )
    module.weight = nn.Parameter(
        torch.cat([weight, weight[index, :, :, :].clone()], dim=0), requires_grad=True
    )

    module.out_channels = new_channels
    module.weight.out_index = index
    module.weight.t = "conv"
    if hasattr(weight, "in_index"):
        module.weight.in_index = weight.in_index
    module.weight.raw_id = weight.raw_id if hasattr(weight, "raw_id") else id(weight)
    return module, index


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
    batchnorm_layer: nn.BatchNorm2d,
    k: float,
    index: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> tuple[nn.BatchNorm2d, torch.Tensor | None]:
    if k < 1:
        return increase_bn_features(batchnorm_layer, k, index, device)
    return reduce_bn_features(batchnorm_layer, k, device), index


def increase_bn_features(
    batchnorm_layer: nn.BatchNorm2d,
    k: float,
    index: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> tuple[nn.BatchNorm2d, torch.Tensor]:
    if not isinstance(batchnorm_layer, nn.BatchNorm2d):
        raise TypeError("Input must be a nn.BatchNorm2d layer.")

    # Get the number of features in the original BatchNorm2d
    num_features = batchnorm_layer.num_features

    # Calculate the new number of features
    new_num_features = int(max(1, num_features // k))

    # Create a new BatchNorm2d layer with the increased number of features
    increased_batchnorm, index = bn_wider(batchnorm_layer, new_num_features, index)

    return increased_batchnorm.to(device), index


def bn_wider(
    module: nn.BatchNorm2d, new_features: int, index: torch.Tensor | None = None
) -> tuple[nn.BatchNorm2d, torch.Tensor]:
    running_mean = module.running_mean
    running_var = module.running_var
    if module.affine:
        weight = module.weight
        bias = module.bias
    num_features = module.num_features

    if index is None:
        index = torch.randint(
            low=0, high=num_features, size=(new_features - num_features,)
        )
    module.running_mean = torch.cat([running_mean, running_mean[index].clone()])
    module.running_var = torch.cat([running_var, running_var[index].clone()])
    if module.affine:
        module.weight = nn.Parameter(
            torch.cat([weight, weight[index].clone()], dim=0), requires_grad=True
        )
        module.bias = nn.Parameter(
            torch.cat([bias, bias[index].clone()], dim=0), requires_grad=True
        )

        module.weight.out_index = index
        module.bias.out_index = index
        module.weight.t = "bn"
        module.bias.t = "bn"
        if hasattr(module.weight, "raw_id"):
            module.weight.raw_id = (
                weight.raw_id if hasattr(weight, "raw_id") else id(weight)
            )
        if hasattr(module.bias, "raw_id"):
            module.bias.raw_id = bias.raw_id if hasattr(bias, "raw_id") else id(bias)
    module.num_features = new_features
    return module, index


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


def configure_optimizer(
    optimizer_old: optim.Optimizer, optimizer_new: optim.Optimizer
) -> optim.Optimizer:
    for i, p in enumerate(optimizer_new.param_groups[0]["params"]):
        if not hasattr(p, "raw_id"):
            optimizer_new.state[i] = optimizer_old.state_dict()["state"][i]
            continue
        state_old = optimizer_old.state_dict()["state"][i]
        state_new = optimizer_new.state[i]

        if state_old.get("momentum_buffer", None) is not None:
            state_new["momentum_buffer"] = state_old["momentum_buffer"]
            if p.t == "bn":
                # BN layer
                state_new["momentum_buffer"] = torch.cat(
                    [
                        state_new["momentum_buffer"],
                        state_new["momentum_buffer"][p.out_index].clone(),
                    ],
                    dim=0,
                )

            elif p.t == "conv":
                # conv layer
                if hasattr(p, "in_index"):
                    state_new["momentum_buffer"] = torch.cat(
                        [
                            state_new["momentum_buffer"],
                            state_new["momentum_buffer"][:, p.in_index, :, :].clone(),
                        ],
                        dim=1,
                    )
                if hasattr(p, "out_index"):
                    state_new["momentum_buffer"] = torch.cat(
                        [
                            state_new["momentum_buffer"],
                            state_new["momentum_buffer"][p.out_index, :, :, :].clone(),
                        ],
                        dim=0,
                    )
            # clean to enable multiple call
            del p.t, p.raw_id
            if hasattr(p, "in_index"):
                del p.in_index
            if hasattr(p, "out_index"):
                del p.out_index
    return optimizer_new


def configure_scheduler(
    scheduler_old: LRSchedulerType,
    scheduler_new: LRSchedulerType,
) -> LRSchedulerType:
    scheduler_new.load_state_dict(scheduler_old.state_dict())
    return scheduler_new
