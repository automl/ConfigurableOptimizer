from __future__ import annotations

import torch


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, num_channels, height, width = x.size()

    # Calculate remainder and determine padding size
    remainder = num_channels % groups
    if remainder != 0:
        pad_size = groups - remainder
        pad_zeros = torch.zeros(batch_size, pad_size, height, width, device=x.device)
        x = torch.cat([x, pad_zeros], dim=1)

        # generate mask
        original_mask = torch.ones(num_channels, dtype=torch.bool)
        zero_mask = torch.zeros(pad_size, dtype=torch.bool)
        mask = torch.cat([original_mask, zero_mask], dim=0)

        num_channels = num_channels + pad_size

    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    if remainder != 0:
        mask = mask.view(groups, channels_per_group)
        mask = mask.transpose(0, 1).contiguous()
        mask = mask.view(-1)

        x = x[:, mask, :, :]

    return x
