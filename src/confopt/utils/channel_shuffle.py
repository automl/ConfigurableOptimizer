from __future__ import annotations

import torch


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x
