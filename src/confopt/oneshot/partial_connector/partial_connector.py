from __future__ import annotations

import torch
from torch import nn

from confopt.utils.channel_shuffle import channel_shuffle


class PartialConnector(nn.Module):
    def __init__(
        self,
        is_reduction_cell: bool = False,
        k: int = 4,
        num_warm_epoch: int = 15,
    ) -> None:
        super().__init__()
        self.k = k
        self.is_reduction_cell = is_reduction_cell
        self.mp = nn.MaxPool2d(2, 2)
        self.num_warm_epoch = num_warm_epoch

    def forward(
        self,
        x: torch.Tensor,
        alphas: list[torch.Tensor],
        ops: list[nn.Module],
        is_argmax_sampler: bool = False,
    ) -> torch.Tensor:
        assert len(alphas) == len(
            ops
        ), "Number of operations and architectural weights do not match"

        dim_2 = x.shape[1]  # channel number of the input to the layer
        xtemp = x[:, : dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k :, :, :]

        if is_argmax_sampler:
            argmax = torch.argmax(alphas)
            temp1 = [
                alphas[i] * op(xtemp) if i == argmax else alphas[i]
                for i, op in enumerate(ops)
            ]
            temp1 = sum(temp1)
        else:
            temp1 = sum(op(xtemp) * alpha for op, alpha in zip(ops, alphas))

        if (
            hasattr(ops[-1], "C_in")
            and hasattr(ops[-1], "C_out")
            and (ops[-1].C_in != ops[-1].C_out)
        ):
            xtemp2 = expand_tensor(
                xtemp2, ops[-1].C_out - temp1.size(1)  # type: ignore
            )
        # reduction cell needs pooling before concat
        # Not all reduction cell have stride = 2
        if temp1.shape[2] != xtemp2.shape[2]:  # type: ignore
            states = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        else:
            states = torch.cat([temp1, xtemp2], dim=1)

        states = channel_shuffle(states, self.k)
        return states


def expand_tensor(original_tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    # Calculate the size difference
    size_difference = new_size - original_tensor.size(1)
    if size_difference <= 0:
        # If the new size is not greater than the original size,
        # return the original tensor
        return original_tensor
    # Generate random indices for sampling from the original tensor
    random_indices = torch.randint(
        low=0, high=original_tensor.size(1), size=(size_difference,)
    )

    # Sample values from the original tensor
    random_tensor = original_tensor[:, random_indices, :, :].clone()
    # Concatenate the original tensor with the repeated random values
    new_tensor = torch.cat((original_tensor, random_tensor), dim=1)

    return new_tensor
