from __future__ import annotations

from typing import Literal

import torch

from confopt.oneshot.archsampler import BaseSampler


class DARTSSampler(BaseSampler):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"] = "step",
        arch_combine_fn: Literal["default", "sigmoid"] = "default",
    ) -> None:
        super().__init__(
            arch_parameters=arch_parameters,
            sample_frequency=sample_frequency,
            arch_combine_fn=arch_combine_fn,
        )

    def sample_alphas(
        self, arch_parameters: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        sampled_alphas = []
        for alpha in arch_parameters:
            if self.arch_combine_fn == "default":
                sampled_alpha = torch.nn.functional.softmax(alpha, dim=-1)
            elif self.arch_combine_fn == "sigmoid":
                sampled_alpha = torch.nn.functional.sigmoid(alpha)

            sampled_alphas.append(sampled_alpha)
        return sampled_alphas
