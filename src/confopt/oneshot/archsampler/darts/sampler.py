from __future__ import annotations

from typing import Literal

import torch

from confopt.oneshot.archsampler import BaseSampler


class DARTSSampler(BaseSampler):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"] = "step",
        post_sample_fn: Literal["default", "sigmoid"] = "default",
    ) -> None:
        super().__init__(
            arch_parameters=arch_parameters,
            sample_frequency=sample_frequency,
            post_sample_fn=post_sample_fn,
        )

    def sample_alphas(
        self, arch_parameters: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        sampled_alphas = []
        for alpha in arch_parameters:
            if self.post_sample_fn == "default":
                sampled_alpha = torch.nn.functional.softmax(alpha, dim=-1)
            elif self.post_sample_fn == "sigmoid":
                sampled_alpha = torch.nn.functional.sigmoid(alpha)

            sampled_alphas.append(sampled_alpha)
        return sampled_alphas
