from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

from confopt.oneshot.archsampler import BaseSampler


class DRNASSampler(BaseSampler):
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

    def sample_alphas(self, arch_parameters: torch.Tensor) -> list[torch.Tensor]:
        sampled_alphas = []
        for alpha in arch_parameters:
            sampled_alphas.append(self.sample(alpha))
        return sampled_alphas

    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        beta = F.elu(alpha) + 1
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()

        if self.post_sample_fn == "sigmoid":
            weights = torch.nn.functional.sigmoid(weights)

        return weights  # type: ignore
