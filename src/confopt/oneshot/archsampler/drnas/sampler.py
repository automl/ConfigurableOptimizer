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
        arch_combine_fn: Literal["default", "sigmoid"] = "default",
    ) -> None:
        super().__init__(
            arch_parameters=arch_parameters,
            sample_frequency=sample_frequency,
            arch_combine_fn=arch_combine_fn,
        )

    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        weights_list = []
        for alpha_edge in alpha:
            beta = F.elu(alpha_edge) + 1
            weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()

            if self.arch_combine_fn == "sigmoid":
                weights = torch.nn.functional.sigmoid(weights)
            weights_list.append(weights)

        weights = torch.stack(weights_list)
        return weights
