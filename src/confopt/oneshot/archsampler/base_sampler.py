from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import torch

from confopt.oneshot.base import OneShotComponent


class BaseSampler(OneShotComponent):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"],
        arch_combine_fn: Literal["default", "sigmoid"] = "default",
    ) -> None:
        super().__init__()
        self.arch_parameters = arch_parameters
        self.sampled_alphas: list[torch.Tensor] | None = None

        assert sample_frequency in [
            "epoch",
            "step",
        ], "sample_frequency must be either 'epoch' or 'step'"
        self.sample_frequency = sample_frequency
        self.arch_combine_fn = arch_combine_fn

    @abstractmethod
    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        pass
