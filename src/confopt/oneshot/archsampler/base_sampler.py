from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import torch

from confopt.oneshot.base_component import OneShotComponent


class BaseSampler(OneShotComponent):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"],
    ) -> None:
        super().__init__()
        self.arch_parameters: list[torch.Tensor] = arch_parameters
        self.sampled_arch_parameters: list[torch.Tensor] = arch_parameters

        assert sample_frequency in [
            "epoch",
            "step",
        ], "sample_frequency must be either 'epoch' or 'step'"
        self.sample_frequency = sample_frequency

    @abstractmethod
    def sample_alphas(
        self, arch_parameters: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        pass

    def _sample_and_update_alphas(self) -> None:  # type: ignore
        sampled_alphas = self.sample_alphas(self.arch_parameters)
        # print(sampled_alphas)
        if sampled_alphas is not None:
            self.sampled_arch_parameters = sampled_alphas

    def new_epoch(self) -> None:
        super().new_epoch()
        if self.sample_frequency == "epoch":
            self._sample_and_update_alphas()

    def new_step(self) -> None:  # type: ignore
        super().new_step()

        if self.sample_frequency == "step":
            self._sample_and_update_alphas()
