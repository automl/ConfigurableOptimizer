from __future__ import annotations

import torch

from confopt.oneshot.archsampler import BaseSampler
from confopt.oneshot.base import OneShotComponent


class CompositeSampler(OneShotComponent):
    def __init__(
        self,
        arch_samplers: list[BaseSampler],
        arch_parameters: list[torch.Tensor],
    ) -> None:
        super().__init__()
        self.arch_samplers = arch_samplers
        self.arch_parameters = arch_parameters

        # get sample frequency from the samplers
        self.sample_frequency = arch_samplers[0].sample_frequency
        for sampler in arch_samplers:
            assert (
                self.sample_frequency == sampler.sample_frequency
            ), "All samplers must have the same sample frequency"

    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        sampled_alphas = alpha
        for sampler in self.arch_samplers:
            sampled_alphas = sampler.sample(sampled_alphas)

        return sampled_alphas

    def new_epoch(self) -> None:
        super().new_epoch()
        for sampler in self.arch_samplers:
            sampler.new_epoch()

    def new_step(self) -> None:
        super().new_step()
        for sampler in self.arch_samplers:
            sampler.new_step()
