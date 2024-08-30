from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

from confopt.oneshot.archsampler import BaseSampler


class SNASSampler(BaseSampler):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"] = "step",
        temp_init: float = 1.0,
        temp_min: float = 0.33,
        temp_annealing: bool = True,
        total_epochs: int = 250,
    ) -> None:
        super().__init__(arch_parameters, sample_frequency)

        assert temp_init >= temp_min
        assert temp_init <= 1.0
        assert temp_min > 0

        self.temp_init = temp_init
        self.temp_annealing = temp_annealing
        self.temp_min = temp_min
        self.total_epochs = total_epochs

        self.curr_temp = (1 - self._epoch / self.total_epochs) * (
            self.temp_init - self.temp_min
        ) + self.temp_min

    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        temperature = self.curr_temp
        weights = F.gumbel_softmax(alpha, temperature)
        return weights

    def new_epoch(self) -> None:
        if self.temp_annealing is True:
            self.curr_temp = (1 - self._epoch / self.total_epochs) * (
                self.temp_init - self.temp_min
            ) + self.temp_min

        return super().new_epoch()
