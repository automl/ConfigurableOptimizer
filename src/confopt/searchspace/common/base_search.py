from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn  # noqa: PLR0402

from confopt.oneshot.base_component import OneShotComponent
from confopt.utils import reset_gm_score_attributes


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


class SearchSpace(ModelWrapper, ABC):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.components: list[OneShotComponent] = []

    @property
    @abstractmethod
    def arch_parameters(self) -> list[nn.Parameter]:
        pass

    @property
    @abstractmethod
    def beta_parameters(self) -> list[nn.Parameter] | None:
        pass

    @abstractmethod
    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        pass

    def set_sample_function(self, sample_function: Callable) -> None:
        self.model.sample = sample_function

    def model_weight_parameters(self) -> list[nn.Parameter]:
        arch_param_ids = {id(p) for p in getattr(self, "arch_parameters", [])}
        beta_param_ids = {id(p) for p in getattr(self, "beta_parameters", [])}

        all_parameters = [
            p
            for p in self.model.parameters()
            if id(p) not in arch_param_ids and id(p) not in beta_param_ids
        ]

        return all_parameters

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)  # type: ignore

    def new_epoch(self) -> None:
        for component in self.components:
            component.new_epoch()

    def new_step(self) -> None:
        for component in self.components:
            component.new_step()

    def reset_gm_score_attributes(self) -> None:
        for module in self.modules():
            reset_gm_score_attributes(module)

    def get_num_skip_ops(self) -> tuple[int, int]:
        return -1, -1


class ArchAttentionHandler(ModelWrapper):
    def set_arch_attention(self, enabled: bool) -> None:
        self.model.is_arch_attention_enabled = enabled
