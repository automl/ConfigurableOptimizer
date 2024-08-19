from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn  # noqa: PLR0402

from confopt.oneshot.base_component import OneShotComponent
from confopt.utils import reset_gm_score_attributes


class ModelWrapper(nn.Module, ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


class SearchSpace(ModelWrapper):
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

    def prune(self, num_keep: int) -> None:
        """Prune the candidates operations of the supernet."""
        self.model.prune(num_keep=num_keep)  # type: ignore

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)  # type: ignore

    def new_epoch(self) -> None:
        for component in self.components:
            component.new_epoch()

    def new_step(self) -> None:
        for component in self.components:
            component.new_step()


class ArchAttentionSupport(ModelWrapper):
    def set_arch_attention(self, enabled: bool) -> None:
        """Enable or disable attention between architecture parameters."""
        self.model.is_arch_attention_enabled = enabled


class GradientMatchingScoreSupport(ModelWrapper):
    @abstractmethod
    def preserve_grads(self) -> None:
        """Preserve the gradients of the model for gradient matching later."""
        ...

    @abstractmethod
    def update_gradient_matching_scores(
        self,
        early_stop: bool = False,
        early_stop_frequency: int = 20,
        early_stop_threshold: float = 0.4,
    ) -> None:
        """Update the gradient matching scores of the model."""
        ...

    @abstractmethod
    def calc_avg_gm_score(self) -> float:
        """Calculate the average gradient matching score of the model.

        Returns:
            float: The average gradient matching score of the model.
        """
        ...

    def reset_gm_score_attributes(self) -> None:
        """Reset the gradient matching score attributes of the model."""
        for module in self.modules():
            reset_gm_score_attributes(module)

    def reset_gm_scores(self) -> None:
        """Reset the gradient matching scores of the model."""
        for module in self.model.modules():
            if hasattr(module, "running_sim"):
                module.running_sim.reset()


class LayerAlignmentScoreSupport(ModelWrapper):
    @abstractmethod
    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        """Get the mean layer alignment score of the model.

        Returns:
            tuple[float, float]: The mean layer alignment score of the normal
            and reduction cell.

        """


class OperationStatisticsSupport(ModelWrapper):
    @abstractmethod
    def get_num_skip_ops(self) -> dict[str, int]:
        """Get the number of skip operations in the model.

        Returns:
            dict[str, int]: A dictionary containing the number of skip operations
            in different types of cells. E.g., for DARTS, the dictionary would
            contain the keys "skip_connections/normal" and "skip_connections/reduce"
            with the number of skip operations.
            In NB201, the dictionary would contain only "skip_connections/normal".
        """

    def get_op_stats(self) -> dict[str, Any]:
        """Get the all the candidate operation statistics of the model."""
        skip_ops_stats = self.get_num_skip_ops()

        all_stats = {}
        all_stats.update(skip_ops_stats)
        # all_stats.update(other_stats) # Add other stats here

        return all_stats
