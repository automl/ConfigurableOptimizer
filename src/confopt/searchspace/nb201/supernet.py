from __future__ import annotations

from functools import partial
from typing import Literal

import torch
from torch import nn

from confopt.searchspace.common import (
    ArchAttentionSupport,
    GradientMatchingScoreSupport,
    LayerAlignmentScoreSupport,
    OperationStatisticsSupport,
    PerturbationArchSelectionSupport,
    SearchSpace,
)
from confopt.searchspace.nb201.core.operations import OLES_OPS
from confopt.utils import update_gradient_matching_scores

from .core.genotypes import Structure as NB201Gynotype
from .core.model_search import (
    NB201SearchModel,
    preserve_grads,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NASBench201SearchSpace(
    SearchSpace,
    ArchAttentionSupport,
    GradientMatchingScoreSupport,
    OperationStatisticsSupport,
    LayerAlignmentScoreSupport,
    PerturbationArchSelectionSupport,
):
    def __init__(self, *args, **kwargs):  # type: ignore
        """Initialize the custom search model of NASBench201SearchSpace.

        Args:
            *args: Positional arguments to pass to the NB201SearchModel constructor.
            **kwargs: Keyword arguments to pass to the NB201SearchModel constructor.

        Note:
            This constructor initializes the custom search model by creating an instance
            of NB201SearchModel with the provided arguments and keyword arguments.
            The resulting model is then moved to the specified device (DEVICE).
        """
        model = NB201SearchModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.

        Note:
            This method sets the architectural parameters of the model to the provided
            values.
        """
        return [self.model.arch_parameters]  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the beta parameters of the model.

        Returns:
            list[nn.Parameter]: A list containing the beta parameters for the model.
        """
        return [self.model.beta_parameters]  # type: ignore

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.

        Note:
            This method sets the architectural parameters of the model to the provided
            values.
        """
        self.model.arch_parameters.data = arch_parameters[0]

    def discretize(self) -> nn.Module:
        return self.model._discretize()  # type: ignore

    def get_genotype(self) -> NB201Gynotype:
        return self.model.genotype()  # type: ignore

    def preserve_grads(self) -> None:
        self.model.apply(preserve_grads)

    def update_gradient_matching_scores(
        self,
        early_stop: bool = False,
        early_stop_frequency: int = 20,
        early_stop_threshold: float = 0.4,
    ) -> None:
        partial_fn = partial(
            update_gradient_matching_scores,
            oles_ops=OLES_OPS,
            early_stop=early_stop,
            early_stop_frequency=early_stop_frequency,
            early_stop_threshold=early_stop_threshold,
        )
        self.model.apply(partial_fn)

    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        return self.model._get_mean_layer_alignment_score(), 0

    def get_num_skip_ops(self) -> dict[str, int]:
        alphas_normal = self.model.arch_parameters
        count_skip = lambda alphas: sum(alphas.argmax(dim=-1) == 1)

        stats = {
            "skip_connections/normal": count_skip(alphas_normal),
        }

        return stats

    def get_num_ops(self) -> int:
        return self.model.num_ops

    def get_num_edges(self) -> int:
        return self.model.num_edges

    def get_num_nodes(self) -> int:
        return self.model.num_nodes

    def get_candidate_flags(self, cell_type: Literal["normal", "reduce"]) -> list:
        assert cell_type == "normal"
        return self.model.candidate_flags

    def remove_from_projected_weights(
        self,
        selected_edge: int,
        selected_op: int | None,
        cell_type: Literal["normal", "reduce"],
    ) -> None:
        assert cell_type == "normal"
        assert selected_op is not None
        self.model.remove_from_projected_weights(selected_edge, selected_op)

    def mark_projected_operation(
        self,
        selected_edge: int,
        selected_op: int,
        cell_type: Literal["normal", "reduce"],
    ) -> None:
        assert cell_type == "normal"
        self.model.mark_projected_op(selected_edge, selected_op)

    def set_projection_mode(self, value: bool) -> None:
        self.model.projection_mode = value

    def set_projection_evaluation(self, value: bool) -> None:
        self.model.projection_evaluation = value

    def is_topology_supported(self) -> bool:
        return False
