from __future__ import annotations

from functools import partial
from typing import Any, Literal
import warnings

import torch
from torch import nn

from confopt.searchspace.common import SearchSpace
from confopt.searchspace.common.base_search import (
    ArchAttentionSupport,
    DrNASRegTermSupport,
    FLOPSRegTermSupport,
    GradientMatchingScoreSupport,
    GradientStatsSupport,
    LayerAlignmentScoreSupport,
    PerturbationArchSelectionSupport,
)
from confopt.searchspace.nb1_shot_1.core import (
    NB1Shot1Space1,
    NB1Shot1Space2,
    NB1Shot1Space3,
)
from confopt.searchspace.nb1_shot_1.core import (
    Network as NASBench1Shot1Network,
)
from confopt.searchspace.nb1_shot_1.core.model_search import preserve_grads
from confopt.searchspace.nb1_shot_1.core.operations import OLES_OPS
from confopt.utils import update_gradient_matching_scores

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

search_space_map = {
    "S1": NB1Shot1Space1,
    "S2": NB1Shot1Space2,
    "S3": NB1Shot1Space3,
}


class NASBench1Shot1SearchSpace(
    SearchSpace,
    PerturbationArchSelectionSupport,
    LayerAlignmentScoreSupport,
    ArchAttentionSupport,
    GradientStatsSupport,
    DrNASRegTermSupport,
    FLOPSRegTermSupport,
    GradientMatchingScoreSupport,
):
    def __init__(self, search_space: Literal["S1", "S2", "S3"], **kwargs: dict) -> None:
        self.search_space_type = search_space_map[search_space]()  # type: ignore
        self.search_space = search_space

        if "steps" in kwargs:
            warnings.warn(
                "The steps arguments should not be provided explicitly in"
                "the initializer. Ignoring it.",
                stacklevel=1,
            )
            del kwargs["steps"]

        model = NASBench1Shot1Network(
            steps=self.search_space_type.num_intermediate_nodes,
            search_space=self.search_space_type,
            **kwargs,  # type: ignore
        ).to(DEVICE)

        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return self.model.arch_parameters()

    @property
    def beta_parameters(self) -> list[nn.Parameter] | None:
        return self.model.beta_parameters()

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        assert len(arch_parameters) == len(self.arch_parameters)

        for old_params, new_params in zip(self.arch_parameters, arch_parameters):
            assert old_params.shape == new_params.shape, (
                f"New arch params have shape {new_params.shape}"
                + ". Expected {old_params.shape}."
            )
            old_params.data = new_params.data

    def get_genotype(self) -> Any:
        return self.model.genotype()

    def get_weighted_flops(self) -> torch.Tensor:
        return self.model.get_weighted_flops()

    def get_num_ops(self) -> int:
        return self.model.num_ops

    def get_num_edges(self) -> int:
        return self.model.num_edges

    def get_num_nodes(self) -> int:
        return self.search_space_type.num_intermediate_nodes

    def get_candidate_flags(self, cell_type: Literal["normal", "reduce"]) -> list:
        assert cell_type == "normal"
        if self.topology:
            return self.model.candidate_flags_edge
        return self.model.candidate_flags

    def get_edges_at_node(self, selected_node: int) -> list:
        return self.model.nid2eids[selected_node]

    def remove_from_projected_weights(
        self,
        selected_edge: int,
        selected_op: int | None,
        cell_type: Literal["normal", "reduce"] = "normal",
    ) -> None:
        assert cell_type == "normal"
        if self.topology is None:
            self.set_topology(False)
        self.model.remove_from_projected_weights(
            selected_edge, selected_op, self.topology
        )

    def mark_projected_operation(
        self,
        selected_edge: int,
        selected_op: int,
        cell_type: Literal["normal"] | Literal["reduce"],
    ) -> None:
        assert cell_type == "normal"
        self.model.mark_projected_op(selected_edge, selected_op)

    def mark_projected_edge(
        self,
        selected_node: int,
        selected_edges: list[int],
        cell_type: str | None = None,
    ) -> None:
        assert cell_type == "normal"
        self.model.mark_projected_edges(selected_node, selected_edges)

    def set_projection_mode(self, value: bool) -> None:
        self.model.projection_mode = value

    def set_projection_evaluation(self, value: bool) -> None:
        self.model.projection_evaluation = value

    def is_topology_supported(self) -> bool:
        return True

    def get_projected_arch_parameters(self) -> list[torch.Tensor]:
        return list(self.model.get_projected_weights())

    def get_max_input_edges_at_node(self, selected_node: int) -> int:
        return self.model.get_max_input_edges_at_node(selected_node)

    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        return self.model.get_mean_layer_alignment_score(), 0

    def get_first_and_last_layer_alignment_score(self) -> tuple[float, float]:
        return self.model.get_mean_layer_alignment_score(only_first_and_last=True), 0

    def get_drnas_anchors(self) -> list[torch.Tensor]:
        return self.model.get_drnas_anchors()

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


if __name__ == "__main__":
    search_space = NASBench1Shot1SearchSpace("S1")
    print(search_space.arch_parameters)
    print(search_space.beta_parameters)

    x = torch.randn(1, 3, 32, 32).to(DEVICE)
    print(search_space(x))
