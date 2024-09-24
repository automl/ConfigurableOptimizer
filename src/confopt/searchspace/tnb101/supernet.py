from __future__ import annotations

from functools import partial
from typing import Literal

import torch
from torch import nn

from confopt.searchspace.common.base_search import (
    ArchAttentionSupport,
    DrNASRegTermSupport,
    FairDARTSRegTermSupport,
    FLOPSRegTermSupport,
    GradientMatchingScoreSupport,
    GradientStatsSupport,
    LayerAlignmentScoreSupport,
    OperationStatisticsSupport,
    PerturbationArchSelectionSupport,
    SearchSpace,
)
from confopt.utils import update_gradient_matching_scores

from .core import TNB101MicroModel
from .core.model_search import preserve_grads
from .core.operations import OLES_OPS
from .core.operations import TRANS_NAS_BENCH_101 as PRIMITIVES

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TransNASBench101SearchSpace(
    SearchSpace,
    GradientMatchingScoreSupport,
    ArchAttentionSupport,
    GradientStatsSupport,
    FLOPSRegTermSupport,
    OperationStatisticsSupport,
    LayerAlignmentScoreSupport,
    FairDARTSRegTermSupport,
    DrNASRegTermSupport,
    PerturbationArchSelectionSupport,
):
    def __init__(self, *args, **kwargs):  # type: ignore
        model = TNB101MicroModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return [self.model.arch_parameters()]  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        return [self.model.beta_parameters()]

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        self.model._arch_parameters.data = arch_parameters[0]

    def discretize(self) -> nn.Module:
        return self.model._discretize()  # type: ignore

    def get_genotype(self) -> str:
        return self.model.genotype()

    def get_weighted_flops(self) -> torch.Tensor:
        return self.model.get_weighted_flops()

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

    def get_num_skip_ops(self) -> dict[str, int]:
        alphas_normal = self.arch_parameters[0]
        count_skip = lambda alphas: sum(
            alphas.argmax(dim=-1) == PRIMITIVES.index("skip_connect")
        )

        stats = {
            "skip_connections/normal": count_skip(alphas_normal),
        }

        return stats

    ### LayerAlignmentScoreSupport methods ###
    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        return self.model.get_mean_layer_alignment_score(), 0

    def get_first_and_last_layer_alignment_score(self) -> tuple[float, float]:
        return self.model.get_mean_layer_alignment_score(only_first_and_last=True), 0

    ### End of LayerAlignmentScoreSupport methods ###

    def get_fair_darts_arch_parameters(self) -> list[torch.Tensor]:
        return self.get_sampled_weights()

    def get_num_ops(self) -> int:
        return self.model.num_ops

    def get_num_edges(self) -> int:
        return self.model.num_edge

    def get_num_nodes(self) -> int:
        return self.model.num_nodes

    def is_topology_supported(self) -> bool:
        return False

    def set_topology(self, value: bool) -> None:
        self.topology = value

    def get_candidate_flags(self, cell_type: Literal["normal", "reduce"]) -> list:
        assert cell_type == "normal"
        return self.model.candidate_flags

    def remove_from_projected_weights(
        self,
        selected_edge: int,
        selected_op: int | None,
        cell_type: Literal["normal", "reduce"] = "normal",
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

    def get_max_input_edges_at_node(self, selected_node: int) -> int:  # noqa: ARG002
        return 1

    def get_drnas_anchors(self) -> list[torch.Tensor]:
        return [self.model.anchor_normal]

    def get_projected_arch_parameters(self) -> list[torch.Tensor]:
        return [self.model.get_projected_weights()]


if __name__ == "__main__":
    searchspace = TransNASBench101SearchSpace()
    print(searchspace.arch_parameters)
