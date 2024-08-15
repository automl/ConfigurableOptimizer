from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import (
    ArchAttentionSupport,
    GradientMatchingScoreSupport,
    LayerAlignmentScoreSupport,
    OperationStatisticsSupport,
    SearchSpace,
)

from .core.genotypes import Structure as NB201Gynotype
from .core.model_search import (
    NB201SearchModel,
    preserve_grads,
    update_grads_cosine_similarity,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NASBench201SearchSpace(
    SearchSpace,
    ArchAttentionSupport,
    GradientMatchingScoreSupport,
    OperationStatisticsSupport,
    LayerAlignmentScoreSupport,
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

    def update_grads_cosine_similarity(self) -> None:
        self.model.apply(update_grads_cosine_similarity)

    def apply_operator_early_stopping(self) -> None:
        self.model.apply()

    def calc_avg_gm_score(self) -> float:
        sim_avg = []
        for module in self.model.modules():
            if hasattr(module, "running_sim"):
                sim_avg.append(module.running_sim.avg)
        if len(sim_avg) == 0:
            return 0
        avg_gm_score = sum(sim_avg) / len(sim_avg)
        return avg_gm_score

    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        return self.model._get_mean_layer_alignment_score(), 0

    def get_num_skip_ops(self) -> dict[str, int]:
        alphas_normal = self.model.arch_parameters
        count_skip = lambda alphas: sum(alphas.argmax(dim=-1) == 1)

        stats = {
            "skip_connections/normal": count_skip(alphas_normal),
        }

        return stats
