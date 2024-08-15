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

from .core import DARTSSearchModel
from .core.genotypes import DARTSGenotype
from .core.model_search import (
    apply_operator_early_stopping,
    preserve_grads,
    update_grads_cosine_similarity,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DARTSSearchSpace(
    SearchSpace,
    ArchAttentionSupport,
    GradientMatchingScoreSupport,
    OperationStatisticsSupport,
    LayerAlignmentScoreSupport,
):
    def __init__(self, *args, **kwargs):  # type: ignore
        """DARTS Search Space for Neural Architecture Search.

        This class represents a search space for neural architecture search using
        DARTS (Differentiable Architecture Search).

        Args:
            *args: Variable length positional arguments. These arguments will be
                passed to the constructor of the internal DARTSSearchModel.
            **kwargs: Variable length keyword arguments. These arguments will be
                passed to the constructor of the internal DARTSSearchModel.

        Keyword Args:
            C (int): Number of channels.
            num_classes (int): Number of output classes.
            layers (int): Number of layers in the network.
            criterion (nn.modules.loss._Loss): Loss function.
            steps (int): Number of steps in the search space cell.
            multiplier (int): Multiplier for channels in the cells.
            stem_multiplier (int): Stem multiplier for channels.
            edge_normalization (bool): Whether to use edge normalization.

        Methods:
            - arch_parameters: Get architectural parameters.
            - beta_parameters: Get beta parameters.
            - set_arch_parameters(arch_parameters): Set architectural parameters

        Example:
            You can create an instance of DARTSSearchSpace with optional arguments as:
            >>> search_space = DARTSSearchSpace(
                                    C=32,
                                    num_classes=20,
                                    layers=10,
                                    criterion=nn.CrossEntropyLoss(),
                                    steps=5,
                                    multiplier=3,
                                    stem_multiplier=2,
                                    edge_normalization=True,
                                    dropout=0.2)
        """
        model = DARTSSearchModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the alpha parameters of the model
        Return:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.
        """
        return self.model.arch_parameters()  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the beta parameters of the model.

        Returns:
            list[nn.Parameter]: A list containing the beta parameters for the model.
        """
        return self.model.beta_parameters()

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.

        Note:
            This method sets the architectural parameters of the model to the provided
            values.
        """
        assert len(arch_parameters) == len(self.arch_parameters)
        assert arch_parameters[0].shape == self.arch_parameters[0].shape
        (
            self.model.alphas_normal.data,
            self.model.alphas_reduce.data,
        ) = arch_parameters
        self.model._arch_parameters = [
            self.model.alphas_normal,
            self.model.alphas_reduce,
        ]

    def discretize(self) -> nn.Module:
        return self.model.discretize()  # type: ignore

    def get_genotype(self) -> DARTSGenotype:
        return self.model.genotype()  # type: ignore

    def preserve_grads(self) -> None:
        self.model.apply(preserve_grads)

    def update_grads_cosine_similarity(self) -> None:
        self.model.apply(update_grads_cosine_similarity)

    def apply_operator_early_stopping(self) -> None:
        self.model.apply(apply_operator_early_stopping)

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
        return self.model._get_mean_layer_alignment_score()

    def get_num_skip_ops(self) -> dict[str, int]:
        alphas_normal, alphas_reduce = self.model.arch_parameters()
        count_skip = lambda alphas: sum(alphas[:, 1:].argmax(dim=1) == 2)

        stats = {
            "skip_connections/normal": count_skip(alphas_normal),
            "skip_connections/reduce": count_skip(alphas_reduce),
        }

        return stats
