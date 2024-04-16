from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace
from confopt.searchspace.darts.core.genotypes import DARTSGenotype

from .core import RobustDARTSSearchModel
from .core.spaces import spaces_dict

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RobustDARTSSearchSpace(SearchSpace):
    def __init__(self, space: str = "s1", *args, **kwargs):  # type: ignore
        """Robust-DARTS Search Space for Neural Architecture Search.

        This class represents a search space for neural architecture search using
        DARTS (Differentiable Architecture Search).

        Args:
            space: The space to use (s1, s2, s3, s4)
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
        try:
            primitives = spaces_dict[space]
        except KeyError as err:
            raise ValueError(
                f"Invalid space {space}. Available spaces are {spaces_dict.keys()}"
            ) from err

        model = RobustDARTSSearchModel(primitives, *args, **kwargs).to(DEVICE)
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

    def prune(self, wider: int | None = None) -> None:
        sparsity = 0.125
        self.model._prune(sparsity, wider)  # type: ignore

    def discretize(self) -> nn.Module:
        return self.model._discretize()  # type: ignore

    def get_genotype(self) -> DARTSGenotype:
        return self.model.genotype()  # type: ignore
