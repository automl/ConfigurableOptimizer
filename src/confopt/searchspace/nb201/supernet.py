from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace

from .core import NB201SearchModel
from .core.genotypes import Structure

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NASBench201SearchSpace(SearchSpace):
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

    def prune(self, wider: int | None = None) -> None:
        """Discretize the model's architecture parameters to enforce sparsity.

        Note:
            This method discretizes the model's architecture parameters to enforce
            sparsity. It sets the sparsity level to 0.2 (20% of operations will be kept)
            and calls the `_discretize` method to apply the discretization.
        """
        # TODO: add a function that would return valid sparsity values based on search
        # space
        sparsity = 0.2
        self.model._prune(sparsity, wider)  # type: ignore

    def discretize(self) -> nn.Module:
        return self.model._discretize()  # type: ignore

    def get_genotype(self) -> Structure:
        return self.model.genotype()
