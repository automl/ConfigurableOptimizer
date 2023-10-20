from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace

from .core import TNB101MicroModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TransNASBench101SearchSpace(SearchSpace):
    def __init__(self, *args, **kwargs):  # type: ignore
        """Initialize the custom search model of TransNASBench101SearchSpace.

        Args:
            *args: Positional arguments to pass to the TNB101MicroModel constructor.
            **kwargs: Keyword arguments to pass to the TNB101MicroModel constructor.

        Note:
            This constructor initializes the custom search model by creating an instance
            of TNB101MicroModel with the provided arguments and keyword arguments.
            The resulting model is then moved to the specified device (DEVICE).
        """
        model = TNB101MicroModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.
        """
        return [self.model.arch_parameters()]  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the beta parameters of the model.

        Returns:
            list[nn.Parameter]: A list containing the beta parameters for the model.
        """
        return [self.model.beta_parameters()]

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.
        """
        self.model._arch_parameters.data = arch_parameters[0]

    def discretize(self) -> None:
        sparsity = 0.25
        self.model._discretize(sparsity)


if __name__ == "__main__":
    searchspace = TransNASBench101SearchSpace()
    print(searchspace.arch_parameters)
