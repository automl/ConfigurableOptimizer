from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace

from .core import DARTSSearchModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DARTSSearchSpace(SearchSpace):
    def __init__(self, *args, **kwargs):  # type: ignore
        model = DARTSSearchModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return self.model.arch_parameters()  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        return self.model.beta_parameters()

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        assert len(arch_parameters) == len(self.arch_parameters)
        assert arch_parameters[0].shape == self.arch_parameters[0].shape
        self.model.alphas_normal.data, self.model.alphas_reduce.data = arch_parameters
        self.model._arch_parameters = [
            self.model.alphas_normal,
            self.model.alphas_reduce,
        ]
