from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace

from .core import NB201SearchModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NASBench201SearchSpace(SearchSpace):
    def __init__(self, *args, **kwargs):  # type: ignore
        model = NB201SearchModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return [self.model.arch_parameters]  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        return [self.model.beta_parameters]

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        self.model.arch_parameters.data = arch_parameters[0]
