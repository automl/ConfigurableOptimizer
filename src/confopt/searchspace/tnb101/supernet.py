from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace

from .core import TNB101MicroModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TransNASBench101SearchSpace(SearchSpace):
    def __init__(self, *args, **kwargs):  # type: ignore
        model = TNB101MicroModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return [self.model.arch_parameters]  # type: ignore


if __name__ == "__main__":
    searchspace = TransNASBench101SearchSpace()
    print(searchspace.arch_parameters)
