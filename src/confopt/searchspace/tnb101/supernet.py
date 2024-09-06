from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import (
    ArchAttentionSupport,
    FairDARTSRegTermSupport,
    GradientStatsSupport,
    SearchSpace,
)

from .core import TNB101MicroModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TransNASBench101SearchSpace(
    SearchSpace, ArchAttentionSupport, GradientStatsSupport, FairDARTSRegTermSupport
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

    def get_fair_darts_arch_parameters(self) -> list[torch.Tensor]:
        return self.get_sampled_weights()


if __name__ == "__main__":
    searchspace = TransNASBench101SearchSpace()
    print(searchspace.arch_parameters)
