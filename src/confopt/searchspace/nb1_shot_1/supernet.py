from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from confopt.searchspace.common import SearchSpace
from confopt.searchspace.nb1_shot_1.core import (
    NB1Shot1Space1,
    NB1Shot1Space2,
    NB1Shot1Space3,
)
from confopt.searchspace.nb1_shot_1.core import (
    Network as NASBench1Shot1Network,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

search_space_map = {
    "S1": NB1Shot1Space1,
    "S2": NB1Shot1Space2,
    "S3": NB1Shot1Space3,
}


class NASBench1Shot1SearchSpace(SearchSpace):
    def __init__(
        self, search_space: Literal["S1", "S2", "S3"], *args: Any, **kwargs: dict
    ) -> None:
        self.search_space = search_space
        self.search_space_type = search_space_map[search_space]()

        model = NASBench1Shot1Network(
            *args,
            **kwargs,
            steps=self.search_space_type.num_intermediate_nodes,
            search_space=self.search_space_type,
        ).to(DEVICE)

        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return self.model.arch_parameters()

    @property
    def beta_parameters(self) -> list[nn.Parameter] | None:
        return self.model.beta_parameters()

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        assert len(arch_parameters) == len(self.arch_parameters)

        for old_params, new_params in zip(self.arch_parameters, arch_parameters):
            assert old_params.shape == new_params.shape, (
                f"New arch params have shape {new_params.shape}"
                + ". Expected {old_params.shape}."
            )
            old_params.data = new_params.data


if __name__ == "__main__":
    search_space = NASBench1Shot1SearchSpace("S1")
    print(search_space.arch_parameters)
    print(search_space.beta_parameters)

    x = torch.randn(1, 3, 32, 32).to(DEVICE)
    print(search_space(x))
