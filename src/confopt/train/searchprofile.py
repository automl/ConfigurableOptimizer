from __future__ import annotations

import torch

from confopt.oneshot.archsampler import BaseSampler, DARTSSampler, GDASSampler
from confopt.oneshot.dropout import Dropout
from confopt.oneshot.partial_connector import PartialConnector
from confopt.oneshot.perturbator import BasePerturbator
from confopt.searchspace import DARTSSearchSpace
from confopt.searchspace.common import (
    OperationBlock,
    OperationChoices,
    SearchSpace,
)


class Profile:
    def __init__(
        self,
        sampler: BaseSampler,
        edge_normalization: bool = False,
        partial_connector: PartialConnector | None = None,
        perturbation: BasePerturbator | None = None,
        dropout: Dropout | None = None,
    ) -> None:
        self.sampler = sampler
        self.edge_normalization = edge_normalization
        self.partial_connector = partial_connector
        self.perturbation = perturbation
        self.dropout = dropout

        self.is_argmax_sampler = False
        if isinstance(self.sampler, GDASSampler):
            self.is_argmax_sampler = True

    def adapt_search_space(self, search_space: SearchSpace) -> None:
        if hasattr(search_space.model, "edge_normalization"):
            search_space.model.edge_normalization = self.edge_normalization

        for name, module in search_space.named_modules(remove_duplicate=False):
            if isinstance(module, OperationChoices):
                new_module = self._initialize_operation_block(
                    module.ops, module.is_reduction_cell
                )
                parent_name, attribute_name = self.get_parent_and_attribute(name)
                setattr(
                    eval("search_space" + parent_name),
                    attribute_name,
                    new_module,
                )
        search_space.components.append(self.sampler)
        if self.perturbation:
            search_space.components.append(self.perturbation)

        if self.dropout:
            search_space.components.append(self.dropout)

    def perturb_parameter(self, search_space: SearchSpace) -> None:
        if self.perturbation is not None:
            self.perturbation._perturb_and_update_alphas()
            search_space.set_arch_parameters(self.perturbation.perturbed_alphas)

    def update_sample_function_from_sampler(self, search_space: SearchSpace) -> None:
        search_space.set_sample_function(self.sampler.sample_alphas)

    def default_sample_function(self, alphas: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(alphas, dim=-1)

    def reset_sample_function(self, search_space: SearchSpace) -> None:
        search_space.set_sample_function(self.default_sample_function)

    def _initialize_operation_block(
        self, ops: torch.nn.Module, is_reduction_cell: bool = False
    ) -> OperationBlock:
        op_block = OperationBlock(
            ops,
            is_reduction_cell,
            self.partial_connector,
            self.dropout,
            is_argmax_sampler=self.is_argmax_sampler,
        )
        return op_block

    def get_parent_and_attribute(self, module_name: str) -> tuple[str, str]:
        split_index = module_name.rfind(".")
        if split_index != -1:
            parent_name = module_name[:split_index]
            attribute_name = module_name[split_index + 1 :]
        else:
            parent_name = ""
            attribute_name = module_name
            return parent_name, attribute_name
        parent_name_list = parent_name.split(".")
        for idx, comp in enumerate(parent_name_list):
            try:
                if isinstance(eval(comp), int):
                    parent_name_list[idx] = "[" + comp + "]"
            except:  # noqa: E722, S112
                continue

        parent_name = ""
        for comp in parent_name_list:
            if "[" in comp:
                parent_name += comp
            else:
                parent_name += "." + comp
        return parent_name, attribute_name


if __name__ == "__main__":
    search_space = DARTSSearchSpace()
    sampler = DARTSSampler(search_space.arch_parameters)
    profile = Profile(sampler=sampler)
    profile.adapt_search_space(search_space=search_space)
    print("success")
