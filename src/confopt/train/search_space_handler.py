from __future__ import annotations

import torch

from confopt.oneshot.archsampler import (
    BaseSampler,
    CompositeSampler,
    DARTSSampler,
    GDASSampler,
)
from confopt.oneshot.dropout import Dropout
from confopt.oneshot.dynamic_exploration import DynamicAttentionExplorer
from confopt.oneshot.lora_toggler import LoRAToggler
from confopt.oneshot.partial_connector import PartialConnector
from confopt.oneshot.perturbator import BasePerturbator
from confopt.oneshot.pruner import Pruner
from confopt.oneshot.regularizer import Regularizer
from confopt.oneshot.weightentangler import WeightEntangler
from confopt.searchspace import DARTSSearchSpace
from confopt.searchspace.common import (
    LambdaReg,
    LoRALayer,
    OperationBlock,
    OperationChoices,
    SearchSpace,
)
from confopt.searchspace.common.base_search import (
    ArchAttentionSupport,
    LambdaDARTSSupport,
)


class SearchSpaceHandler:
    def __init__(
        self,
        sampler: BaseSampler | CompositeSampler,
        edge_normalization: bool = False,
        partial_connector: PartialConnector | None = None,
        perturbation: BasePerturbator | None = None,
        dropout: Dropout | None = None,
        weight_entangler: WeightEntangler | None = None,
        lora_configs: dict | None = None,
        pruner: Pruner | None = None,
        lora_toggler: LoRAToggler | None = None,
        is_arch_attention_enabled: bool = False,
        regularizer: Regularizer | None = None,
        lambda_regularizer: LambdaReg | None = None,
        use_auxiliary_skip_connection: bool = False,
        dynamic_explorer: DynamicAttentionExplorer | None = None,
    ) -> None:
        self.sampler = sampler
        self.edge_normalization = edge_normalization
        self.partial_connector = partial_connector
        self.perturbation = perturbation
        self.dropout = dropout
        self.weight_entangler = weight_entangler
        self.lora_configs = lora_configs
        self.pruner = pruner
        self.lora_toggler = lora_toggler
        self.regularizer = regularizer
        self.lambda_regularizer = lambda_regularizer

        self.is_argmax_sampler = False
        if isinstance(self.sampler, GDASSampler):
            self.is_argmax_sampler = True

        self.is_arch_attention_enabled = is_arch_attention_enabled
        self.use_auxiliary_skip_connection = use_auxiliary_skip_connection

        self.dynamic_explorer = dynamic_explorer

    def adapt_search_space(self, search_space: SearchSpace) -> None:  # noqa: C901
        if hasattr(search_space.model, "edge_normalization"):
            search_space.model.edge_normalization = self.edge_normalization

        for name, module in search_space.named_modules(remove_duplicate=False):
            if isinstance(module, OperationChoices):
                new_module = self._initialize_operation_block(
                    module.ops,
                    module.is_reduction_cell,
                    module.aux_skip,
                    module.dan,
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

        if self.pruner:
            search_space.components.append(self.pruner)

        if self.lora_toggler:
            search_space.components.append(self.lora_toggler)

        if self.is_arch_attention_enabled and isinstance(
            search_space, ArchAttentionSupport
        ):
            search_space.set_arch_attention(True)

        if isinstance(search_space, LambdaDARTSSupport) and isinstance(
            self.lambda_regularizer, LambdaReg
        ):
            search_space.set_lambda_darts_params(self.lambda_regularizer)
            search_space.enable_lambda_darts()

        if self.dynamic_explorer:
            search_space.components.append(self.dynamic_explorer)

    def perturb_parameter(self, search_space: SearchSpace) -> None:
        if self.perturbation is not None:
            self.perturbation._perturb_and_update_alphas()
            search_space.set_arch_parameters(self.perturbation.perturbed_alphas)

    def update_sample_function_from_sampler(self, search_space: SearchSpace) -> None:
        search_space.set_sample_function(self.sampler.sample)

    def default_sample_function(self, alphas: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(alphas, dim=-1)

    def reset_sample_function(self, search_space: SearchSpace) -> None:
        search_space.set_sample_function(self.default_sample_function)

    def _initialize_operation_block(
        self,
        ops: torch.nn.Module,
        is_reduction_cell: bool = False,
        aux_skip: torch.nn.Module | None = None,
        dan: torch.nn.Module | None = None,
    ) -> OperationBlock:
        op_block = OperationBlock(
            ops,
            is_reduction_cell=is_reduction_cell,
            partial_connector=self.partial_connector,
            dropout=self.dropout,
            weight_entangler=self.weight_entangler,
            is_argmax_sampler=self.is_argmax_sampler,
            aux_skip=aux_skip if self.use_auxiliary_skip_connection else None,
            dan=dan if self.dynamic_explorer else None,
        )
        return op_block

    def activate_lora(
        self,
        searchspace: SearchSpace,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0,
        merge_weights: bool = True,
    ) -> None:
        if r > 0:
            for _, module in searchspace.named_modules(remove_duplicate=False):
                if isinstance(module, LoRALayer):
                    module.activate_lora(
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout_rate=lora_dropout,
                        merge_weights=merge_weights,
                    )

    def deactivate_lora(
        self,
        searchspace: SearchSpace,
    ) -> None:
        for _, module in searchspace.named_modules(remove_duplicate=False):
            if isinstance(module, LoRALayer):
                module.deactivate_lora()

    def toggle_lora(
        self,
        searchspace: SearchSpace,
    ) -> None:
        for _, module in searchspace.named_modules(remove_duplicate=True):
            if isinstance(module, LoRALayer):
                module.toggle_lora()

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

    def add_reg_terms(
        self, search_space: SearchSpace, loss: torch.Tensor
    ) -> torch.Tensor:
        if self.regularizer:
            return self.regularizer.add_reg_terms(search_space, loss)
        return loss


if __name__ == "__main__":
    search_space = DARTSSearchSpace()
    sampler = DARTSSampler(search_space.arch_parameters)
    search_space_handler = SearchSpaceHandler(sampler=sampler)
    search_space_handler.adapt_search_space(search_space=search_space)
    print("success")
