from __future__ import annotations

from torch.nn import DataParallel

from confopt.oneshot.base_component import OneShotComponent
from confopt.searchspace.common.base_search import SearchSpace
from confopt.searchspace.common.lora_layers import LoRALayer


class LoRAToggler(OneShotComponent):
    def __init__(self, searchspace: SearchSpace, toggle_epochs: list[int]) -> None:
        super().__init__()
        self.searchspace = searchspace
        self.toggle_epochs = toggle_epochs

    def new_epoch(self) -> None:
        if self._epoch in self.toggle_epochs:
            for _, module in self.searchspace.named_modules(remove_duplicate=True):
                if isinstance(module, LoRALayer):
                    module.toggle_lora()

            unwrapped_model = (
                self.searchspace.module
                if isinstance(self.searchspace, DataParallel)
                else self.searchspace
            )
            unwrapped_model.reset_gm_score_attributes()

        super().new_epoch()
