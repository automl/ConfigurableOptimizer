from __future__ import annotations

import numpy as np

from confopt.oneshot.base_component import OneShotComponent
from confopt.searchspace.common.base_search import SearchSpace
from confopt.searchspace.common.lora_layers import LoRALayer
from confopt.utils import reset_gm_score_attributes


class LoRAToggler(OneShotComponent):
    def __init__(
        self,
        searchspace: SearchSpace,
        toggle_epochs: list[int],
        toggle_probability: float | None = None,
    ) -> None:
        super().__init__()
        self.searchspace = searchspace
        self.toggle_epochs = toggle_epochs
        self.toggle_probability = toggle_probability

    def new_epoch(self) -> None:
        if self._epoch in self.toggle_epochs:
            for _, module in self.searchspace.named_modules(remove_duplicate=True):
                if isinstance(module, LoRALayer) and (
                    (self.toggle_probability is None)
                    or (self.toggle_probability > np.random.random())
                ):
                    module.toggle_lora()
                    reset_gm_score_attributes(module)

        super().new_epoch()
