from __future__ import annotations

from typing import TYPE_CHECKING

from confopt.oneshot.base import OneShotComponent
from confopt.searchspace.common.mixop import DynamicAttentionNetwork

if TYPE_CHECKING:
    from confopt.searchspace.common.base_search import SearchSpace


class DynamicAttentionExplorer(OneShotComponent):
    def __init__(
        self,
        searchspace: SearchSpace,
        total_epochs: int,
        attention_weight: float = 1,
        min_attention_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.searchspace = searchspace
        self.init_attention_weight = attention_weight
        self.min_attention_weight = min_attention_weight

        assert total_epochs not in [0, None], f"{total_epochs} is invalid"
        self.decay_rate = self.min_attention_weight ** (1 / total_epochs)

    def get_attention_weight(self) -> float:
        return self.init_attention_weight * (self.decay_rate**self._epoch)

    def update_weights(self) -> None:
        attention_weight = self.get_attention_weight()
        for module in self.searchspace.modules():
            if isinstance(module, DynamicAttentionNetwork):
                module.update_attention_weight(attention_weight)

    def new_epoch(self) -> None:
        self.update_weights()
        super().new_epoch()
