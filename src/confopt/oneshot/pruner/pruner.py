from __future__ import annotations

import warnings

from confopt.oneshot.base_component import OneShotComponent
from confopt.searchspace import SearchSpace


class Pruner(OneShotComponent):
    def __init__(
        self,
        searchspace: SearchSpace,
        prune_epochs: list[int],
        prune_widers: list[int | None] | None = None,
    ):
        super().__init__()
        self.prune_epochs = prune_epochs
        self.searchspace = searchspace
        self.use_prune = True
        if not hasattr(searchspace, "prune"):
            warnings.warn(
                f"The searchspace {type(searchspace)} does not have "
                + "prune functionality",
                stacklevel=1,
            )
            self.use_prune = False

        self.prune_epoch_to_wider = {}
        if prune_widers is not None:
            assert len(prune_widers) == len(prune_epochs)
            for idx, epoch in enumerate(prune_epochs):
                self.prune_epoch_to_wider[epoch] = prune_widers[idx]

    def new_epoch(self) -> None:
        super().new_epoch()
        if self.use_prune and self._epoch in self.prune_epochs:
            # apply the pruning mask
            self.searchspace.prune(wider=self.prune_epoch_to_wider.get(self._epoch))
