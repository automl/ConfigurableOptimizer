from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from confopt.searchspace import SearchSpace
from confopt.searchspace.common.base_search import OperationStatisticsSupport
from confopt.utils import TrainingMetrics


class EarlyStopper:
    def __init__(self) -> None:
        self.epoch = 0
        self.search_losses: dict[int, float] = {}
        self.search_accs_top1: dict[int | str, float | int] = {"best": -1}
        self.search_accs_top5: dict[int, float] = {}
        self.valid_losses: dict[int, float] = {}
        self.valid_accs_top1: dict[int | str, float | int] = {"best": -1}
        self.valid_accs_top5: dict[int, float] = {}

    def check_stop(
        self,
        epoch: int,
        model: SearchSpace,
        search_metrics: TrainingMetrics,
        valid_metrics: TrainingMetrics,
    ) -> bool:
        (
            self.search_losses[epoch],
            self.search_accs_top1[epoch],
            self.search_accs_top5[epoch],
        ) = search_metrics

        (
            self.valid_losses[epoch],
            self.valid_accs_top1[epoch],
            self.valid_accs_top5[epoch],
        ) = valid_metrics

        arch_params = model.arch_parameters
        self.normal_alphas = arch_params[0]
        self.reduce_alphas = None

        if len(arch_params) == 2:
            self.reduce_alphas = arch_params[1]
        elif len(arch_params) > 2:
            raise Exception("More than two sets of architectural parameters found!")

        return self.is_stopping_condition_met(epoch, model)

    @abstractmethod
    def is_stopping_condition_met(self, epoch: int, model: SearchSpace) -> bool:
        ...


class SkipConnectionEarlyStopper(EarlyStopper):
    def __init__(
        self,
        max_skip_normal: int,
        max_skip_reduce: int | None,
        min_epochs: int,
    ) -> None:
        super().__init__()
        self.max_skip_normal = max_skip_normal
        self.max_skip_reduce = max_skip_reduce
        self.min_epochs = min_epochs

    def is_stopping_condition_met(self, epoch: int, model: SearchSpace) -> bool:
        assert isinstance(
            model, OperationStatisticsSupport
        ), "SearchSpace has to implement OperationStatisticsSupport \
            to use SkipConnectionEarlyStopper"

        num_skip_ops = model.get_num_skip_ops()

        if (
            "op_counts/normal/skip_connect" in num_skip_ops
        ):  # Only for DARTS. TODO: Handle others later
            n_skip_normal = num_skip_ops["op_counts/normal/skip_connect"]
            n_skip_reduce = num_skip_ops.get("op_counts/reduce/skip_connect", 0)
        else:
            n_skip_normal = num_skip_ops["skip_connections/normal"]
            n_skip_reduce = num_skip_ops.get("skip_connections/reduce", 0)

        if epoch > self.min_epochs:
            print("n_skip_normal", n_skip_normal)
            print("n_skip_reduce", n_skip_reduce)
            if n_skip_normal > self.max_skip_normal:
                return True
            if (
                self.max_skip_reduce is not None
                and n_skip_reduce > self.max_skip_reduce
            ):
                return True

        return False
