from __future__ import annotations

from abc import abstractmethod
from typing import Any


class BenchmarkBase:
    def __init__(self, api: Any) -> None:
        self.api = api

    @abstractmethod
    def query(
        self, genotype: Any, dataset: str = "cifar10"
    ) -> tuple[float, float, float]:
        pass

    @abstractmethod
    def download_api(self) -> None:
        pass
