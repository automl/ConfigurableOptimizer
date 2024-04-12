from __future__ import annotations

import os
from typing import Literal

import nasbench301 as nb301

from confopt.searchspace.darts.core.genotypes import Genotype

from .benchmark_base import BenchmarkBase


class NB301Benchmark(BenchmarkBase):
    def __init__(
        self,
        model_version: Literal["0.9", "1.0"] = "0.9",
        model_name: Literal["xgb", "gnn_gin", "lgb_runtime"] = "xgb",
    ) -> None:
        self.model_version = model_version
        self.model_name = model_name
        self.api_dir = "api/nb301"
        self.download_api()

        query_model_dir = self.model_paths[model_name]
        query_model = nb301.load_ensemble(query_model_dir)

        super().__init__(query_model)

    def download_api(self) -> None:
        # download model version
        if not os.path.exists(self.api_dir):
            os.makedirs(self.api_dir)

        models_0_9_dir = os.path.join(self.api_dir, "nb_models_0.9")
        model_paths_0_9 = {
            model_name: os.path.join(models_0_9_dir, f"{model_name}_v0.9")
            for model_name in ["xgb", "gnn_gin", "lgb_runtime"]
        }
        models_1_0_dir = os.path.join(self.api_dir, "nb_models_1.0")
        model_paths_1_0 = {
            model_name: os.path.join(models_1_0_dir, f"{model_name}_v1.0")
            for model_name in ["xgb", "gnn_gin", "lgb_runtime"]
        }
        self.model_paths = (
            model_paths_0_9 if self.model_version == "0.9" else model_paths_1_0
        )
        if not all(os.path.exists(model) for model in self.model_paths.values()):
            nb301.download_models(
                version=self.model_version, delete_zip=True, download_dir=self.api_dir
            )

    def query(
        self, genotype: Genotype, dataset: str = "cifar10", **api_kwargs: str
    ) -> tuple[float, float, float]:
        if dataset != "cifar10":
            raise ValueError(f"Dataset {dataset} is not supported with NB301 API")
        result_test = self.api.predict(
            config=genotype, representation="genotype", **api_kwargs
        )
        return 0.0, 0.0, result_test


if __name__ == "__main__":
    nb301_benchmark = NB301Benchmark()
    genotype_config = Genotype(
        normal=[
            ("sep_conv_3x3", 0),
            ("sep_conv_3x3", 1),
            ("sep_conv_3x3", 0),
            ("sep_conv_3x3", 1),
            ("sep_conv_3x3", 1),
            ("skip_connect", 0),
            ("skip_connect", 0),
            ("dil_conv_3x3", 2),
        ],
        normal_concat=[2, 3, 4, 5],
        reduce=[
            ("max_pool_3x3", 0),
            ("max_pool_3x3", 1),
            ("skip_connect", 2),
            ("max_pool_3x3", 1),
            ("max_pool_3x3", 0),
            ("skip_connect", 2),
            ("skip_connect", 2),
            ("max_pool_3x3", 1),
        ],
        reduce_concat=[2, 3, 4, 5],
    )
    print(nb301_benchmark.query(genotype_config))
