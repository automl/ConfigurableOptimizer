from __future__ import annotations

import os
from typing import Literal

import gdown

from confopt.benchmarks.benchmark_base import BenchmarkBase
from confopt.searchspace.tnb101.core.genotypes import TNB101Genotype as Genotype
from confopt.utils import TransNASBenchAPI


class TNB101Benchmark(BenchmarkBase):
    def __init__(self) -> None:
        self.api_dir = "api/tnb101"
        self.api_file_name = "transnas-bench_v10141024.pth"
        self.api_path = f"{self.api_dir}/{self.api_file_name}"
        self.download_api()
        api = TransNASBenchAPI(self.api_path)
        self.search_space_name = "micro"
        super().__init__(api)

    def download_api(self) -> None:
        file_id = "1BV_BRMsCUVBtSVj4SN4QmA9Pjd35B2M4"
        if not os.path.exists(self.api_path):
            if not os.path.exists(self.api_dir):
                os.makedirs(self.api_dir)
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output=self.api_path,
            )

    def query(
        self,
        genotype: Genotype,
        # TODO: After TNB101 Datasets classes' names are decided, revisit this
        dataset: Literal["class_scene", "class_object"] = "class_object",
    ) -> dict:
        assert (
            dataset in self.api.task_list
        ), "The dataset is not one of valid task for TNB101SearchSpace"
        task = dataset
        metrics_list = self.api.metrics_dict[task]
        arch_str = genotype.get_arch_str()

        results_metric = {}
        for metric in metrics_list:
            result = self.api.get_single_metric(arch_str, task, metric, mode="best")
            results_metric[f"benchmark/{metric}"] = result

        return results_metric


if __name__ == "__main__":
    tnb101_api = TNB101Benchmark()
    genotype = Genotype(
        node_edge_dict={
            1: [("skip_connect", 0)],
            2: [("none", 0), ("nor_conv_1x1", 1)],
            3: [("nor_conv_3x3", 0), ("nor_conv_1x1", 1), ("nor_conv_1x1", 2)],
        },
        op_idx_list=[1, 0, 2, 3, 2, 2],
    )
    print(tnb101_api.query(genotype))
