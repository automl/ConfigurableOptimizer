from __future__ import annotations

import os
from typing import Literal

import gdown
from nas_201_api import NASBench201API

from confopt.searchspace.nb201.core.genotypes import Structure as Genotype

from .benchmark_base import BenchmarkBase


class NB201Benchmark(BenchmarkBase):
    def __init__(self) -> None:
        self.api_dir = "api/nb201"
        # self.api_file_name = "NAS-Bench-201-v1_1-096897.pth" # newer version
        self.api_file_name = "NAS-Bench-201-v1_0-e61699.pth"
        self.api_path = f"{self.api_dir}/{self.api_file_name}"
        self.download_api()
        api = NASBench201API(self.api_path)
        super().__init__(api)

    def download_api(self) -> None:
        # file_id = "16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_" # newer version (4.7 GB)
        file_id = "1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs"  # older version (2.2 GB)
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
        dataset: Literal[  # type: ignore
            "cifar10", "cifar100", "imagenet16"
        ] = "cifar10",
        **api_kwargs: str,  # noqa: ARG002
    ) -> dict:
        result = self.api.query_by_arch(genotype, hp="200")
        result_train, result_valid, result_test = self._distill_result(  # type: ignore
            result, dataset
        )

        results_metric = {
            "benchmark/train_top1": result_train,
            "benchmark/valid_top1": result_valid,
            "benchmark/test_top1": result_test,
        }

        return results_metric

    def _distill_result(
        self,
        result: str,
        dataset: Literal["cifar10", "cifar100", "imagenet16"] = "cifar10",
    ) -> tuple[float, float, float]:
        result = result.split("\n")  # type: ignore

        if dataset == "cifar10":
            cifar10 = result[5].replace(" ", "").split(":")
            cifar10_train = float(
                cifar10[1].strip(",test")[-7:-2].strip("=")  # noqa: B005
            )
            cifar10_test = float(cifar10[2][-7:-2].strip("="))
            return cifar10_train, 0.0, cifar10_test

        elif dataset == "cifar100":  # noqa: RET505
            cifar100 = result[7].replace(" ", "").split(":")
            cifar100_train = float(cifar100[1].strip(",valid")[-7:-2].strip("="))
            cifar100_valid = float(
                cifar100[2].strip(",test")[-7:-2].strip("=")  # noqa: B005
            )
            cifar100_test = float(cifar100[3][-7:-2].strip("="))
            return cifar100_train, cifar100_valid, cifar100_test

        elif dataset == "imagenet16":
            imagenet16 = result[9].replace(" ", "").split(":")
            imagenet16_train = float(imagenet16[1].strip(",valid")[-7:-2].strip("="))
            imagenet16_valid = float(
                imagenet16[2].strip(",test")[-7:-2].strip("=")  # noqa: B005
            )
            imagenet16_test = float(imagenet16[3][-7:-2].strip("="))

            return imagenet16_train, imagenet16_valid, imagenet16_test

        else:
            raise ValueError(f"Dataset {dataset} is not supported with NB201 API")
