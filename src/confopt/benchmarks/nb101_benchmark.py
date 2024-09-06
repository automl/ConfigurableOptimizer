from __future__ import annotations

import hashlib
import os
from typing import Any, Literal
import urllib.request

from nasbench import api

from confopt.benchmarks.benchmark_base import BenchmarkBase
from confopt.searchspace.nb1_shot_1.core.search_spaces.genotypes import (
    NASBench1Shot1ConfoptGenotype,
)


class NB101Benchmark(BenchmarkBase):
    def __init__(
        self,
        benchmark_type: Literal["full", "only108"],
    ) -> None:
        self.api_dir = "api/nb101"
        self.benchmark_type = benchmark_type

        if benchmark_type == "full":
            self.api_path = os.path.join(self.api_dir, "nasbench_full.tfrecord")
        elif benchmark_type == "only108":
            self.api_path = os.path.join(self.api_dir, "nasbench_only108.tfrecord")
        else:
            raise ValueError(f"Unknown benchmark type {benchmark_type}")

        self.file_urls = {
            "full": "https://storage.googleapis.com/nasbench/nasbench_full.tfrecord",  # noqa: E501
            "only108": "https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord",  # noqa: E501
        }

        if os.path.exists(self.api_path):
            self.verify_api()
        else:
            self.download_api()

        query_model = api.NASBench(self.api_path)

        super().__init__(query_model)

    def download_api(self) -> None:
        # download model version
        if not os.path.exists(self.api_dir):
            os.makedirs(self.api_dir)

        # URL of the file
        url = self.file_urls[self.benchmark_type]

        # Path to save the file
        file_path = self.api_path

        print("Downloading benchmark files. This might take a while...")
        urllib.request.urlretrieve(url, file_path)

        self.verify_api()
        print(f"File downloaded successfully and saved as {file_path}")

    def verify_api(self) -> bool:
        # Function to calculate SHA256
        def calculate_sha256(file_path: str) -> str:
            sha256_hash = hashlib.sha256()

            # Open the file in binary mode
            with open(file_path, "rb") as f:
                # Read and update hash in chunks of 4K
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            return sha256_hash.hexdigest()

        # Expected SHA256 hash
        expected_hash = {
            "full": "3d64db8180fb1b0207212f9032205064312b6907a3bbc81eabea10db2f5c7e9c",  # noqa: E501
            "only108": "4c39c3936e36a85269881d659e44e61a245babcb72cb374eacacf75d0e5f4fd1",  # noqa: E501
        }

        # Calculate the hash of the file
        file_hash = calculate_sha256(self.api_path)

        assert file_hash == expected_hash[self.benchmark_type], (
            "SHA256 hash of the file does not match the expected hash."
            + "Please download the file again."
        )

        return True

    def query(
        self,
        genotype: NASBench1Shot1ConfoptGenotype,
        dataset: str = "cifar10",
        **api_kwargs: Any,
    ) -> dict:
        if dataset != "cifar10":
            raise ValueError(f"Dataset {dataset} is not supported with NB101 API")

        epochs = api_kwargs["epochs"] if "epochs" in api_kwargs else 108

        if self.benchmark_type == "only108" and epochs != 108:
            raise ValueError(
                "Only 108 epochs are available in the 'only108' benchmark."
                + "Please use epoch=108 or switch to the 'full' benchmark "
                + "for epochs [4, 12, 36, 108]."
            )

        if (self.benchmark_type == "full") and (epochs not in [4, 12, 36, 108]):
            raise ValueError(
                "Invalid epochs: {epochs}. Only the following epochs are "
                + "available in the full benchmark: [4, 12, 36, 108]."
            )

        model_spec = api.ModelSpec(
            matrix=genotype.matrix,
            ops=genotype.ops,
        )
        result = self.api.query(model_spec, epochs=epochs)
        result = {f"benchmark/{k}": v for k, v in result.items()}
        result["benchmark/test_top1"] = result["benchmark/test_accuracy"]
        del result["benchmark/test_accuracy"]

        return result


if __name__ == "__main__":
    nb101_benchmark = NB101Benchmark("only108")

    INPUT = "input"
    OUTPUT = "output"
    CONV1X1 = "conv1x1-bn-relu"
    CONV3X3 = "conv3x3-bn-relu"
    MAXPOOL3X3 = "maxpool3x3"
    OUTPUT_NODE = 6

    model_spec = NASBench1Shot1ConfoptGenotype(
        # Adjacency matrix of the module
        matrix=[
            [0, 1, 1, 1, 0, 1, 0],  # input layer
            [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
            [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
            [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
            [0, 0, 0, 0, 0, 0, 0],
        ],  # output layer
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT],
    )

    # Query this model from dataset, returns a dictionary containing the metrics
    # associated with this model.
    test_accuracies: set[str] = set()

    while len(test_accuracies) < 3:
        data = nb101_benchmark.query(model_spec, epochs=108)
        test_accuracies.add(data["benchmark/test_accuracy"])
        print(data)

    print(test_accuracies)
