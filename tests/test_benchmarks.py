from __future__ import annotations

import unittest

import numpy as np
import pytest

from confopt.searchspace.darts.core.genotypes import Genotype as NB301Genotype
from confopt.searchspace.nb1shot1.core.search_spaces.genotypes import (
    NASBench1Shot1ConfoptGenotype,
)
from confopt.searchspace.nb201.core.genotypes import Structure as NB201Genotype
from confopt.searchspace.tnb101.core.genotypes import TNB101Genotype

nb201_genotype_fail = NB201Genotype(
    [
        (("none", 0),),
        (("none", 0), ("none", 1)),
        (("none", 0), ("none", 1), ("none", 2)),
    ]
)

nb201_genotype = NB201Genotype(
    [
        (("nor_conv_1x1", 0),),
        (("nor_conv_1x1", 0), ("nor_conv_1x1", 1)),
        (("nor_conv_3x3", 0), ("skip_connect", 1), ("none", 2)),
    ]
)

nb301_genotype = NB301Genotype(
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

nb301_genotype_fail = NB301Genotype(
    normal=[
        ("skip_connect", 0),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("skip_connect", 1),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("skip_connect", 0),
        ("skip_connect", 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("skip_connect", 0),
        ("skip_connect", 1),
        ("skip_connect", 2),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("skip_connect", 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)

test_nb301_acc = 94.166695
test_nb301_fail_acc = 88.165985

tnb101_genotype_fail = TNB101Genotype(
    node_edge_dict={
        1: [("none", 0)],
        2: [("none", 0), ("none", 1)],
        3: [("none", 0), ("none", 1), ("none", 2)],
    },
    op_idx_list=[0, 0, 0, 0, 0, 0],
)

tnb101_genotype = TNB101Genotype(
    node_edge_dict={
        1: [("nor_conv_3x3", 0)],
        2: [("nor_conv_3x3", 0), ("nor_conv_1x1", 1)],
        3: [("skip_connect", 0), ("none", 1), ("nor_conv_1x1", 2)],
    },
    op_idx_list=[3, 3, 2, 1, 0, 2],
)

test_tnb101_acc_top1 = 50.8658447265625
test_tnb101_fail_acc_top1 = 29.433717727661133



INPUT = "input"
OUTPUT = "output"
CONV1X1 = "conv1x1-bn-relu"
CONV3X3 = "conv3x3-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
matrix=np.array([[0, 1, 1, 1, 0, 1, 0],    # input layer
        [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
        [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
        [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
        [0, 0, 0, 0, 0, 0, 0]])   # output layer
# Operations at the vertices of the module, matches order of matrix
ops=np.array([INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])
nb101_genotype = NASBench1Shot1ConfoptGenotype(matrix=matrix, ops=ops)

class TestBenchmarks(unittest.TestCase):
    @pytest.mark.benchmark()  # type: ignore
    def test_nb201_benchmark(self) -> None:
        from confopt.benchmark import NB201Benchmark

        api = NB201Benchmark()
        query_result = api.query(nb201_genotype)

        # check cifar 10
        dataset = "cifar10"
        train_result = 99.78
        test_result = 92.32
        assert query_result[f"benchmark/{dataset}/train_top1"] == train_result
        assert query_result[f"benchmark/{dataset}/test_top1"] == test_result

        # check cifar100
        dataset = "cifar100"
        train_result = 91.19
        valid_result = 67.7
        test_result = 67.94
        assert query_result[f"benchmark/{dataset}/train_top1"] == train_result
        assert query_result[f"benchmark/{dataset}/valid_top1"] == valid_result
        assert query_result[f"benchmark/{dataset}/test_top1"] == test_result

        # check imagenet
        dataset="imagenet16"
        train_result = 46.84
        valid_result = 41.0
        test_result = 41.47
        assert query_result[f"benchmark/{dataset}/train_top1"] == train_result
        assert query_result[f"benchmark/{dataset}/valid_top1"] == valid_result
        assert query_result[f"benchmark/{dataset}/test_top1"] == test_result

    @pytest.mark.benchmark()  # type: ignore
    def test_nb201_benchmark_fail(self) -> None:
        from confopt.benchmark import NB201Benchmark

        api = NB201Benchmark()
        query_result = api.query(nb201_genotype_fail)

        # check cifar 10
        dataset = "cifar10"
        assert query_result[f"benchmark/{dataset}/train_top1"] == 10.0
        assert query_result[f"benchmark/{dataset}/valid_top1"] == 0.0
        assert query_result[f"benchmark/{dataset}/test_top1"] == 10.0

        # check cifar100
        dataset="cifar100"
        assert query_result[f"benchmark/{dataset}/train_top1"] == 1.0
        assert query_result[f"benchmark/{dataset}/valid_top1"] == 1.0
        assert query_result[f"benchmark/{dataset}/test_top1"] == 1.0

        # check imagenet
        dataset="imagenet16"
        assert query_result[f"benchmark/{dataset}/train_top1"] == 0.86
        assert query_result[f"benchmark/{dataset}/valid_top1"] == 0.83
        assert query_result[f"benchmark/{dataset}/test_top1"] == 0.83

    @pytest.mark.benchmark()  # type: ignore
    def test_nb301_benchmark(self) -> None:
        from confopt.benchmark import NB301Benchmark

        api = NB301Benchmark()
        query_result = api.query(nb301_genotype, with_noise=False)

        self.assertAlmostEqual(  # noqa: PT009
            query_result["benchmark/test_top1"], test_nb301_acc, 4
        )

    @pytest.mark.benchmark()  # type: ignore
    def test_nb301_benchmark_fail_genotype(self) -> None:
        from confopt.benchmark import NB301Benchmark

        api = NB301Benchmark()
        query_result = api.query(nb301_genotype_fail, with_noise=False)

        assert query_result["benchmark/test_top1"] < 89.0

    @pytest.mark.benchmark()  # type: ignore
    def test_tnb101_benchmark(self) -> None:
        from confopt.benchmark import TNB101Benchmark

        api = TNB101Benchmark()
        query_result = api.query(tnb101_genotype)

        assert query_result["benchmark/test_top1"] == test_tnb101_acc_top1

    @pytest.mark.benchmark()  # type: ignore
    def test_tnb101_benchmark_fail(self) -> None:
        from confopt.benchmark import TNB101Benchmark

        api = TNB101Benchmark()
        query_result = api.query(tnb101_genotype_fail)

        assert query_result["benchmark/test_top1"] == test_tnb101_fail_acc_top1

    @pytest.mark.benchmark()  # type: ignore
    def test_nb101_benchmark(self) -> None:
        from confopt.benchmark import NB101Benchmark

        api = NB101Benchmark("only108")
        import random
        random.seed(0)
        query_result = api.query(nb101_genotype, dataset="cifar10", epochs=108)

        self.assertAlmostEqual(  # noqa: PT009
            query_result["benchmark/training_time"], 1157.675048828125, 4
        )

        self.assertAlmostEqual(  # noqa: PT009
            query_result["benchmark/test_top1"],0.932692289352417, 4
        )

        assert query_result["benchmark/trainable_parameters"] == 2694282

if __name__ == "__main__":
    unittest.main()
