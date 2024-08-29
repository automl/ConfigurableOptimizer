from __future__ import annotations

import unittest

import pytest

from confopt.searchspace.darts.core.genotypes import Genotype as NB301Genotype
from confopt.searchspace.nb201.core.genotypes import Structure as NB201Genotype

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


class TestBenchmarks(unittest.TestCase):
    @pytest.mark.benchmark()  # type: ignore
    def test_nb201_benchmark(self) -> None:
        from confopt.benchmarks import NB201Benchmark

        api = NB201Benchmark()

        # check cifar 10
        query_result = api.query(nb201_genotype, dataset="cifar10")
        train_result = 99.78
        test_result = 92.32
        assert query_result["benchmark/train_top1"] == train_result
        assert query_result["benchmark/test_top1"] == test_result

        # check cifar100
        query_result = api.query(nb201_genotype, dataset="cifar100")
        train_result = 91.19
        valid_result = 67.7
        test_result = 67.94
        assert query_result["benchmark/train_top1"] == train_result
        assert query_result["benchmark/valid_top1"] == valid_result
        assert query_result["benchmark/test_top1"] == test_result

        # check imagenet
        query_result = api.query(nb201_genotype, dataset="imagenet16")
        train_result = 46.84
        valid_result = 41.0
        test_result = 41.47
        assert query_result["benchmark/train_top1"] == train_result
        assert query_result["benchmark/valid_top1"] == valid_result
        assert query_result["benchmark/test_top1"] == test_result

    @pytest.mark.benchmark()  # type: ignore
    def test_nb201_benchmark_fail(self) -> None:
        from confopt.benchmarks import NB201Benchmark

        api = NB201Benchmark()

        # check cifar 10
        query_result = api.query(nb201_genotype_fail, dataset="cifar10")
        assert query_result["benchmark/train_top1"] == 10.0
        assert query_result["benchmark/valid_top1"] == 0.0
        assert query_result["benchmark/test_top1"] == 10.0

        # check cifar100
        query_result = api.query(nb201_genotype_fail, dataset="cifar100")
        assert query_result["benchmark/train_top1"] == 1.0
        assert query_result["benchmark/valid_top1"] == 1.0
        assert query_result["benchmark/test_top1"] == 1.0

        # check imagenet
        query_result = api.query(nb201_genotype_fail, dataset="imagenet16")
        assert query_result["benchmark/train_top1"] == 0.86
        assert query_result["benchmark/valid_top1"] == 0.83
        assert query_result["benchmark/test_top1"] == 0.83

    @pytest.mark.benchmark()  # type: ignore
    def test_nb301_benchmark(self) -> None:
        from confopt.benchmarks import NB301Benchmark

        api = NB301Benchmark()
        query_result = api.query(nb301_genotype, with_noise=False)

        self.assertAlmostEqual(  # noqa: PT009
            query_result["benchmark/test_top1"], test_nb301_acc, 4
        )

    @pytest.mark.benchmark()  # type: ignore
    def test_nb301_benchmark_fail_genotype(self) -> None:
        from confopt.benchmarks import NB301Benchmark

        api = NB301Benchmark()
        query_result = api.query(nb301_genotype_fail, with_noise=False)

        assert query_result["benchmark/test_top1"] < 89.0


if __name__ == "__main__":
    unittest.main()
