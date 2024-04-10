from __future__ import annotations

import os
import unittest

import pytest

from confopt.benchmarks import NB201Benchmark, NB301Benchmark

from confopt.searchspace.darts.core.genotypes import Genotype as NB301Genotype
from confopt.searchspace.nb201.core.genotypes import Structure as NB201Genotype


test_nb201_genotype = NB201Genotype(
    [
        tuple([("nor_conv_1x1", 0)]),
        tuple([("nor_conv_1x1", 0), ("nor_conv_1x1", 1)]),
        tuple([("nor_conv_3x3", 0), ("skip_connect", 1), ("none", 2)]),
    ]
)


test_nb301_genotype = NB301Genotype(
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
test_nb301_result = 94.171960


class TestBenchmarks(unittest.TestCase):
    @pytest.mark.benchmark  # type: ignore
    def test_nb201_benchmark(self) -> None:
        api = NB201Benchmark()

        # check cifar 10
        query_result = api.query(test_nb201_genotype, dataset="cifar10")
        train_result = 99.78
        test_result = 92.32
        self.assertEqual(query_result[0], train_result)
        self.assertEqual(query_result[2], test_result)

        # check cifar100
        query_result = api.query(test_nb201_genotype, dataset="cifar100")
        train_result = 91.19
        valid_result = 67.7
        test_result = 67.94
        self.assertEqual(query_result[0], train_result)
        self.assertEqual(query_result[1], valid_result)
        self.assertEqual(query_result[2], test_result)

        # check imagenet
        query_result = api.query(test_nb201_genotype, dataset="imagenet16")
        train_result = 46.84
        valid_result = 41.0
        test_result = 41.47
        self.assertEqual(query_result[0], train_result)
        self.assertEqual(query_result[1], valid_result)
        self.assertEqual(query_result[2], test_result)

    @pytest.mark.benchmark  # type: ignore
    def test_nb301_benchmark(self) -> None:
        api = NB301Benchmark()
        query_result = api.query(test_nb301_genotype)
        self.assertAlmostEqual(query_result[-1], test_nb301_result, 4)


if __name__ == "__main__":
    unittest.main()
