import unittest

import torch

from confopt.oneshot.archsampler import (
    BaseSampler,
    DARTSSampler,
    DRNASSampler,
    GDASSampler,
    ReinMaxSampler,
)


def _test_arch_combine_fn_default(
    sampler: BaseSampler, alphas: list[torch.Tensor]
) -> None:
    alphas_normal, alphas_reduction = _sample_alphas(sampler, alphas)

    assert torch.allclose(alphas_normal.sum(dim=-1), torch.ones((14,)))
    assert torch.allclose(alphas_reduction.sum(dim=-1), torch.ones((14,)))


def _test_arch_combine_fn_sigmoid(
    sampler: BaseSampler, alphas: list[torch.Tensor]
) -> None:
    alphas_normal, alphas_reduction = _sample_alphas(sampler, alphas)

    assert not torch.allclose(alphas_normal.sum(dim=-1), torch.ones((14,)))
    assert not torch.allclose(alphas_reduction.sum(dim=-1), torch.ones((14,)))


def _sample_alphas(
    sampler: BaseSampler, alphas: list[torch.Tensor]
) -> list[torch.Tensor]:
    sampled_alphas = []
    for alpha in alphas:
        sampled_alphas.append(sampler.sample(alpha))
    return sampled_alphas


class TestDARTSSampler(unittest.TestCase):
    def test_post_sample_fn_default(self) -> None:
        alphas = [torch.randn(14, 8), torch.randn(14, 8)]
        sampler = DARTSSampler(
            alphas, sample_frequency="epoch", arch_combine_fn="default"
        )
        _test_arch_combine_fn_default(sampler, alphas)

    def test_post_sample_fn_sigmoid(self) -> None:
        alphas = [torch.randn(14, 8), torch.randn(14, 8)]
        sampler = DARTSSampler(
            alphas, sample_frequency="epoch", arch_combine_fn="sigmoid"
        )
        _test_arch_combine_fn_sigmoid(sampler, alphas)
        alphas_normal, alphas_reduction = _sample_alphas(sampler, alphas)
        assert torch.allclose(alphas_normal, torch.nn.functional.sigmoid(alphas[0]))
        assert torch.allclose(alphas_reduction, torch.nn.functional.sigmoid(alphas[1]))


class TestDRNASSampler(unittest.TestCase):
    def test_post_sample_fn_default(self) -> None:
        alphas = [torch.randn(14, 8), torch.randn(14, 8)]
        sampler = DRNASSampler(
            alphas, sample_frequency="epoch", arch_combine_fn="default"
        )
        _test_arch_combine_fn_default(sampler, alphas)

    def test_post_sample_fn_sigmoid(self) -> None:
        alphas = [torch.randn(14, 8), torch.randn(14, 8)]
        sampler = DRNASSampler(
            alphas, sample_frequency="epoch", arch_combine_fn="sigmoid"
        )
        _test_arch_combine_fn_sigmoid(sampler, alphas)


class TestGDASSampler(unittest.TestCase):
    def test_post_sample_fn_default(self) -> None:
        alphas = [torch.randn(14, 8), torch.randn(14, 8)]
        sampler = GDASSampler(
            alphas, sample_frequency="epoch", arch_combine_fn="default"
        )
        _test_arch_combine_fn_default(sampler, alphas)

    def test_post_sample_fn_sigmoid(self) -> None:
        alphas = [torch.randn(14, 8), torch.randn(14, 8)]
        sampler = GDASSampler(
            alphas, sample_frequency="epoch", arch_combine_fn="sigmoid"
        )
        # GDAS must ignore the "sigmoid" arch_combine_fn since it respecting it
        # wouldn't really make a difference, considering that only one operation
        # is chosen per edge
        _test_arch_combine_fn_default(sampler, alphas)


class TestReinmaxSampler(unittest.TestCase):
    def test_post_sample_fn_default(self) -> None:
        alphas = [torch.randn(14, 8), torch.randn(14, 8)]
        sampler = ReinMaxSampler(
            alphas, sample_frequency="epoch", arch_combine_fn="default"
        )
        _test_arch_combine_fn_default(sampler, alphas)

    def test_post_sample_fn_sigmoid(self) -> None:
        alphas = [torch.randn(14, 8), torch.randn(14, 8)]
        sampler = ReinMaxSampler(
            alphas, sample_frequency="epoch", arch_combine_fn="sigmoid"
        )
        # Reinmax must ignore the "sigmoid" arch_combine_fn since it respecting it
        # wouldn't really make a difference, considering that only one operation
        # is chosen per edge
        _test_arch_combine_fn_default(sampler, alphas)


if __name__ == "__main__":
    unittest.main()
