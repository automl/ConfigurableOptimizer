from __future__ import annotations

import unittest

import torch

from confopt.oneshot.archsampler import (
    BaseSampler,
    DARTSSampler,
    DRNASSampler,
    GDASSampler,
    SNASSampler,
)
from confopt.oneshot.archsampler.reinmax.sampler import ReinMaxSampler
from confopt.oneshot.dropout import Dropout
from confopt.oneshot.perturbator import SDARTSPerturbator
from confopt.searchspace import NASBench201SearchSpace

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestArchSamplers(unittest.TestCase):
    def _sampler_new_step_or_epoch(
        self, sampler: BaseSampler, sample_frequency: str
    ) -> None:
        if sample_frequency == "epoch":
            sampler.new_epoch()
        elif sample_frequency == "step":
            sampler.new_step()
        else:
            raise ValueError(f"Unknown sample_frequency: {sample_frequency}")

    def assert_rows_one_hot(self, alphas: list[torch.Tensor]) -> None:
        for row in alphas:
            assert torch.sum(row == 1.0) == 1
            assert torch.sum(row == 0.0) == row.numel() - 1

    def test_darts_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = DARTSSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        # assert that the tensors are close
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

    def _test_darts_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = DARTSSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency,
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        # assert that the tensors are close
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

    def test_darts_sampler_new_step(self) -> None:
        self._test_darts_sampler_new_step_epoch(sample_frequency="step")

    def test_darts_sampler_new_epoch(self) -> None:
        self._test_darts_sampler_new_step_epoch(sample_frequency="epoch")

    def test_gdas_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = GDASSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
            self.assert_rows_one_hot(arch_param_after)

    def _test_gdas_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = GDASSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency,
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
            self.assert_rows_one_hot(arch_param_after)

    def test_gdas_sampler_new_step(self) -> None:
        self._test_gdas_sampler_new_step_epoch(sample_frequency="step")

    def test_gdas_sampler_new_epoch(self) -> None:
        self._test_gdas_sampler_new_step_epoch(sample_frequency="epoch")

    def test_reinmax_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = ReinMaxSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
            self.assert_rows_one_hot(arch_param_after)

    def _test_reinmax_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = ReinMaxSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency,
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
            self.assert_rows_one_hot(arch_param_after)

    def test_reinmax_sampler_new_step(self) -> None:
        self._test_reinmax_sampler_new_step_epoch(sample_frequency="step")

    def test_reinmax_sampler_new_epoch(self) -> None:
        self._test_reinmax_sampler_new_step_epoch(sample_frequency="epoch")

    def test_drnas_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = DRNASSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

            for row in arch_param_after:
                assert torch.allclose(torch.sum(row), torch.Tensor([1.0]).to(DEVICE))

    def _test_drnas_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = DRNASSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency,
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

            for row in arch_param_after:
                assert torch.allclose(torch.sum(row), torch.Tensor([1.0]).to(DEVICE))

    def test_drnas_sampler_new_step(self) -> None:
        self._test_drnas_sampler_new_step_epoch(sample_frequency="step")

    def test_drnas_sampler_new_epoch(self) -> None:
        self._test_drnas_sampler_new_step_epoch(sample_frequency="epoch")

    def test_snas_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = SNASSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

    def _test_snas_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = SNASSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency,
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

    def test_snas_sampler_new_step(self) -> None:
        self._test_snas_sampler_new_step_epoch(sample_frequency="step")

    def test_snas_sampler_new_epoch(self) -> None:
        self._test_snas_sampler_new_step_epoch(sample_frequency="epoch")

    def _test_snas_illegal_temperatures(
        self, temp_init: float, temp_min: float
    ) -> None:
        arch_parameters = [torch.randn(5, 5)]
        with self.assertRaises(AssertionError):
            SNASSampler(
                arch_parameters=arch_parameters,
                temp_init=temp_init,
                temp_min=temp_min,
            )

    def test_snas_temperature_lower_bound(self) -> None:
        self._test_snas_illegal_temperatures(1.0, -1.0)

    def test_snas_temperature_upper_bound(self) -> None:
        self._test_snas_illegal_temperatures(1.1, 0.1)

    def test_snas_temperature_mixedup(self) -> None:
        self._test_snas_illegal_temperatures(0.1, 1.0)

    def test_sdarts_perturbator(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        epsilon = 0.03
        loss_criterion = torch.nn.CrossEntropyLoss()
        X = torch.randn(2, 3, 32, 32).to(DEVICE)
        target = torch.randint(0, 9, (2,)).to(DEVICE)

        perturbator = SDARTSPerturbator(
            search_space=searchspace,
            loss_criterion=loss_criterion,
            arch_parameters=searchspace.arch_parameters,
            epsilon=epsilon,
            data=(X, target),
        )

        # Random Attack
        alphas_before = searchspace.arch_parameters
        alphas_after = perturbator.perturb_alphas(alphas_before)
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

        # Adverserial Attack
        # Changes the model's alpha as well, but if the loss does not decrease, it does
        # not change alpha
        # TODO Improve this test
        perturbator.attack_type = "adversarial"
        alphas_before = [
            arch_param.clone() for arch_param in searchspace.arch_parameters
        ]

        alphas_after = perturbator.perturb_alphas(searchspace.arch_parameters)

    def test_illegal_sample_frequency(self) -> None:
        arch_parameters = [torch.randn(5, 5)]
        with self.assertRaises(AssertionError):
            DARTSSampler(arch_parameters=arch_parameters, sample_frequency="illegal")

        with self.assertRaises(AssertionError):
            GDASSampler(arch_parameters=arch_parameters, sample_frequency="illegal")

        with self.assertRaises(AssertionError):
            ReinMaxSampler(arch_parameters=arch_parameters, sample_frequency="illegal")

        with self.assertRaises(AssertionError):
            DRNASSampler(arch_parameters=arch_parameters, sample_frequency="illegal")

        with self.assertRaises(AssertionError):
            SNASSampler(
                arch_parameters=arch_parameters,
                sample_frequency="illegal",
            )

        with self.assertRaises(AssertionError):
            SDARTSPerturbator(
                arch_parameters=arch_parameters,
                sample_frequency="illegal",
                epsilon=0.03,
            )


class TestDropout(unittest.TestCase):
    def test_dropout_probability(self) -> None:
        probability = 0.1
        arch_parameters = torch.ones(1000)

        dropout = Dropout(p=probability)
        output = dropout.apply_mask(arch_parameters)
        dropped_percent = (1000 - torch.count_nonzero(output)) / 1000

        self.assertAlmostEqual(
            probability, dropped_percent.numpy(), places=1
        )  # type: ignore

    def test_negative_probability(self) -> None:
        self._test_probabilities(-1.0)

    def test_too_large_probability(self) -> None:
        self._test_probabilities(1.0)

    def _test_probabilities(self, probability: float) -> None:
        with self.assertRaises(AssertionError):
            Dropout(p=probability)

    def test_illegal_anneal_frequency(self) -> None:
        with self.assertRaises(AssertionError):
            Dropout(p=0.5, anneal_frequency="illegal")

    def test_illegal_anneal_type_and_frequency(self) -> None:
        with self.assertRaises(AssertionError):
            Dropout(p=0.5, anneal_frequency="epoch")

        with self.assertRaises(AssertionError):
            Dropout(p=0.5, anneal_type="linear")


if __name__ == "__main__":
    unittest.main()
