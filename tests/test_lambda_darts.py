from __future__ import annotations

import unittest
import torch

from confopt.searchspace.common import LambdaReg
from confopt.searchspace import (
    DARTSSearchSpace,
    NASBench201SearchSpace,
    TransNASBench101SearchSpace,
    SearchSpace,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestLambda(unittest.TestCase):

    def _forward_and_backward_pass(self, model: SearchSpace) -> SearchSpace:
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(10, (2,)).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()

        _, preds = model(x)

        loss = criterion(preds, y)
        loss.backward()

        return model

    def test_shapes_darts(self) -> None:
        search_space = DARTSSearchSpace().to(DEVICE)
        search_space = self._forward_and_backward_pass(search_space)
        self._test_shapes(search_space, (6, 2), (14, 8))

    def test_shapes_nb201(self) -> None:
        search_space = NASBench201SearchSpace().to(DEVICE)
        search_space = self._forward_and_backward_pass(search_space)
        self._test_shapes(search_space, (15, 0), (6, 5))

    def test_shapes_tnb101(self) -> None:
        search_space = TransNASBench101SearchSpace().to(DEVICE)
        search_space = self._forward_and_backward_pass(search_space)
        self._test_shapes(search_space, (10, 0), (6, 4))

    def test_perturbations_disabled_darts(self) -> None:
        search_space = DARTSSearchSpace().to(DEVICE)
        self._test_perturbations_disabled(search_space)

    def test_perturbations_disabled_nb201(self) -> None:
        search_space = NASBench201SearchSpace().to(DEVICE)
        self._test_perturbations_disabled(search_space)

    def test_perturbations_disabled_tnb101(self) -> None:
        search_space = TransNASBench101SearchSpace().to(DEVICE)
        self._test_perturbations_disabled(search_space)

    def test_perturbations_enabled_darts(self) -> None:
        search_space = DARTSSearchSpace().to(DEVICE)
        self._test_perturbations_enabled(search_space)

    def test_perturbations_enabled_nb201(self) -> None:
        search_space = NASBench201SearchSpace().to(DEVICE)
        self._test_perturbations_enabled(search_space)

    def test_perturbations_enabled_tnb101(self) -> None:
        search_space = TransNASBench101SearchSpace().to(DEVICE)
        self._test_perturbations_enabled(search_space)

    def _test_shapes(
        self, search_space: SearchSpace, n_cells: tuple, grads_shape: tuple
    ) -> None:
        grads_normal, grads_reduce = search_space.model.get_arch_grads()
        has_reduce = n_cells[1] > 0

        def assert_shape_correct(grads: list, n_cells: int, grads_shape: tuple) -> None:
            assert isinstance(grads, list)
            assert len(grads) == n_cells

            for grad in grads:
                assert grad.shape == (grads_shape[0] * grads_shape[1],)

        assert_shape_correct(grads_normal, n_cells[0], grads_shape)
        if has_reduce:
            assert_shape_correct(grads_reduce, n_cells[1], grads_shape)
        else:
            assert grads_reduce is None

        perts = search_space.get_perturbations()

        for p in perts:
            assert p.shape == grads_shape

    def _get_grads(self, model: SearchSpace) -> list[torch.Tensor]:
        grads = []
        for p in model.model_weight_parameters():
            if p.grad is not None:
                grads.append(p.grad.clone())

        return grads

    def _test_perturbations_disabled(self, model: SearchSpace) -> None:
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(10, (2,)).to(DEVICE)

        model.disable_lambda_darts()

        criterion = torch.nn.CrossEntropyLoss()
        _, preds = model(x)

        loss = criterion(preds, y)
        loss.backward()

        old_grads = self._get_grads(model)
        model.add_lambda_regularization(x, y, criterion)
        new_grads = self._get_grads(model)

        for new_grad, old_grad in zip(new_grads, old_grads):
            assert (new_grad == old_grad).all()

    def _test_perturbations_enabled(self, model: SearchSpace) -> None:
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(10, (2,)).to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss()
        _, preds = model(x)

        loss = criterion(preds, y)
        loss.backward()

        model.lambda_reg.enabled = True

        old_grads = self._get_grads(model)
        model.add_lambda_regularization(x, y, criterion)
        new_grads = self._get_grads(model)


if __name__ == "__main__":
    unittest.main()
