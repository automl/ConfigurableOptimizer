import copy
import unittest

import torch
from torch import nn

from confopt.searchspace import (
    DARTSSearchSpace,
    NASBench1Shot1SearchSpace,
    NASBench201SearchSpace,
    TransNASBench101SearchSpace,
)
from confopt.searchspace.darts.core.model_search import Cell as DARTSSearchCell
from confopt.searchspace.nb1shot1.core.model_search import (
    Cell as NasBench1Shot1SearchCell,
)
from confopt.searchspace.nb201.core import NAS201SearchCell
from confopt.searchspace.nb201.core.operations import ReLUConvBN, ResNetBasicblock
from confopt.searchspace.tnb101.core.model_search import TNB101SearchCell
from utils import get_modules_of_type

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestNASBench201SearchSpace(unittest.TestCase):
    def test_arch_parameters(self) -> None:
        search_space = NASBench201SearchSpace()
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 1
        assert isinstance(arch_params[0], nn.Parameter)

    def test_beta_parameters(self) -> None:
        search_space = NASBench201SearchSpace()
        beta_params = search_space.arch_parameters
        assert len(beta_params) == 1
        assert isinstance(beta_params[0], nn.Parameter)

    def test_forward_pass(self) -> None:
        search_space = NASBench201SearchSpace()
        x = torch.randn(2, 3, 32, 32).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 64])
        assert out[1].shape == torch.Size([2, 10])

    def test_forward_pass_with_edge_normalization(self) -> None:
        search_space = NASBench201SearchSpace(edge_normalization=True)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 64])
        assert out[1].shape == torch.Size([2, 10])

    def test_supernet_init(self) -> None:
        C = 32
        N = 3
        num_classes = 11
        search_space = NASBench201SearchSpace(C=C, N=N, num_classes=num_classes)

        search_cells = get_modules_of_type(search_space.model, NAS201SearchCell)
        assert len(search_cells) == N * 3

        resnet_cells = get_modules_of_type(search_space.model, ResNetBasicblock)
        assert len(resnet_cells) == 2

        first_cell_edge_ops = search_cells[0].edges["1<-0"].ops
        for op in first_cell_edge_ops:
            if isinstance(op, ReLUConvBN):
                assert op.op[1].in_channels == C
                assert op.op[1].out_channels == C

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_discretize_supernet(self) -> None:
        C = 32
        N = 3
        num_classes = 11
        search_space = NASBench201SearchSpace(C=C, N=N, num_classes=num_classes)

        search_cells = get_modules_of_type(search_space.model, NAS201SearchCell)
        assert len(search_cells) == N * 3
        # for cell in search_cells:
        #     cell._discretize(search_space.arch_parameters[0])

        resnet_cells = get_modules_of_type(search_space.model, ResNetBasicblock)
        assert len(resnet_cells) == 2

        search_cells[0].edges["1<-0"]
        # assert type(first_cell_edge_ops) in OPS.values()
        new_model = search_space._discretize()

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = new_model(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_prune(self) -> None:
        search_space = NASBench201SearchSpace(edge_normalization=True)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        search_space.prune()
        arch_params = search_space.arch_parameters[0]
        assert torch.count_nonzero(arch_params) == len(arch_params)
        assert torch.equal(
            torch.count_nonzero(arch_params, dim=-1),
            torch.ones(len(arch_params)).to(DEVICE),
        )
        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 64])
        assert out[1].shape == torch.Size([2, 10])

    def test_optim_forward_pass(self) -> None:
        search_space = NASBench201SearchSpace(edge_normalization=True)
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(low=0, high=9, size=(2,)).to(DEVICE)
        arch_optim = torch.optim.Adam(
            [*search_space.arch_parameters, *search_space.beta_parameters]
        )
        arch_optim.zero_grad()
        out = search_space(x)
        loss = loss_fn(out[1], y)
        loss.backward()
        alphas_before = copy.deepcopy(search_space.arch_parameters)
        betas_before = copy.deepcopy(search_space.beta_parameters)
        arch_optim.step()
        alphas_after = search_space.arch_parameters
        betas_after = search_space.beta_parameters
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
        for beta_param_before, beta_param_after in zip(betas_before, betas_after):
            assert not torch.allclose(beta_param_before, beta_param_after)


class TestDARTSSearchSpace(unittest.TestCase):
    def test_arch_parameters(self) -> None:
        search_space = DARTSSearchSpace()
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 2
        assert isinstance(arch_params[0], nn.Parameter)
        assert isinstance(arch_params[1], nn.Parameter)

    def test_beta_parameters(self) -> None:
        search_space = DARTSSearchSpace(edge_normalization=True)
        beta_params = search_space.beta_parameters
        assert len(beta_params) == 2
        assert isinstance(beta_params[0], nn.Parameter)
        assert isinstance(beta_params[1], nn.Parameter)

    def test_forward_pass(self) -> None:
        search_space = DARTSSearchSpace()
        x = torch.randn(2, 3, 64, 64).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

    def test_forward_pass_with_edge_normalization(self) -> None:
        search_space = DARTSSearchSpace(edge_normalization=True)
        x = torch.randn(2, 3, 64, 64).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

    def test_supernet_init(self) -> None:
        C = 32
        layers = 6
        num_classes = 11
        search_space = DARTSSearchSpace(C=C, layers=layers, num_classes=num_classes)

        search_cells = get_modules_of_type(search_space.model, DARTSSearchCell)
        assert len(search_cells) == layers

        reduction_cells = [cell for cell in search_cells if cell.reduction is True]
        assert len(reduction_cells) == 2

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_prune(self) -> None:
        search_space = DARTSSearchSpace(edge_normalization=True)
        x = torch.randn(2, 3, 64, 64).to(DEVICE)
        search_space.prune()
        arch_params = search_space.arch_parameters
        for p in arch_params:
            assert torch.count_nonzero(p) == len(p)
            assert torch.equal(
                torch.count_nonzero(p, dim=-1), torch.ones(len(p)).to(DEVICE)
            )

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

    def test_discretize_supernet(self) -> None:
        # TODO: check to have one operation on each edge of the search space
        C = 32
        layers = 6
        num_classes = 11
        search_space = DARTSSearchSpace(
            C=C, layers=layers, num_classes=num_classes, edge_normalization=True
        )

        new_model = search_space.discretize()

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = new_model(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_optim_forward_pass(self) -> None:
        search_space = DARTSSearchSpace(edge_normalization=True)
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(low=0, high=9, size=(2,)).to(DEVICE)
        arch_optim = torch.optim.Adam(
            [*search_space.arch_parameters, *search_space.beta_parameters]
        )
        arch_optim.zero_grad()
        out = search_space(x)
        loss = loss_fn(out[1], y)
        loss.backward()
        alphas_before = copy.deepcopy(search_space.arch_parameters)
        betas_before = copy.deepcopy(search_space.beta_parameters)
        arch_optim.step()
        alphas_after = search_space.arch_parameters
        betas_after = search_space.beta_parameters
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
        for beta_param_before, beta_param_after in zip(betas_before, betas_after):
            assert not torch.allclose(beta_param_before, beta_param_after)


class TestNASBench1Shot1SearchSpace(unittest.TestCase):
    def test_arch_parameters(self) -> None:
        search_space = NASBench1Shot1SearchSpace()
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 4
        assert isinstance(arch_params[0], nn.Parameter)

    def _test_forward_pass(self, model: nn.Module) -> None:
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out = model(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 64])
        assert out[1].shape == torch.Size([2, 10])

    def test_forward_pass_s1(self) -> None:
        search_space = NASBench1Shot1SearchSpace(
            num_intermediate_nodes=4,
            search_space_type="S1",
        )
        self._test_forward_pass(search_space)

    def test_forward_pass_s2(self) -> None:
        search_space = NASBench1Shot1SearchSpace(
            num_intermediate_nodes=4,
            search_space_type="S2",
        )
        self._test_forward_pass(search_space)

    def test_forward_pass_s3(self) -> None:
        search_space = NASBench1Shot1SearchSpace(
            num_intermediate_nodes=4,
            search_space_type="S3",
        )
        self._test_forward_pass(search_space)

    def test_supernet_init(self) -> None:
        layers = 7
        num_classes = 13
        search_space = NASBench1Shot1SearchSpace(
            num_intermediate_nodes=4,
            search_space_type="S1",
            layers=7,
            num_classes=num_classes,
        )

        search_cells = get_modules_of_type(search_space.model, NasBench1Shot1SearchCell)
        assert len(search_cells) == layers

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])
        assert out.shape == torch.Size([2, 64])

    def test_discretize(self) -> None:
        search_space = NASBench1Shot1SearchSpace(
            num_intermediate_nodes=4, search_space_type="S2"
        )
        search_space.discretize()

        self._test_forward_pass(search_space)

    def test_prune(self) -> None:
        search_space = NASBench1Shot1SearchSpace(
            num_intermediate_nodes=4, search_space_type="S2"
        )
        search_space.prune()
        alphas_mixed_op = search_space.arch_parameters[0]

        assert torch.count_nonzero(alphas_mixed_op) == len(alphas_mixed_op)
        assert torch.equal(
            torch.count_nonzero(alphas_mixed_op, dim=-1),
            torch.ones(len(alphas_mixed_op)).to(DEVICE),
        )

        self._test_forward_pass(search_space)

    def test_optim_forward_pass(self) -> None:
        search_space = NASBench1Shot1SearchSpace()
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(low=0, high=9, size=(2,)).to(DEVICE)
        arch_optim = torch.optim.Adam(search_space.arch_parameters)
        arch_optim.zero_grad()
        out = search_space(x)
        loss = loss_fn(out[1], y)
        loss.backward()
        alphas_before = copy.deepcopy(search_space.arch_parameters)
        arch_optim.step()
        alphas_after = search_space.arch_parameters
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)


class TestTransNASBench101SearchSpace(unittest.TestCase):
    def test_arch_parameters(self) -> None:
        search_space = TransNASBench101SearchSpace()
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 1
        assert isinstance(arch_params[0], (nn.Parameter, torch.Tensor))

    def test_beta_parameters(self) -> None:
        search_space = TransNASBench101SearchSpace()
        beta_params = search_space.arch_parameters
        assert len(beta_params) == 1
        assert isinstance(beta_params[0], nn.Parameter)

    def test_forward_pass(self) -> None:
        search_space = TransNASBench101SearchSpace()
        x = torch.randn(2, 3, 32, 32).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 10])
        assert out[1].shape == torch.Size([2, 10])

    def test_supernet_init(self) -> None:
        C = 32
        num_classes = 11
        search_space = TransNASBench101SearchSpace(C=C, num_classes=num_classes)

        search_cells = get_modules_of_type(search_space.model, TNB101SearchCell)
        assert len(search_cells) == 10

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_forward_pass_with_edge_normalization(self) -> None:
        search_space = TransNASBench101SearchSpace(edge_normalization=True)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 10])
        assert out[1].shape == torch.Size([2, 10])

    def test_discretize(self) -> None:
        search_space = TransNASBench101SearchSpace(edge_normalization=True)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        search_space.discretize()
        arch_params = search_space.arch_parameters[0]
        assert torch.count_nonzero(arch_params) == len(arch_params)
        assert torch.equal(
            torch.count_nonzero(arch_params, dim=-1),
            torch.ones(len(arch_params)).to(DEVICE),
        )
        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 10])
        assert out[1].shape == torch.Size([2, 10])

    def test_optim_forward_pass(self) -> None:
        search_space = TransNASBench101SearchSpace(edge_normalization=True)
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(low=0, high=9, size=(2,)).to(DEVICE)
        arch_optim = torch.optim.Adam(
            [*search_space.arch_parameters, *search_space.beta_parameters]
        )
        arch_optim.zero_grad()
        out = search_space(x)
        loss = loss_fn(out[1], y)
        loss.backward()
        alphas_before = copy.deepcopy(search_space.arch_parameters)
        betas_before = copy.deepcopy(search_space.beta_parameters)
        arch_optim.step()
        alphas_after = search_space.arch_parameters
        betas_after = search_space.beta_parameters
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
        for beta_param_before, beta_param_after in zip(betas_before, betas_after):
            assert not torch.allclose(beta_param_before, beta_param_after)


if __name__ == "__main__":
    unittest.main()
