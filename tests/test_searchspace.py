import copy
import unittest

import torch
from torch import nn

from confopt.searchspace import (
    BabyDARTSSearchSpace,
    DARTSSearchSpace,
    NASBench1Shot1SearchSpace,
    NASBench201SearchSpace,
    TransNASBench101SearchSpace,
    RobustDARTSSearchSpace,
)
from confopt.searchspace.common.lora_layers import LoRALayer
from confopt.searchspace.darts.core.model_search import Cell as DARTSSearchCell
from confopt.searchspace.darts.core.operations import (
    Identity,
    SepConv,
    Zero,
    FactorizedReduce,
)
from confopt.searchspace.nb1shot1.core.model_search import (
    Cell as NasBench1Shot1SearchCell,
)
from confopt.searchspace.nb201.core import NAS201SearchCell
from confopt.searchspace.nb201.core.operations import ReLUConvBN, ResNetBasicblock
from confopt.searchspace.tnb101.core.model_search import TNB101SearchCell
from confopt.searchspace.robust_darts.core.operations import NoiseOp
from confopt.searchspace.robust_darts.core.spaces import spaces_dict
from utils import get_modules_of_type  # type: ignore

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestBabyDARTS(unittest.TestCase):
    def test_arch_parameters(self) -> None:
        search_space = BabyDARTSSearchSpace()
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 2
        assert isinstance(arch_params[0], nn.Parameter)
        assert isinstance(arch_params[1], nn.Parameter)

    def test_beta_parameters(self) -> None:
        search_space = BabyDARTSSearchSpace(edge_normalization=True)
        beta_params = search_space.beta_parameters
        assert len(beta_params) == 2
        assert isinstance(beta_params[0], nn.Parameter)
        assert isinstance(beta_params[1], nn.Parameter)

    def test_forward_pass(self) -> None:
        search_space = BabyDARTSSearchSpace()
        x = torch.randn(2, 3, 64, 64).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 32])
        assert out[1].shape == torch.Size([2, 10])

    def test_forward_pass_with_edge_normalization(self) -> None:
        search_space = BabyDARTSSearchSpace(edge_normalization=True)
        x = torch.randn(2, 3, 64, 64).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 32])
        assert out[1].shape == torch.Size([2, 10])

    def test_supernet_init(self) -> None:
        C = 32
        num_classes = 11
        search_space = BabyDARTSSearchSpace(C=C, num_classes=num_classes)

        search_cells = get_modules_of_type(search_space.model, DARTSSearchCell)
        assert len(search_cells) == 1

        reduction_cells = [cell for cell in search_cells if cell.reduction is True]
        assert len(reduction_cells) == 1

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_prune(self) -> None:
        search_space = BabyDARTSSearchSpace(edge_normalization=True)
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
        assert out[0].shape == torch.Size([2, 32])
        assert out[1].shape == torch.Size([2, 10])

    def test_discretize_supernet(self) -> None:
        C = 32
        num_classes = 10
        search_space = BabyDARTSSearchSpace(
            C=C, num_classes=num_classes, edge_normalization=True
        )

        new_model = search_space.discretize()

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = new_model(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_optim_forward_pass(self) -> None:
        search_space = BabyDARTSSearchSpace(edge_normalization=True)
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
        alphas_before = copy.deepcopy(search_space.arch_parameters[1])
        betas_before = copy.deepcopy(search_space.beta_parameters[1])
        arch_optim.step()
        alphas_after = search_space.arch_parameters[1]
        betas_after = search_space.beta_parameters[1]
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
        for beta_param_before, beta_param_after in zip(betas_before, betas_after):
            assert not torch.allclose(beta_param_before, beta_param_after)


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

        first_cell_edge_ops = search_cells[0].edges["1<-0"].ops  # type: ignore
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

        # search_cells[0].edges["1<-0"]
        # assert type(first_cell_edge_ops) in OPS.values()
        new_model = search_space.discretize()

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

    def test_lora_parameters(self) -> None:
        search_space = NASBench201SearchSpace(edge_normalization=True)
        model_optimizer = torch.optim.Adam(search_space.model_weight_parameters())
        for _, module in search_space.named_modules(remove_duplicate=False):
            if isinstance(module, LoRALayer):
                module.activate_lora(r=4)
        opt_hyperparams = model_optimizer.defaults
        model_optimizer = type(model_optimizer)(
            search_space.model_weight_parameters(), **opt_hyperparams
        )
        model_params = search_space.model_weight_parameters()

        assert model_params == model_optimizer.param_groups[0]["params"]


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
        num_classes = 10
        search_space = DARTSSearchSpace(
            C=C, layers=layers, num_classes=num_classes, edge_normalization=True
        )

        new_model = search_space.discretize()
        new_model.drop_path_prob = 0.1  # type: ignore

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

    def test_lora_parameters(self) -> None:
        search_space = DARTSSearchSpace(edge_normalization=True)
        model_optimizer = torch.optim.Adam(search_space.model_weight_parameters())
        for _, module in search_space.named_modules(remove_duplicate=False):
            if isinstance(module, LoRALayer):
                module.activate_lora(r=4)
        opt_hyperparams = model_optimizer.defaults
        model_optimizer = type(model_optimizer)(
            search_space.model_weight_parameters(), **opt_hyperparams
        )
        model_params = search_space.model_weight_parameters()

        assert model_params == model_optimizer.param_groups[0]["params"]


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

    def test_prune(self) -> None:
        search_space = TransNASBench101SearchSpace(edge_normalization=True)
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
        assert out[0].shape == torch.Size([2, 10])
        assert out[1].shape == torch.Size([2, 10])

    def test_discretize_supernet(self) -> None:
        search_space = TransNASBench101SearchSpace()

        new_model = search_space.discretize()

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out = new_model(x)

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

    def test_lora_parameters(self) -> None:
        search_space = TransNASBench101SearchSpace(edge_normalization=True)
        model_optimizer = torch.optim.Adam(search_space.model_weight_parameters())
        for _, module in search_space.named_modules(remove_duplicate=False):
            if isinstance(module, LoRALayer):
                module.activate_lora(r=4)

        opt_hyperparams = model_optimizer.defaults
        model_optimizer = type(model_optimizer)(
            search_space.model_weight_parameters(), **opt_hyperparams
        )
        model_params = search_space.model_weight_parameters()

        assert model_params == model_optimizer.param_groups[0]["params"]


class TestRobustDARTSSearchSpace(unittest.TestCase):
    def test_init_search_spaces(self) -> None:
        s1 = RobustDARTSSearchSpace(space="s1")
        assert s1.model is not None

        s2 = RobustDARTSSearchSpace(space="s2")
        assert s1.model is not None

        s3 = RobustDARTSSearchSpace(space="s3")
        assert s1.model is not None

        s4 = RobustDARTSSearchSpace(space="s4")
        assert s1.model is not None

        with self.assertRaises(ValueError):
            RobustDARTSSearchSpace(space="s5")

        with self.assertRaises(ValueError):
            RobustDARTSSearchSpace(space="S1")

    def _test_search_space_forward(self, space: str) -> None:
        s1 = RobustDARTSSearchSpace(space=space)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out = s1(x)
        assert out.shape == (2, 10)

    def test_search_space_forward_s1(self) -> None:
        self._test_search_space_forward("s1")

    def test_search_space_forward_s2(self) -> None:
        self._test_search_space_forward("s2")

    def test_search_space_forward_s3(self) -> None:
        self._test_search_space_forward("s3")

    def test_search_space_forward_s4(self) -> None:
        self._test_search_space_forward("s4")

    def _test_search_space_candidate_ops(
        self, space: str, candidate_ops: list[str]
    ) -> None:
        search_space = RobustDARTSSearchSpace(space=space)
        cells = search_space.model.cells

        op_mapping = {
            "skip_connect": (Identity, FactorizedReduce),
            "sep_conv_3x3": SepConv,
            "none": Zero,
            "noise": NoiseOp,
        }

        for cell in cells[:3]:
            for operation_choices in cell._ops:
                ops = operation_choices.ops

                for idx, op in enumerate(ops):
                    correct_op = op_mapping[candidate_ops[idx]]
                    if isinstance(correct_op, tuple):
                        assert isinstance(op, correct_op[0]) or isinstance(
                            op, correct_op[1]
                        )
                    else:
                        assert isinstance(op, correct_op)

    def test_search_space_s1_ops(self) -> None:
        s1 = RobustDARTSSearchSpace(space="s1")
        cells = s1.model.cells

        for cell in cells:
            for operation_choices in cell._ops:
                ops = operation_choices.ops
                assert len(ops) == 2
                assert isinstance(ops[0], nn.Module)

    def test_search_space_s2_ops(self) -> None:
        self._test_search_space_candidate_ops("s2", ["skip_connect", "sep_conv_3x3"])

    def test_search_space_s3_ops(self) -> None:
        self._test_search_space_candidate_ops(
            "s3", ["none", "skip_connect", "sep_conv_3x3"]
        )

    def test_search_space_s4_ops(self) -> None:
        self._test_search_space_candidate_ops("s4", ["noise", "sep_conv_3x3"])


if __name__ == "__main__":
    unittest.main()
