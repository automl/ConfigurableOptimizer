import copy
import unittest

import torch
from torch import nn

from confopt.searchspace import (
    BabyDARTSSearchSpace,
    DARTSSearchSpace,
    NASBench1Shot1SearchSpace,
    NASBench201SearchSpace,
    RobustDARTSSearchSpace,
    TransNASBench101SearchSpace,
)
from confopt.searchspace.common.base_search import SearchSpace
from confopt.searchspace.common.lora_layers import LoRALayer
from confopt.searchspace.darts.core.model_search import Cell as DARTSSearchCell
from confopt.searchspace.darts.core.operations import (
    FactorizedReduce,
    Identity,
    SepConv,
    Zero,
)
from confopt.searchspace.nb1_shot_1.core.model_search import (
    Cell as NasBench1Shot1SearchCell,
)
from confopt.searchspace.nb201.core import NAS201SearchCell
from confopt.searchspace.nb201.core.operations import ReLUConvBN, ResNetBasicblock
from confopt.searchspace.robust_darts.core.model_search import (
    Cell as RobustDARTSSearchCell,
)
from confopt.searchspace.robust_darts.core.operations import NoiseOp
from confopt.searchspace.robust_darts.core.spaces import spaces_dict
from confopt.searchspace.tnb101.core.model_search import TNB101SearchCell
from utils import get_modules_of_type  # type: ignore

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _check_pruned_alphas(pruned_alphas: list[torch.Tensor], num_keep: int) -> None:
    for alpha in pruned_alphas:
        assert torch.count_nonzero(alpha) == len(alpha) * num_keep
        assert torch.equal(
            torch.count_nonzero(alpha, dim=-1),
            num_keep * torch.ones(len(alpha)).to(DEVICE),
        )


def _check_prune_mask(masks: list[torch.Tensor], num_keep: int) -> None:
    for mask in masks:
        assert torch.count_nonzero(mask) == len(mask) * num_keep
        assert torch.equal(
            torch.count_nonzero(mask, dim=-1),
            num_keep * torch.ones(len(mask)).to(DEVICE),
        )


def _test_deactivate_lora(search_space: SearchSpace) -> None:
    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            module.activate_lora(r=4)

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 4
            assert module.conv.weight.requires_grad is False

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            module.deactivate_lora()

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 0
            assert hasattr(module, "_original_r")
            assert module._original_r == 4
            assert module.conv.weight.requires_grad is True


def _test_toggle_lora(search_space: SearchSpace) -> None:  # noqa: C901
    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            module.activate_lora(r=4)

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 4
            assert module.conv.weight.requires_grad is False

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            module.toggle_lora()

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 0
            assert hasattr(module, "_original_r")
            assert module._original_r == 4
            assert module.conv.weight.requires_grad is True

    for _, module in search_space.named_modules(remove_duplicate=True):
        if isinstance(module, LoRALayer):
            module.toggle_lora()

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 4
            assert not hasattr(module, "_original_r")
            assert module.conv.weight.requires_grad is False


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
        search_space.prune(prune_fraction=0.4)
        masks = search_space.model.mask
        _check_prune_mask(masks, num_keep=1)

        # Check for prune fraction = 0
        search_space.prune(prune_fraction=0)
        masks = search_space.model.mask
        assert masks[0].sum() == masks[0].numel()
        assert masks[1].sum() == masks[1].numel()

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
        num_ops = 5

        for num_keep in range(1, num_ops - 1):
            proxy_search_space = copy.deepcopy(search_space)
            prune_fraction = 1 - (num_keep / num_ops)
            proxy_search_space.prune(prune_fraction)

            _check_pruned_alphas(proxy_search_space.arch_parameters, num_keep)

            mask = proxy_search_space.model.mask
            _check_prune_mask([mask], num_keep)

            # Check that operations are freezed
            out = proxy_search_space(x)
            for cell in proxy_search_space.model.cells:
                if isinstance(cell, NAS201SearchCell):
                    for k in range(1, cell.max_nodes):
                        for j in range(k):
                            node_str = f"{k}<-{j}"
                            operation_block = cell.edges[node_str]
                            edge_mask = mask[cell.edge2index[node_str]]
                            zero_indices = torch.nonzero(edge_mask == 0).flatten()
                            for i in zero_indices:
                                params = [
                                    p.requires_grad
                                    for p in operation_block.ops[i].parameters()
                                ]
                                if len(params) != 0:
                                    assert not all(params)

            assert isinstance(out, tuple)
            assert len(out) == 2
            assert isinstance(out[0], torch.Tensor)
            assert isinstance(out[1], torch.Tensor)
            assert out[0].shape == torch.Size([2, 64])
            assert out[1].shape == torch.Size([2, 10])

        # Check for prune fraction = 0
        search_space.prune(prune_fraction=0)
        masks = search_space.model.mask
        assert masks.sum() == masks.numel()

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

    def test_deactivate_lora(self) -> None:
        _test_deactivate_lora(NASBench201SearchSpace())

    def test_toggle_lora(self) -> None:
        _test_toggle_lora(NASBench201SearchSpace())


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
        num_ops = 8
        for num_keep in range(1, num_ops - 1):
            proxy_search_space = copy.deepcopy(search_space)

            prune_fraction = 1 - (num_keep / num_ops)
            proxy_search_space.prune(prune_fraction)

            # Check mask shapes
            masks = proxy_search_space.model.mask
            _check_prune_mask(masks, num_keep)

            # Check pruned alphas
            pruned_alphas = proxy_search_space.arch_parameters
            _check_pruned_alphas(pruned_alphas, num_keep)

            # Check that operations are freezed
            for cell in proxy_search_space.model.cells:
                mask = masks[1] if cell.reduction else masks[0]
                for operation_block, edge_mask in zip(cell._ops, mask):
                    zero_indices = torch.nonzero(edge_mask == 0).flatten()
                    for i in zero_indices:
                        # print(i, operation_block.ops[i])
                        params = [
                            p.requires_grad for p in operation_block.ops[i].parameters()
                        ]
                        if len(params) != 0:
                            assert not all(params)

            out = proxy_search_space(x)

            assert isinstance(out, tuple)
            assert len(out) == 2
            assert isinstance(out[0], torch.Tensor)
            assert isinstance(out[1], torch.Tensor)
            assert out[0].shape == torch.Size([2, 256])
            assert out[1].shape == torch.Size([2, 10])

        # Check for prune fraction = 0
        search_space.prune(prune_fraction=0)
        masks = search_space.model.mask
        assert masks[0].sum() == masks[0].numel()
        assert masks[1].sum() == masks[1].numel()

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

    def test_deactivate_lora(self) -> None:
        _test_deactivate_lora(DARTSSearchSpace())

    def test_toggle_lora(self) -> None:
        _test_toggle_lora(DARTSSearchSpace())


class TestNASBench1Shot1SearchSpace(unittest.TestCase):
    def test_arch_parameters_s1(self) -> None:
        search_space = NASBench1Shot1SearchSpace("S1")
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 4

        for arch_param in arch_params:
            assert isinstance(arch_param, nn.Parameter)
        
        assert arch_params[0].shape == (4, 3)
        assert arch_params[1].shape == (1, 5)
        assert arch_params[2].shape == (1, 3)
        assert arch_params[3].shape == (1, 4)

    def test_arch_parameters_s2(self) -> None:
        search_space = NASBench1Shot1SearchSpace("S2")
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 5

        for arch_param in arch_params:
            assert isinstance(arch_param, nn.Parameter)
        
        assert arch_params[0].shape == (4, 3)
        assert arch_params[1].shape == (1, 5)
        assert arch_params[2].shape == (1, 2)
        assert arch_params[3].shape == (1, 3)
        assert arch_params[4].shape == (1, 4)


    def test_arch_parameters_s3(self) -> None:
        search_space = NASBench1Shot1SearchSpace("S3")
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 6

        for arch_param in arch_params:
            assert isinstance(arch_param, nn.Parameter)
        
        assert arch_params[0].shape == (5, 3)
        assert arch_params[1].shape == (1, 6)
        assert arch_params[2].shape == (1, 2)
        assert arch_params[3].shape == (1, 3)
        assert arch_params[4].shape == (1, 4)
        assert arch_params[5].shape == (1, 5)

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
        search_space = NASBench1Shot1SearchSpace("S1")
        self._test_forward_pass(search_space)

    def test_forward_pass_s2(self) -> None:
        search_space = NASBench1Shot1SearchSpace("S2")
        self._test_forward_pass(search_space)

    def test_forward_pass_s3(self) -> None:
        search_space = NASBench1Shot1SearchSpace("S3")
        self._test_forward_pass(search_space)

    def test_supernet_init(self) -> None:
        layers = 7
        num_classes = 13
        search_space = NASBench1Shot1SearchSpace(
            search_space="S1",
            layers=7,
            num_classes=num_classes,
        )

        search_cells = get_modules_of_type(search_space.model, NasBench1Shot1SearchCell)
        assert len(search_cells) == layers

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])
        assert out.shape == torch.Size([2, 64])

    def test_prune(self) -> None:
        # TODO: Update NB1Shot1 later
        search_space = NASBench1Shot1SearchSpace("S2")
        search_space.prune()
        alphas_mixed_op = search_space.arch_parameters[0]

        assert torch.count_nonzero(alphas_mixed_op) == len(alphas_mixed_op)
        assert torch.equal(
            torch.count_nonzero(alphas_mixed_op, dim=-1),
            torch.ones(len(alphas_mixed_op)).to(DEVICE),
        )

        self._test_forward_pass(search_space)

    def _test_optim_forward_pass(self, search_space: NASBench1Shot1SearchSpace) -> None:
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

    def test_optim_forward_pass_s1(self) -> None:
        search_space = NASBench1Shot1SearchSpace("S1", output_weights=True)
        self._test_optim_forward_pass(search_space)

    def test_optim_forward_pass_s2(self) -> None:
        search_space = NASBench1Shot1SearchSpace("S2", output_weights=True)
        self._test_optim_forward_pass(search_space)

    def test_optim_forward_pass_s3(self) -> None:
        search_space = NASBench1Shot1SearchSpace("S3", output_weights=True)
        self._test_optim_forward_pass(search_space)


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
        num_ops = 4
        for num_keep in range(1, num_ops - 1):
            proxy_search_space = copy.deepcopy(search_space)
            prune_fraction = 1 - (num_keep / num_ops)
            proxy_search_space.prune(prune_fraction)

            mask = proxy_search_space.model.mask
            _check_prune_mask([mask], num_keep)

            pruned_alphas = proxy_search_space.arch_parameters
            _check_pruned_alphas(pruned_alphas, num_keep)

            for cell in proxy_search_space.model.cells:
                for k in range(1, cell.max_nodes):
                    for j in range(k):
                        node_str = f"{k}<-{j}"
                        operation_block = cell.edges[node_str]
                        edge_mask = mask[cell.edge2index[node_str]]
                        zero_indices = torch.nonzero(edge_mask == 0).flatten()
                        for i in zero_indices:
                            params = [
                                p.requires_grad
                                for p in operation_block.ops[i].parameters()
                            ]
                            if len(params) != 0:
                                assert not all(params)

            out = proxy_search_space(x)

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
    def test_supernet_init(self) -> None:
        C = 32
        layers = 6
        num_classes = 11
        search_space = RobustDARTSSearchSpace(
            "s1", C=C, layers=layers, num_classes=num_classes
        )

        search_cells = get_modules_of_type(search_space.model, RobustDARTSSearchCell)
        assert len(search_cells) == layers

        reduction_cells = [cell for cell in search_cells if cell.reduction is True]
        assert len(reduction_cells) == 2

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

    def _test_arch_parameters(self, space: str, n_ops: int) -> None:
        search_space = RobustDARTSSearchSpace(space=space)
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 2

        assert isinstance(arch_params[0], nn.Parameter)
        assert isinstance(arch_params[1], nn.Parameter)
        assert arch_params[0].shape == (14, n_ops)
        assert arch_params[0].shape == (14, n_ops)

    def test_arch_parameters_s1(self) -> None:
        self._test_arch_parameters("s1", 2)

    def test_arch_parameters_s2(self) -> None:
        self._test_arch_parameters("s2", 2)

    def test_arch_parameters_s3(self) -> None:
        self._test_arch_parameters("s3", 3)

    def test_arch_parameters_s4(self) -> None:
        self._test_arch_parameters("s4", 2)

    def test_beta_parameters(self) -> None:
        search_space = RobustDARTSSearchSpace("s1")
        beta_params = search_space.beta_parameters
        assert len(beta_params) == 2
        assert isinstance(beta_params[0], nn.Parameter)
        assert isinstance(beta_params[1], nn.Parameter)

    def test_forward_pass(self) -> None:
        search_space = RobustDARTSSearchSpace("s1")
        x = torch.randn(2, 3, 64, 64).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

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
        search_space = RobustDARTSSearchSpace(space=space)
        x = torch.randn(2, 3, 64, 64).to(DEVICE)
        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

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

    def test_optim_forward_pass(self) -> None:
        search_space = RobustDARTSSearchSpace(space="s1")
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
        # TODO: Uncomment this after beta normalization has been implemented
        # for beta_param_before, beta_param_after in zip(betas_before, betas_after):
        #     assert not torch.allclose(beta_param_before, beta_param_after)


if __name__ == "__main__":
    unittest.main()
