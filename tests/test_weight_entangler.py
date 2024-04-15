from __future__ import annotations

import unittest
import pytest
import torch
import numpy as np

from confopt.oneshot.weightentangler import WeightEntangler
from confopt.searchspace.darts.core.operations import (
    DilConv,
    Pooling,
    ReLUConvBN,
    SepConv,
)


class TestWeightEntangler(unittest.TestCase):
    def test_slice_kernel_weight(self) -> None:
        entangler = WeightEntangler()
        weight = torch.ones((32, 32, 7, 7))

        weight_slice = entangler._slice_kernel_weight(weight, 5)
        assert (weight_slice[:, :, 1:6, 1:6] == weight[:, :, 1:6, 1:6]).all()
        assert (weight_slice[:, :, :1, :] == 0).all()
        assert (weight_slice[:, :, 6:, :] == 0).all()
        assert (weight_slice[:, :, :, :1] == 0).all()
        assert (weight_slice[:, :, :, 6:] == 0).all()

        weight_slice = entangler._slice_kernel_weight(weight, 3)
        assert (weight_slice[:, :, 2:5, 2:5] == weight[:, :, 2:5, 2:5]).all()
        assert (weight_slice[:, :, :2, :] == 0).all()
        assert (weight_slice[:, :, 5:, :] == 0).all()
        assert (weight_slice[:, :, :, :2] == 0).all()
        assert (weight_slice[:, :, :, 5:] == 0).all()

    def test_slice_kernel_weight_illegal_inputs(self) -> None:
        entangler = WeightEntangler()

        weight = torch.ones((32, 32, 7, 7))
        with pytest.raises(AssertionError):
            entangler._slice_kernel_weight(weight, 4)

        weight = torch.ones(32, 32, 6, 6)
        with pytest.raises(AssertionError):
            entangler._slice_kernel_weight(weight, 3)

        weight = torch.ones(32, 32, 5, 7)
        with pytest.raises(AssertionError):
            entangler._slice_kernel_weight(weight, 3)

    def test_get_weight_entangle_ops(self) -> None:
        entangler = WeightEntangler()
        sep_conv7x7 = SepConv(32, 32, 7, 1, 3)

        entangled_modules = entangler._get_weight_entangle_ops(sep_conv7x7)
        assert len(entangled_modules) == 2
        assert set([1, 5]) == set(entangled_modules.keys())

        modules = list(entangled_modules.values())
        assert modules[0] == sep_conv7x7.op[1].conv
        assert modules[1] == sep_conv7x7.op[5].conv

        dil_conv5x5 = DilConv(32, 32, 5, 1, 2, 1)

        entangled_modules = entangler._get_weight_entangle_ops(dil_conv5x5)
        assert len(entangled_modules) == 1
        assert 1 == list(entangled_modules.keys())[0]

        modules = list(entangled_modules.values())
        assert modules[0] == dil_conv5x5.op[1].conv

        pooling = Pooling(32, 1, "avg")
        entangled_modules = entangler._get_weight_entangle_ops(pooling)
        assert entangled_modules == {}

        relu_conv_bn = ReLUConvBN(32, 32, 7, 1, 3)
        entangled_modules = entangler._get_weight_entangle_ops(relu_conv_bn)
        assert entangled_modules == {}

    def test_get_entanglement_op_sets(self) -> None:
        entangler = WeightEntangler()

        sep_convs = [SepConv(32, 32, k, 1, d) for k, d in zip([3, 5, 7], [1, 2, 3])]
        entangle_modules = entangler._get_entanglement_op_sets(sep_convs)

        assert len(entangle_modules) == 2
        assert set([1, 5]) == set(entangle_modules.keys())
        assert len(entangle_modules[1]) == 3
        assert len(entangle_modules[5]) == 3

        assert entangle_modules[1] == [
            sep_convs[0].op[1].conv,
            sep_convs[1].op[1].conv,
            sep_convs[2].op[1].conv,
        ]
        assert entangle_modules[5] == [
            sep_convs[0].op[5].conv,
            sep_convs[1].op[5].conv,
            sep_convs[2].op[5].conv,
        ]

        dil_convs = [DilConv(32, 32, k, 1, d, 1) for k, d in zip([3, 5], [1, 2])]
        entangle_modules = entangler._get_entanglement_op_sets(dil_convs)

        assert len(entangle_modules) == 1
        assert set([1]) == set(entangle_modules.keys())
        assert len(entangle_modules[1]) == 2

        assert entangle_modules[1] == [dil_convs[0].op[1].conv, dil_convs[1].op[1].conv]

        with pytest.raises(AssertionError):
            entangler._get_entanglement_op_sets(dil_convs + [Pooling(32, 1, "max")])

    def test_forward_entangled_ops(self) -> None:
        entangler = WeightEntangler()
        x = torch.ones((1, 3, 7, 7))
        alphas = torch.Tensor([0.1, 0.2, 0.7])

        sep_convs = [SepConv(3, 1, k, 1, d) for k, d in zip([3, 5, 7], [1, 2, 3])]
        for sep_conv, alpha in zip(sep_convs, alphas):

            for op in sep_conv.op:
                if hasattr(op, "weight"):
                    op.weight.data = torch.ones_like(op.weight)
                if hasattr(op, "bias") and op.bias is not None:
                    op.bias.data = torch.ones_like(op.bias)
            sep_conv._alpha = alpha

        out = entangler._forward_entangled_ops(x, sep_convs)

        expected_new_weights = torch.ones_like(sep_conv.op[1].weight)
        expected_new_weights *= alphas[2]
        expected_new_weights[:, :, 1:6, 1:6] += (
            sep_conv.op[1].weight[:, :, 1:6, 1:6] * alphas[1]
        )
        expected_new_weights[:, :, 2:5, 2:5] += (
            sep_conv.op[1].weight[:, :, 2:5, 2:5] * alphas[0]
        )
        # print(expected_new_weights[0, 0])
        # print(entangler.new_weights[1][0, 0])

        assert len(entangler.original_weights) == 2
        assert set([1, 5]) == set(entangler.original_weights.keys())
        assert (entangler.original_weights[1] == sep_convs[2].op[1].weight).all()
        assert (entangler.original_weights[5] == sep_convs[2].op[5].weight).all()
        assert (entangler.new_weights[1] == expected_new_weights).all()

    def test_forward_entangled_ops_with_lora(self) -> None:
        entangler = WeightEntangler()
        x = torch.ones((1, 3, 7, 7))
        alphas = torch.Tensor([0.1, 0.2, 0.7])
        sep_convs = [SepConv(3, 1, k, 1, d) for k, d in zip([3, 5, 7], [1, 2, 3])]

        for sep_conv, alpha in zip(sep_convs, alphas):
            for op in sep_conv.op:
                if hasattr(op, "weight"):
                    op.weight.data = torch.ones_like(op.weight)
                if hasattr(op, "bias") and op.bias is not None:
                    op.bias.data = torch.ones_like(op.bias)
            sep_conv._alpha = alpha

        out_normal = entangler._forward_entangled_ops(x, sep_convs)
        out_normal2 = entangler._forward_entangled_ops(x, sep_convs)
        assert np.allclose(out_normal.detach().numpy(), out_normal2.detach().numpy())

        for sep_conv, alpha in zip(sep_convs, alphas):
            sep_conv.op[1].activate_lora(r=1)
            sep_conv.op[1].lora_A.data = torch.ones_like(sep_conv.op[1].lora_A)
            sep_conv.op[1].lora_B.data = torch.ones_like(sep_conv.op[1].lora_B)

        out_we = entangler._forward_entangled_ops(x, sep_convs)
        assert not np.allclose(out_normal.detach().numpy(), out_we.detach().numpy())

        expected_new_weights = torch.ones_like(sep_conv.op[1].weight)
        expected_new_weights *= alphas[2]
        expected_new_weights[:, :, 1:6, 1:6] += (
            sep_conv.op[1].weight[:, :, 1:6, 1:6] * alphas[1]
        )
        expected_new_weights[:, :, 2:5, 2:5] += (
            sep_conv.op[1].weight[:, :, 2:5, 2:5] * alphas[0]
        )

        assert len(entangler.original_weights) == 2
        assert set([1, 5]) == set(entangler.original_weights.keys())
        assert (entangler.original_weights[1] == sep_convs[2].op[1].weight).all()
        assert (entangler.original_weights[5] == sep_convs[2].op[5].weight).all()
        assert (entangler.new_weights[1] == expected_new_weights).all()


    def test_forward_entangled_ops_illegal_inputs(self) -> None:
        entangler = WeightEntangler()
        x = torch.ones((2, 32, 10, 10))

        sep_convs = [SepConv(32, 32, k, 1, d) for k, d in zip([3, 5, 7], [1, 2, 3])]
        dil_convs = [DilConv(32, 32, k, 1, d, 1) for k, d in zip([3, 5], [1, 2])]

        # Forward pass without the alpha values in the modules
        with pytest.raises(AssertionError):
            entangler._forward_entangled_ops(x, sep_convs)

        alphas = torch.Tensor([0.1, 0.2, 0.7])
        for sep_conv, dil_conv, alpha in zip(sep_convs, dil_convs, alphas):
            sep_conv._alpha = alpha
            dil_conv._alpha = alpha

        # Forward convs of different kinds
        mixed_convs = sep_convs[:1] + dil_convs[1:]
        with pytest.raises(AssertionError):
            entangler._forward_entangled_ops(x, mixed_convs)

        # Forward non WeightEntanglementModule operations
        non_we_modules = [Pooling(32, 1, "avg"), Pooling(32, 1, "max")]
        with pytest.raises(AssertionError):
            entangler._forward_entangled_ops(x, non_we_modules)

        # Forward through modules of the same kernel size
        same_kernel_size_convs = [sep_convs[0], sep_convs[0]]
        with pytest.raises(AssertionError):
            entangler._forward_entangled_ops(x, same_kernel_size_convs)

        # Forward through modules where alpha is not a torch Tensor
        alphas = [0.1, 0.2, 0.7]
        for sep_conv, alpha in zip(sep_convs, alphas):
            sep_conv._alpha = alpha

        with pytest.raises(AssertionError):
            entangler._forward_entangled_ops(x, sep_convs)


if __name__ == "__main__":
    unittest.main()
