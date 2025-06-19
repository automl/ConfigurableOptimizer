import unittest

import torch
from torch import nn
import pytest

from confopt.searchspace import DARTSSearchSpace
from confopt.searchspace.common.mixop import OperationChoices
from confopt.searchspace.darts.core.model_search import Cell
from confopt.searchspace.darts.core.operations import (
    DilConv,
    FactorizedReduce,
    Identity,
    Pooling,
    ReLUConvBN,
    SepConv,
    Zero,
)
from confopt.utils import get_pos_new_cell_darts

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestDARTSSearchSpace(unittest.TestCase):
    def _compare_cells(self, cell1: Cell, cell2: Cell) -> bool:  # noqa: PLR0911
        # Check basic attributes
        if cell1._steps != cell2._steps or cell1._multiplier != cell2._multiplier:
            return False

        if (
            cell1.C != cell2.C
            or cell1.C_prev != cell2.C_prev
            or cell1.C_prev_prev != cell2.C_prev_prev
        ):
            return False

        if cell1.reduction != cell2.reduction:
            return False

        # Check preprocess0 operation
        if not self._compare_conv_layers(cell1.preprocess0, cell2.preprocess0):
            return False

        # Check preprocess1 operation
        if not self._compare_conv_layers(cell1.preprocess1, cell2.preprocess1):
            return False

        # Compare _ops ModuleList
        if len(cell1._ops) != len(cell2._ops):
            return False

        for op1, op2 in zip(cell1._ops, cell2._ops):
            if not self._compare_ops(op1, op2):
                return False

        # If all checks pass, the cells are considered equal
        return True

    def _compare_conv_layers(self, conv1: nn.Module, conv2: nn.Module) -> bool:
        if isinstance(conv1, ReLUConvBN) and isinstance(conv2, ReLUConvBN):
            return self._compare_reluconvbn(conv1, conv2)
        if isinstance(conv2, FactorizedReduce) and isinstance(conv2, FactorizedReduce):
            return self._compare_factorizedreduce(conv1, conv2)
        return False

    def _compare_reluconvbn(self, op1: ReLUConvBN, op2: ReLUConvBN) -> bool:
        return (
            op1.C_in == op2.C_in
            and op1.C_out == op2.C_out
            and op1.op[1].conv.in_channels == op2.op[1].conv.in_channels
            and op1.op[1].conv.out_channels == op2.op[1].conv.out_channels
            and op1.op[1].conv.kernel_size == op2.op[1].conv.kernel_size
            and op1.op[1].conv.stride == op2.op[1].conv.stride
            and op1.op[1].conv.padding == op2.op[1].conv.padding
            and op1.op[2].num_features == op2.op[2].num_features
            and op1.op[2].bias == op2.op[2].bias
            and op1.op[2].affine == op2.op[2].affine
        )

    def _compare_pooling(self, op1: Pooling, op2: Pooling) -> bool:
        return (
            isinstance(op1.op[0], type(op2.op[0]))
            and op1.op[0].stride == op2.op[0].stride
            and op1.op[1].num_features == op2.op[1].num_features
            and op1.op[1].bias == op2.op[1].bias
            and op1.op[1].affine == op2.op[1].affine
        )

    def _compare_dilconv(self, op1: DilConv, op2: DilConv) -> bool:
        return (
            op1.kernel_size == op2.kernel_size
            and op1.stride == op2.stride
            and op1.op[1].conv.in_channels == op2.op[1].conv.in_channels
            and op1.op[1].conv.out_channels == op2.op[1].conv.out_channels
            and op1.op[1].conv.kernel_size == op2.op[1].conv.kernel_size
            and op1.op[1].conv.stride == op2.op[1].conv.stride
            and op1.op[1].conv.padding == op2.op[1].conv.padding
            and op1.op[1].conv.groups == op2.op[1].conv.groups
            and op1.op[1].conv.dilation == op2.op[1].conv.dilation
            and op1.op[2].conv.in_channels == op2.op[2].conv.in_channels
            and op1.op[2].conv.out_channels == op2.op[2].conv.out_channels
            and op1.op[3].num_features == op2.op[3].num_features
            and op1.op[3].bias == op2.op[3].bias
            and op1.op[3].affine == op2.op[3].affine
        )

    def _compare_sepconv(self, op1: SepConv, op2: SepConv) -> bool:
        return (
            op1.kernel_size == op2.kernel_size
            and op1.stride == op2.stride
            and op1.op[1].conv.in_channels == op2.op[1].conv.in_channels
            and op1.op[1].conv.out_channels == op2.op[1].conv.out_channels
            and op1.op[1].conv.stride == op2.op[1].conv.stride
            and op1.op[1].conv.padding == op2.op[1].conv.padding
            and op1.op[1].conv.groups == op2.op[1].conv.groups
            and op1.op[2].conv.in_channels == op2.op[2].conv.in_channels
            and op1.op[2].conv.out_channels == op2.op[2].conv.out_channels
            and op1.op[3].num_features == op2.op[3].num_features
            and op1.op[3].bias == op2.op[3].bias
            and op1.op[3].affine == op2.op[3].affine
            and op1.op[5].in_channels == op2.op[5].in_channels
            and op1.op[5].conv.groups == op2.op[5].conv.groups
            and op1.op[6].conv.in_channels == op2.op[6].conv.in_channels
            and op1.op[6].conv.out_channels == op2.op[6].conv.out_channels
            and op1.op[7].num_features == op2.op[7].num_features
            and op1.op[7].bias == op2.op[7].bias
            and op1.op[7].affine == op2.op[7].affine
        )

    def _compare_identity(self, op1: Identity, op2: Identity) -> bool:  # noqa: ARG002
        # Identity has no internal parameters to compare;
        # they are always considered equal.
        return True

    def _compare_zero(self, op1: Zero, op2: Zero) -> bool:
        return op1.stride == op2.stride

    def _compare_factorizedreduce(
        self, op1: FactorizedReduce, op2: FactorizedReduce
    ) -> bool:
        return (
            op1.C_in == op2.C_in
            and op1.C_out == op2.C_out
            and op1.conv_1.conv.in_channels == op2.conv_1.conv.in_channels
            and op1.conv_1.conv.out_channels == op2.conv_1.conv.out_channels
            and op1.conv_1.conv.stride == op2.conv_1.conv.stride
            and op1.conv_2.conv.in_channels == op2.conv_2.conv.in_channels
            and op2.conv_2.conv.out_channels == op2.conv_2.conv.out_channels
            and op1.conv_2.conv.stride == op2.conv_2.conv.stride
        )

    def _compare_operations(self, op1: nn.Module, op2: nn.Module) -> bool:
        compare_ops = False
        if isinstance(op1, ReLUConvBN) and isinstance(op2, ReLUConvBN):
            compare_ops = self._compare_reluconvbn(op1, op2)
        elif isinstance(op1, Pooling) and isinstance(op2, Pooling):
            compare_ops = self._compare_pooling(op1, op2)
        elif isinstance(op1, DilConv) and isinstance(op2, DilConv):
            compare_ops = self._compare_dilconv(op1, op2)
        elif isinstance(op1, SepConv) and isinstance(op2, SepConv):
            compare_ops = self._compare_sepconv(op1, op2)
        elif isinstance(op1, Identity) and isinstance(op2, Identity):
            compare_ops = True
        elif isinstance(op1, Zero) and isinstance(op2, Zero):
            compare_ops = self._compare_zero(op1, op2)
        elif isinstance(op1, FactorizedReduce) and isinstance(op2, FactorizedReduce):
            compare_ops = self._compare_factorizedreduce(op1, op2)
        return compare_ops

    def _compare_ops(self, op1: nn.Module, op2: nn.Module) -> bool:
        if isinstance(op1, OperationChoices) and isinstance(op2, OperationChoices):
            for sub_op1, sub_op2 in zip(op1.ops, op2.ops):
                if not self._compare_operations(sub_op1, sub_op2):
                    return False
            return True
        return False

    @pytest.mark.experimental()  # type: ignore
    def test_create_new_cell(self) -> None:
        for i in range(2, 15):
            search_space = DARTSSearchSpace(layers=i, C=16)
            pos = get_pos_new_cell_darts(i)
            new_cell = search_space.model.create_new_cell(pos)
            search_space_2 = DARTSSearchSpace(layers=i + 1, C=16)
            assert self._compare_cells(new_cell, search_space_2.model.cells[pos])

    @pytest.mark.experimental() # type: ignore
    def test_insert_new_cells(self) -> None:
        for i in range(2, 10):
            search_space = DARTSSearchSpace(layers=i, C=16)
            pos = get_pos_new_cell_darts(i)
            search_space.model.insert_new_cells(1)
            search_space_2 = DARTSSearchSpace(layers=i + 1, C=16)
            assert self._compare_cells(
                search_space.model.cells[pos], search_space_2.model.cells[pos]
            )


if __name__ == "__main__":
    unittest.main()
