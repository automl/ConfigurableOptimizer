import unittest

import torch
from torch import nn

from confopt.oneshot.partial_connector import PartialConnector
from confopt.searchspace.nb201.core.operations import OPS

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestPartialConnector(unittest.TestCase):
    def test_forward_pass(self) -> None:
        x = torch.randn(2, 16, 8, 8).to(DEVICE)
        arch_parameters = nn.Parameter(1e-3 * torch.randn(1, 2)).to(DEVICE)
        alphas = nn.functional.softmax(arch_parameters, dim=-1).to(DEVICE)
        ops = nn.ModuleList(
            [
                OPS[op_name](4, 4, 1, False, True).to(DEVICE)
                for op_name in ["nor_conv_3x3", "avg_pool_3x3"]
            ]
        )
        partial_connector = PartialConnector(k=4)
        out = partial_connector(x, alphas[0], ops=ops)
        print("hello")

        assert out.shape == x.shape
        assert isinstance(out, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
