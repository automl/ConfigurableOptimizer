from __future__ import annotations

import unittest
import torch

from confopt.oneshot.archsampler import DARTSSampler
from confopt.searchspace import SearchSpace, NASBench201SearchSpace, DARTSSearchSpace
from confopt.searchspace.common.mixop import DynamicAttentionNetwork
from confopt.train import SearchSpaceHandler

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestDynamicAttentionExploration(unittest.TestCase):
    def test_nb201_init_auxiliary(self) -> None:
        # NB201 does not have introduce any new parameters
        searchspace = NASBench201SearchSpace()
        dan = DynamicAttentionNetwork(C=16, num_ops=5, attention_weight=1).to(DEVICE)
        self._test_with_optimizer(searchspace, dan)
        self._test_vanilla_forward(searchspace, dan)

    def test_darts_init_auxiliary(self) -> None:
        searchspace = DARTSSearchSpace()
        dan = DynamicAttentionNetwork(C=16, num_ops=8, attention_weight=1)
        self._test_with_optimizer(searchspace, dan)
        self._test_vanilla_forward(searchspace, dan)

    def _test_with_optimizer(
        self, searchspace: SearchSpace, dan: DynamicAttentionNetwork
    ) -> None:
        loss_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(searchspace.model_weight_parameters(), lr=1e-3)
        searchspace_handler = SearchSpaceHandler(
            sampler=DARTSSampler(searchspace.arch_parameters),
            dynamic_explorer=dan,
        )
        searchspace_handler.adapt_search_space(searchspace)

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(low=0, high=9, size=(2,)).to(DEVICE)

        _, logits = searchspace(x)
        loss = loss_criterion(logits, y)
        loss.backward()

        params_before_step = []
        for name, param in searchspace.named_parameters():
            if param.requires_grad is True and "dan" in name:
                params_before_step.append(param.detach().clone())
                param.grad += torch.ones_like(param.grad) * 10

        optimizer.step()

        params_after_step = []
        for name, param in searchspace.named_parameters():
            if param.requires_grad is True and "dan" in name:
                params_after_step.append(param.detach().clone())

        for param_before, param_after in zip(params_before_step, params_after_step):
            assert not torch.allclose(param_before, param_after)

    def _test_vanilla_forward(
        self, searchspace: SearchSpace, dan: DynamicAttentionNetwork
    ) -> None:

        searchspace_handler = SearchSpaceHandler(
            sampler=DARTSSampler(searchspace.arch_parameters),
            dynamic_explorer=dan,
        )
        searchspace_handler.adapt_search_space(searchspace)

        x = torch.randn(2, 3, 32, 32).to(DEVICE)

        _, logits = searchspace(x)

        assert logits.shape == torch.Size([2, 10])

    def test_dan_forward(self) -> None:
        dan = DynamicAttentionNetwork(C=16, num_ops=7, attention_weight=1).to(DEVICE)
        x = torch.randn(2, 16, 32, 32).to(DEVICE)

        dan_out = dan(x)
        assert dan_out.shape == torch.Size([7])


if __name__ == "__main__":
    unittest.main()
