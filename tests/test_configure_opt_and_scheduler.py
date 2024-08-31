import copy
import unittest
from typing import Dict, List

import torch
from torch import nn, optim

from confopt.searchspace.common import Conv2DLoRA
from confopt.utils.change_channel_size import (
    increase_bn_features,
    increase_conv_channels,
)
from confopt.utils.configure_opt_and_scheduler import (
    configure_optimizer,
)


class TestConfigureOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        # Set up a simple model
        self.model_old = nn.Sequential(
            Conv2DLoRA(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            Conv2DLoRA(16, 32, kernel_size=3),
        )

        # Set up optimizers for the old and new models
        for i, p in enumerate(self.model_old.parameters()):
            p.optimizer_id = i

        self.optimizer_old = optim.SGD(
            self.model_old.parameters(),
            lr=0.025,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )

        # Run a few steps to initialize the optimizer states
        input_data = torch.randn(10, 3, 32, 32)
        target = torch.randint(0, 10, (10,))
        criterion = nn.CrossEntropyLoss()

        output_old = self.model_old(input_data)
        loss_old = criterion(output_old.view(10, -1), target)
        loss_old.backward()
        self.optimizer_old.step()

        self.out_index: Dict[int, List[torch.Tensor | None]] = {}
        self.in_index: Dict[int, List[torch.Tensor | None]] = {}

        # Set up custom attributes for the old model
        def _increase_channels() -> None:
            for i, p in enumerate(self.model_old):
                if isinstance(p, nn.BatchNorm2d):
                    self.model_old[i], _ = increase_bn_features(p, 0.5)
                    self.out_index[i * 2] = self.out_index[i * 2 + 1] = (
                        p.weight.out_index
                    )
                elif isinstance(p, Conv2DLoRA):
                    self.model_old[i], _ = increase_conv_channels(p, 0.5)
                    self.out_index[i * 2] = p.weight.out_index
                    self.in_index[i * 2] = p.weight.in_index

        _increase_channels()
        _increase_channels()

        copied_layers = copy.deepcopy(self.model_old)

        for p, ref_p in zip(copied_layers.parameters(), self.model_old.parameters()):
            p.optimizer_id = ref_p.optimizer_id

        for layer in copied_layers:
            self.model_old.add_module(str(len(self.model_old)), layer)

        self.optimizer_new = optim.SGD(
            self.model_old.parameters(),
            lr=0.025,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )

    def test_configure_optimizer(self) -> None:
        # Function to test
        configure_optimizer(self.optimizer_old, self.optimizer_new)
        keys = list(self.optimizer_old.state.keys())

        # Test that the momentum buffers were copied correctly
        for i, p in enumerate(list(self.model_old.parameters())):
            if not hasattr(p, "optimizer_id"):
                assert torch.equal(
                    self.optimizer_new.state[p]["momentum_buffer"],
                    self.optimizer_old.state[p]["momentum_buffer"],
                )
            elif "momentum_buffer" in self.optimizer_new.state[p]:
                    
                if i in (2, 3):
                    expected_momentum_buffer = self.optimizer_old.state[keys[i]][
                        "momentum_buffer"
                    ]
                    for index in self.out_index[i]:
                        expected_momentum_buffer = torch.cat(
                            [
                                expected_momentum_buffer,
                                expected_momentum_buffer[index].clone(),
                            ],
                            dim=0,
                        )

                elif i in (0, 4):
                    expected_momentum_buffer = self.optimizer_old.state[keys[i]][
                        "momentum_buffer"
                    ]
                    for index in self.out_index[i]:
                        expected_momentum_buffer = torch.cat(
                            [
                                expected_momentum_buffer,
                                expected_momentum_buffer[index, :, :, :].clone(),
                            ],
                            dim=0,
                        )
                    for index in self.in_index[i]:
                        expected_momentum_buffer = torch.cat(
                            [
                                expected_momentum_buffer,
                                expected_momentum_buffer[:, index, :, :].clone(),
                            ],
                            dim=1,
                        )

                elif i in (1, 5):
                    expected_momentum_buffer = (
                        self.optimizer_old.state_dict()["state"]
                        .get(i, {})
                        .get("momentum_buffer")
                    )
                else:
                    opt_id = p.ref_id if hasattr(p, "ref_id") else p.optimizer_id
                    expected_momentum_buffer = (
                        self.optimizer_new.state_dict()["state"]
                        .get(opt_id, {})
                        .get("momentum_buffer")
                    )

                assert torch.equal(
                    self.optimizer_new.state[p]["momentum_buffer"],
                    expected_momentum_buffer,
                )

        print("All tests passed.")


if __name__ == "__main__":
    unittest.main()
