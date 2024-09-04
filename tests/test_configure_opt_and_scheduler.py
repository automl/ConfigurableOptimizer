import copy
import unittest

import torch
from torch import nn, optim

from confopt.searchspace import DARTSSearchSpace
from confopt.searchspace.common import Conv2DLoRA
from confopt.utils.change_channel_size import (
    increase_bn_features,
    increase_conv_channels,
)
from confopt.utils.configure_opt_and_scheduler import (
    configure_optimizer,
)


def setup_optimizer_and_model(
) -> tuple[nn.Module, optim.Optimizer, optim.Optimizer, dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]]]:
    # Set up a simple model
    model_old = nn.Sequential(
        Conv2DLoRA(3, 16, kernel_size=3),
        nn.BatchNorm2d(16),
        Conv2DLoRA(16, 32, kernel_size=3),
    )

    # Set up optimizers for the old and new models
    for i, p in enumerate(model_old.parameters()):
        p.optimizer_id = i

    optimizer_old = optim.SGD(
        model_old.parameters(),
        lr=0.025,
        momentum=0.9,
        nesterov=False,
        weight_decay=1e-4,
    )

    # Run a few steps to initialize the optimizer states
    input_data = torch.randn(10, 3, 32, 32)
    target = torch.randint(0, 10, (10,))
    criterion = nn.CrossEntropyLoss()

    output_old = model_old(input_data)
    loss_old = criterion(output_old.view(10, -1), target)
    loss_old.backward()
    optimizer_old.step()

    out_index = {}
    in_index = {}

    # Set up custom attributes for the old model
    def _increase_channels() -> None:
        for i, p in enumerate(model_old):
            if isinstance(p, nn.BatchNorm2d):
                model_old[i], _ = increase_bn_features(p, 0.5)
                out_index[i * 2] = out_index[i * 2 + 1] = p.weight.out_index
            elif isinstance(p, Conv2DLoRA):
                model_old[i], _ = increase_conv_channels(p, 0.5)
                out_index[i * 2] = p.weight.out_index
                in_index[i * 2] = p.weight.in_index

    _increase_channels()
    _increase_channels()

    copied_layers = copy.deepcopy(model_old)

    for p, ref_p in zip(copied_layers.parameters(), model_old.parameters()):
        p.optimizer_id = ref_p.optimizer_id

    for layer in copied_layers:
        model_old.add_module(str(len(model_old)), layer)

    optimizer_new = optim.SGD(
        model_old.parameters(),
        lr=0.025,
        momentum=0.9,
        nesterov=False,
        weight_decay=1e-4,
    )

    return model_old, optimizer_old, optimizer_new, out_index, in_index

class TestConfigureOptimizer(unittest.TestCase):

    def test_configure_optimizer_basic_model(self) -> None:
        # Use the helper function to set up the models and optimizers
        model_old, optimizer_old, optimizer_new, out_index, in_index = setup_optimizer_and_model()

        # Function to test
        configure_optimizer(optimizer_old, optimizer_new)
        keys = list(optimizer_old.state.keys())

        assert self.check_optimizer_state(model_old, optimizer_old, optimizer_new, out_index, in_index, keys)
    
    def check_optimizer_state(
            self, 
            model_old: nn.Module, 
            optimizer_new: optim.Optimizer, 
            optimizer_old: optim.Optimizer, 
            out_index: dict[int, list[torch.Tensor]],
            in_index: dict[int, list[torch.Tensor]],
            keys: list[torch.Tensor],
        ) -> bool:
        # Test that the momentum buffers were copied correctly
        for i, p in enumerate(list(model_old.parameters())):
            if not hasattr(p, "optimizer_id"):
                assert torch.equal(
                    optimizer_new.state[p]["momentum_buffer"],
                    optimizer_old.state[p]["momentum_buffer"],
                )
            elif "momentum_buffer" in optimizer_new.state[p]:

                if i in (2, 3):
                    expected_momentum_buffer = optimizer_old.state[keys[i]][
                        "momentum_buffer"
                    ]
                    for index in out_index[i]:
                        expected_momentum_buffer = torch.cat(
                            [
                                expected_momentum_buffer,
                                expected_momentum_buffer[index].clone(),
                            ],
                            dim=0,
                        )

                elif i in (0, 4):
                    expected_momentum_buffer = optimizer_old.state[keys[i]][
                        "momentum_buffer"
                    ]
                    for index in out_index[i]:
                        expected_momentum_buffer = torch.cat(
                            [
                                expected_momentum_buffer,
                                expected_momentum_buffer[index, :, :, :].clone(),
                            ],
                            dim=0,
                        )
                    for index in in_index[i]:
                        expected_momentum_buffer = torch.cat(
                            [
                                expected_momentum_buffer,
                                expected_momentum_buffer[:, index, :, :].clone(),
                            ],
                            dim=1,
                        )

                elif i in (1, 5):
                    expected_momentum_buffer = (
                        optimizer_old.state_dict()["state"]
                        .get(i, {})
                        .get("momentum_buffer")
                    )
                else:
                    opt_id = p.optimizer_id
                    assert opt_id is not None
                    expected_momentum_buffer = (
                        optimizer_new.state_dict()["state"]
                        .get(opt_id, {})
                        .get("momentum_buffer")
                    )

                assert torch.equal(
                    optimizer_new.state[p]["momentum_buffer"],
                    expected_momentum_buffer,
                )
        return True

    def test_configure_optimizer_darts_multiple_cells_at_once(self) -> None:
        search_space = DARTSSearchSpace(layers=2, C=16)
        optimizer_old = optim.SGD(
            search_space.parameters(),
            lr=0.025,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        for i, param in enumerate(search_space.parameters()):
            param.optimizer_id = i
        

        search_space.model.insert_new_cells(5)
        optimizer_new = optim.SGD(
            search_space.parameters(),
            lr=0.025,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        configure_optimizer(optimizer_old, optimizer_new)
        in_index: dict[int, list[torch.Tensor]] = {}
        out_index: dict[int, list[torch.Tensor]] = {}
        
        for i, p in enumerate(search_space.model.cells[-5:].parameters()):
            if hasattr(p, "out_index"):
                out_index[i] = p.out_index
            if hasattr(p, "in_index"):
                in_index[i] = p.in_index


        assert self.check_optimizer_state(search_space.model, optimizer_old, optimizer_new, in_index, out_index, list(optimizer_old.state.keys()))

    def test_configure_optimizer_darts_multiple_cells_one_by_one(self) -> None:
        search_space = DARTSSearchSpace(layers=2, C=16)
        optimizer_old = optim.SGD(
            search_space.parameters(),
            lr=0.025,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        for i, param in enumerate(search_space.parameters()):
            param.optimizer_id = i
        
        for j in range(3):

            search_space.model.insert_new_cells(1)
            optimizer_new = optim.SGD(
                search_space.parameters(),
                lr=0.025,
                momentum=0.9,
                nesterov=False,
                weight_decay=1e-4,
            )
            configure_optimizer(optimizer_old, optimizer_new)
            in_index: dict[int, list[torch.Tensor]] = {}
            out_index: dict[int, list[torch.Tensor]] = {}
            
            for i, p in enumerate(search_space.model.cells[-1].parameters()):
                if hasattr(p, "out_index"):
                    out_index[i] = p.out_index
                if hasattr(p, "in_index"):
                    in_index[i] = p.in_index


            assert self.check_optimizer_state(search_space.model, optimizer_old, optimizer_new, in_index, out_index, list(optimizer_old.state.keys()))
            print(f"Test {j} passed.")
            optimizer_old = optimizer_new

    def test_configure_optimizer_darts_add_single_cell(self) -> None:
        search_space = DARTSSearchSpace(layers=2, C=16)
        optimizer_old = optim.SGD(
            search_space.parameters(),
            lr=0.025,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        for i, param in enumerate(search_space.parameters()):
            param.optimizer_id = i
        

        search_space.model.insert_new_cells(1)
        optimizer_new = optim.SGD(
            search_space.parameters(),
            lr=0.025,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        configure_optimizer(optimizer_old, optimizer_new)
        in_index: dict[int, list[torch.Tensor]] = {}
        out_index: dict[int, list[torch.Tensor]] = {}
        
        for i, p in enumerate(search_space.model.cells[-1].parameters()):
            if hasattr(p, "out_index"):
                out_index[i] = p.out_index
            if hasattr(p, "in_index"):
                in_index[i] = p.in_index


        assert self.check_optimizer_state(search_space.model, optimizer_old, optimizer_new, in_index, out_index, list(optimizer_old.state.keys()))
        optimizer_old = optimizer_new


if __name__ == "__main__":
    unittest.main()