from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from confopt.oneshot.base_component import OneShotComponent


class WeightEntanglementModule(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()


class ConvolutionalWEModule(WeightEntanglementModule):
    def __init__(self) -> None:
        super().__init__()
        self.kernel_size = None
        self.stride = None
        self.op: nn.Sequential = None

    def __post__init__(self) -> None:
        assert self.kernel_size is not None, "self.kernel_size cannot be None"
        assert self.stride is not None, "self.stride cannot be None"
        assert isinstance(
            self.op, nn.Sequential
        ), "self.op must be of type nn.Sequential"
        self.mark_entanglement_weights()

        is_module_marked = False
        for m in self.op.children():
            if hasattr(m, "can_entangle_weight") and m.can_entangle_weight is True:
                is_module_marked = True
                break

        assert (
            is_module_marked
        ), "At least one operation in self.op must be marked \
            with .can_entangle_weight = True"

    @abstractmethod
    def mark_entanglement_weights(self) -> None:
        ...


class WeightEntangler(OneShotComponent):
    def _slice_kernel_weight(
        self, weight: torch.Tensor, sub_kernel_size: int
    ) -> torch.Tensor:
        assert (
            weight.shape[-1] == weight.shape[-2]
        ), f"Only square kernels are currently supported. Found {weight.shape[-2:]}"
        assert (
            weight.shape[-1] % 2 == 1
        ), f"Kernel size cannot be even. Found {weight.shape[-1]})"
        assert (
            sub_kernel_size % 2 == 1
        ), f"Sub kernel size cannot be even. Found {sub_kernel_size}"
        start = padding = (weight.shape[-1] - sub_kernel_size) // 2
        end = start + sub_kernel_size
        weight_slice = weight[:, :, start:end, start:end]
        padded_slice = F.pad(weight_slice, (padding, padding, padding, padding))

        assert padded_slice.shape == weight.shape
        return padded_slice

    def _get_weight_entangle_ops(
        self, candidate_op: WeightEntanglementModule
    ) -> dict[int, nn.Module]:
        entangle_modules = {}
        for idx, op in enumerate(candidate_op.op.children()):
            if (
                hasattr(op, "can_entangle_weight")
                and op.can_entangle_weight is True
                and isinstance(op, nn.Conv2d)
            ):
                entangle_modules[idx] = op

        return entangle_modules

    def _get_entanglement_op_sets(
        self, modules: list[WeightEntanglementModule]
    ) -> dict[int, list[nn.Module]]:
        entanglement_set = defaultdict(list)
        assert (
            sum([1 for m in modules if not isinstance(m, WeightEntanglementModule)])
            == 0
        ), "All modules \
            must be of type WeightEntanglementModule"

        for module in modules:
            entangle_ops_in_module = self._get_weight_entangle_ops(module)

            for idx, op in entangle_ops_in_module.items():
                entanglement_set[idx].append(op)

        return dict(entanglement_set)

    def _verify_modules(self, modules: list[nn.Module]) -> None:
        assert len(modules) > 0
        module_type = type(modules[0])
        assert isinstance(modules[0], WeightEntanglementModule)
        stride = modules[0].stride
        kernel_sizes: set[int] = set()

        for m in modules:
            assert isinstance(m, WeightEntanglementModule)
            assert isinstance(
                m, module_type
            ), "All candidate modules must be of the same type with different \
                kernel sizes"
            assert m.stride == stride, "All convolutions must have the same stride"
            assert (
                m.kernel_size not in kernel_sizes
            ), "Cannot entangle more than one kernel of the same size"
            assert hasattr(m, "_alpha"), f"Module {m} missing _alpha attribute"
            assert isinstance(
                m._alpha, torch.Tensor
            ), f"_alpha attribute of {m} must be of type torch.Tensor"
            kernel_sizes.add(m.kernel_size)

    def _forward_entangled_ops(
        self, x: torch.Tensor, entangled_modules: list[nn.Module]
    ) -> torch.Tensor:
        self._verify_modules(entangled_modules)

        # Step 1, store the alphas in the operations to make
        # it easier to track after sorting the operations
        for module in entangled_modules:
            ops = self._get_weight_entangle_ops(module)
            alpha = module._alpha
            for op in ops.values():
                op._entangle_alpha = alpha

        # Step 2, sort the modules in decreasing order of kernel size
        entangled_modules = sorted(
            entangled_modules, key=lambda m: m.kernel_size, reverse=True
        )
        largest_module = entangled_modules[0]

        entanglement_ops_sets = self._get_entanglement_op_sets(entangled_modules)

        # Step 3, create the merged weight, weighted by alphas,
        # and assign it to the largest kernel. Also, back up the original weights
        # of the largest kernel to be restored later
        self.original_weights = {}
        self.new_weights = {}

        for idx, entangle_ops_set in entanglement_ops_sets.items():
            largest_op = entangle_ops_set[0]
            self.original_weights[idx] = largest_op.weight
            sub_kernel_weights = [largest_op.weight * largest_op._entangle_alpha]

            for op in entangle_ops_set[1:]:
                sub_kernel_size = op.kernel_size[0]
                sliced_weight = self._slice_kernel_weight(
                    largest_op.weight, sub_kernel_size
                )
                sub_kernel_weights.append(sliced_weight * op._entangle_alpha)

            new_weight = sum(sub_kernel_weights)
            self.new_weights[idx] = new_weight
            largest_op.weight = nn.Parameter(new_weight)

        # Step 4 - Forward pass through the module with the largest kernel
        output = largest_module(x)

        # Step 5 - Restore original weights to the largest kernel
        for idx, entangle_ops_set in entanglement_ops_sets.items():
            largest_op = entangle_ops_set[0]
            largest_op.weight = self.original_weights[idx]

        # Step 6 - delete the stored alphas
        for module in entangled_modules:
            ops = self._get_weight_entangle_ops(module)
            for op in ops.values():
                if hasattr(op, "_entangle_alpha"):
                    delattr(op, "_entangle_alpha")

        return output

    def forward(
        self, x: torch.Tensor, operations: list[nn.Module], alphas: torch.Tensor
    ) -> torch.Tensor:
        for candidate_op, alpha in zip(operations, alphas):
            candidate_op._alpha = alpha

        non_we_candidate_ops: list[nn.Module] = list(
            filter(lambda x: not isinstance(x, WeightEntanglementModule), operations)
        )

        we_candidate_ops: list[nn.Module] = list(
            filter(lambda x: isinstance(x, WeightEntanglementModule), operations)
        )

        # Compute the outputs for non-weight-entangled operations
        non_we_outputs = [op(x) * op._alpha for op in non_we_candidate_ops]

        we_candidate_types = list({type(m) for m in we_candidate_ops})
        we_candidate_ops_by_type: list[list[type]] = [[] for _ in we_candidate_types]

        for idx, op_class in enumerate(we_candidate_types):
            we_candidate_ops_by_type[idx].extend(
                list(filter(lambda m: isinstance(m, op_class), we_candidate_ops))
            )

        # Compute the outputs for weight-entangled operations
        we_outputs = [
            self._forward_entangled_ops(x, we_candidate_ops)
            for we_candidate_ops in we_candidate_ops_by_type
        ]

        for candidate_op in operations:
            delattr(candidate_op, "_alpha")

        # return the sum of all outputs
        return sum(non_we_outputs + we_outputs)
