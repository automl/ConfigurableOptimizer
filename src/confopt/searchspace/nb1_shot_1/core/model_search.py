from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from confopt.searchspace.common import OperationChoices
from confopt.searchspace.darts.core.operations import ReLUConvBN
from confopt.utils import normalize_params, prune, set_ops_to_prune

from .operations import OPS, ConvBnRelu
from .search_spaces.genotypes import PRIMITIVES, NASBench1Shot1ConfoptGenotype
from .search_spaces.search_space import NB1Shot1Space
from .search_spaces.search_space_1 import NB1Shot1Space1
from .search_spaces.search_space_2 import NB1Shot1Space2
from .search_spaces.search_space_3 import NB1Shot1Space3
from .search_spaces.utils import CONV1X1, INPUT, OUTPUT


class MixedOp(nn.Module):
    def __init__(self, C: int, stride: int) -> None:
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            """
            Not used in NASBench
            if 'pool' in primitive:
              op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            """
            self._ops.append(op)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class ChoiceBlock(nn.Module):
    """Adapted to match Figure 3 in:
    Bender, Gabriel, et al. "Understanding and simplifying one-shot architecture
    search."
    International Conference on Machine Learning. 2018.
    """

    def __init__(self, C_in: int) -> None:
        super().__init__()
        # Pre-processing 1x1 convolution at the beginning of each choice block.
        ops = MixedOp(C_in, stride=1)._ops
        self.mixed_op = OperationChoices(ops)

    def forward(
        self,
        inputs: list[torch.Tensor],
        input_weights: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if input_weights is not None:
            # Weigh the input to the choice block
            inputs = [w * t for w, t in zip(input_weights.squeeze(0), inputs)]

        # Sum input to choice block
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L298
        input_to_mixed_op = sum(inputs)

        # Apply Mixed Op
        output = self.mixed_op(input_to_mixed_op, alphas=weights)
        return output


class Cell(nn.Module):
    def __init__(
        self, steps: int, C_prev: int, C: int, layer: int, search_space: NB1Shot1Space
    ) -> None:
        super().__init__()
        # All cells are normal cells in NASBench case.
        self._steps = steps

        self._choice_blocks = nn.ModuleList()
        self._bns = nn.ModuleList()
        self.search_space = search_space

        self._input_projections = nn.ModuleList()
        # Number of input channels is dependent on whether it is the first layer or not.
        # Any subsequent layer has C_in * (steps + 1) input channels because the output
        # is a concatenation of the input tensor and all choice block outputs
        C_in = C_prev if layer == 0 else C_prev * steps

        # Create the choice block and the input
        for _i in range(self._steps):
            choice_block = ChoiceBlock(C_in=C)
            self._choice_blocks.append(choice_block)
            self._input_projections.append(
                ConvBnRelu(C_in=C_in, C_out=C, kernel_size=1, stride=1, padding=0)
            )

        # Add one more input preprocessing for edge from input to output of the cell
        self._input_projections.append(
            ConvBnRelu(
                C_in=C_in, C_out=C * self._steps, kernel_size=1, stride=1, padding=0
            )
        )

    def forward(
        self,
        s0: torch.Tensor,
        weights: torch.Tensor,
        output_weights: torch.Tensor | None,
        input_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Adaption to NASBench
        # Only use a single input, from the previous cell
        states = []  # type:ignore

        # Loop through the choice blocks of each cell
        for choice_block_idx in range(self._steps):
            # Select the current weighting for input edges to each choice block
            if input_weights is not None:
                # Node 1 has no choice with respect to its input
                if (choice_block_idx == 0) or (
                    choice_block_idx == 1 and type(self.search_space) == NB1Shot1Space1
                ):
                    input_weight = None
                else:
                    input_weight = input_weights.pop(0)

            # Iterate over the choice blocks
            # Apply 1x1 projection only to edges from input of the cell
            # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L289
            s = self._choice_blocks[choice_block_idx](
                inputs=[self._input_projections[choice_block_idx](s0), *states],
                input_weights=input_weight,
                weights=weights[choice_block_idx],
            )
            states.append(s)

        # Add projected input to the state
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L328
        input_to_output_edge = self._input_projections[-1](s0)
        assert len(input_weights) == 0, "Something went wrong here."

        if output_weights is None:
            tensor_list = states
            input_to_output_weight = 1.0
        else:
            # Create weighted concatenation at the output of the cell
            tensor_list = [w * t for w, t in zip(output_weights[0][1:], states)]
            input_to_output_weight = output_weights[0][0]

        # Concatenate to form output tensor
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L325
        return input_to_output_weight * input_to_output_edge + torch.cat(
            tensor_list, dim=1
        )

    def prune_ops(self, mask: torch.Tensor) -> None:
        for choice_block_idx in range(self._steps):
            edge_mask = mask[choice_block_idx]
            set_ops_to_prune(self._choice_blocks[choice_block_idx].mixed_op, edge_mask)


class Network(nn.Module):
    def __init__(
        self,
        search_space: NB1Shot1Space,
        C: int = 16,
        num_classes: int = 10,
        layers: int = 9,
        output_weights: bool = False,
        steps: int = 4,
    ) -> None:
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._output_weights = output_weights
        self.search_space = search_space

        # In NASBench the stem has 128 output channels
        C_curr = C
        self.stem = ConvBnRelu(C_in=3, C_out=C_curr, kernel_size=3, stride=1)

        self.cells = nn.ModuleList()
        C_prev = C_curr
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # Double the number of channels after each down-sampling step
                # Down-sample in forward method
                C_curr *= 2
            cell = Cell(
                steps=self._steps,
                C_prev=C_prev,
                C=C_curr,
                layer=i,
                search_space=search_space,
            )
            self.cells += [cell]
            C_prev = C_curr
        self.postprocess = ReLUConvBN(
            C_in=C_prev * self._steps,
            C_out=C_curr,
            kernel_size=1,
            stride=1,
            padding=0,
            affine=False,
        )

        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()
        self.mask = None

    def new(self) -> Network:
        model_new = Network(
            self.search_space,
            self._C,
            self._num_classes,
            self._layers,
            steps=self.search_space.num_intermediate_nodes,
            output_weights=self._output_weights,
        ).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def sample(self, alphas: torch.Tensor) -> torch.Tensor:
        # Replace this function on the fly to change the sampling method
        return F.softmax(alphas, dim=-1)

    def sample_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        mixed_op_weights = self.sample(self._arch_parameters[0])
        output_weights = (
            self.sample(self._arch_parameters[1]) if self._output_weights else None
        )
        input_weights = [self.sample(alpha) for alpha in self._arch_parameters[2:]]

        if self.mask is not None:
            mixed_op_weights = normalize_params(mixed_op_weights, self.mask)

        return mixed_op_weights, output_weights, input_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NASBench only has one input to each cell
        mixed_op_weights, output_weights, input_weights = self.sample_weights()
        s0 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                # Perform down-sampling by factor 1/2
                # Equivalent to https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L68
                s0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(s0)

            # Sample mixed_op weights for the choice blocks in the graph
            mixed_op_weights_cell = mixed_op_weights.clone()

            # Sample the output weights (if applicable)
            output_weights_cell = (
                output_weights.clone() if self._output_weights else None  # type: ignore
            )

            # Sample the input weights for the nodes in the cell
            input_weights_cell = [weight.clone() for weight in input_weights]
            s0 = cell(
                s0, mixed_op_weights_cell, output_weights_cell, input_weights_cell
            )

        # Include one more preprocessing step here
        s0 = self.postprocess(s0)  # [N, C_max * (steps + 1), w, h] -> [N, C_max, w, h]

        # Global Average Pooling by averaging over last two remaining spatial dimensions
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L92
        out = s0.view(*s0.shape[:2], -1).mean(-1)
        logits = self.classifier(out.view(out.size(0), -1))
        return out, logits

    def _initialize_alphas(self) -> None:
        # Initializes the weights for the mixed ops.
        num_ops = len(PRIMITIVES)
        self.alphas_mixed_op = nn.Parameter(
            1e-3 * torch.randn(self._steps, num_ops).cuda(), requires_grad=True
        )

        # For the alphas on the output node initialize a weighting vector for all
        # choice blocks and the input edge.
        self.alphas_output = nn.Parameter(
            1e-3 * torch.randn(1, self._steps + 1).cuda(), requires_grad=True
        )

        begin = 3 if type(self.search_space) == NB1Shot1Space1 else 2
        # Initialize the weights for the inputs to each choice block.
        self.alphas_inputs = [
            nn.Parameter(1e-3 * torch.randn(1, n_inputs).cuda(), requires_grad=True)
            for n_inputs in range(begin, self._steps + 1)
        ]

        # Total architecture parameters
        self._arch_parameters = [
            self.alphas_mixed_op,
            self.alphas_output,
            *self.alphas_inputs,
        ]

        # TODO-ICLR: Beta parameters for edge normalization
        self._beta_parameters = [None]

    def arch_parameters(self) -> list[torch.Tensor]:
        return self._arch_parameters

    def beta_parameters(self) -> list[torch.Tensor] | None:
        return self._beta_parameters

    def get_genotype(self) -> Any:
        def softmax(weights: torch.Tensor, axis: int = -1) -> np.ndarray:
            return F.softmax(torch.Tensor(weights), axis).data.cpu().numpy()

        def get_top_k(array: np.ndarray, k: int) -> list:
            return list(np.argpartition(array[0], -k)[-k:])

        alphas_mixed_op = self.arch_parameters()[0]
        chosen_node_ops = softmax(alphas_mixed_op, axis=-1).argmax(-1)

        node_list = [PRIMITIVES[i] for i in chosen_node_ops]
        alphas_output = self.arch_parameters()[1]
        alphas_inputs = self.arch_parameters()[2:]

        if isinstance(self.search_space, NB1Shot1Space1):
            num_inputs = list(self.search_space.num_parents_per_node.values())[3:-1]
            parents_node_3, parents_node_4 = (
                get_top_k(softmax(alpha, axis=1), num_input)
                for num_input, alpha in zip(num_inputs, alphas_inputs)
            )
            output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
            parents = {
                "0": [],
                "1": [0],
                "2": [0, 1],
                "3": parents_node_3,
                "4": parents_node_4,
                "5": output_parents,
            }
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

        elif isinstance(self.search_space, NB1Shot1Space2):
            num_inputs = list(self.search_space.num_parents_per_node.values())[2:]
            parents_node_2, parents_node_3, parents_node_4 = (
                get_top_k(softmax(alpha, axis=1), num_input)
                for num_input, alpha in zip(num_inputs[:-1], alphas_inputs)
            )
            output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
            parents = {
                "0": [],
                "1": [0],
                "2": parents_node_2,
                "3": parents_node_3,
                "4": parents_node_4,
                "5": output_parents,
            }
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

        elif isinstance(self.search_space, NB1Shot1Space3):
            num_inputs = list(self.search_space.num_parents_per_node.values())[2:]
            parents_node_2, parents_node_3, parents_node_4, parents_node_5 = (
                get_top_k(softmax(alpha, axis=1), num_input)
                for num_input, alpha in zip(num_inputs[:-1], alphas_inputs)
            )
            output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
            parents = {
                "0": [],
                "1": [0],
                "2": parents_node_2,
                "3": parents_node_3,
                "4": parents_node_4,
                "5": parents_node_5,
                "6": output_parents,
            }
            node_list = [INPUT, *node_list, OUTPUT]

        else:
            raise ValueError("Unknown search space")

        adjacency_matrix = self.search_space.create_nasbench_adjacency_matrix(parents)
        # Convert the adjacency matrix in format for nasbench

        adjacency_list = adjacency_matrix.astype(np.int32).tolist()

        genotype = NASBench1Shot1ConfoptGenotype(matrix=adjacency_list, ops=node_list)

        return genotype

    def prune(self, prune_fraction: float) -> None:
        # mask is only applied to mixedop weights,
        # not to input and output weights

        assert prune_fraction < 1, "Prune fraction should be less than 1"
        assert prune_fraction >= 0, "Prune fraction greater or equal to 0"

        num_ops = len(PRIMITIVES)
        top_k = num_ops - int(num_ops * prune_fraction)

        self.mask = prune(self.alphas_mixed_op, top_k, self.mask)

        for cell in self.cells:
            cell.prune_ops(self.mask)
