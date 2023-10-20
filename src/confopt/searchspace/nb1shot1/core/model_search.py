from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from confopt.searchspace.common.mixop import OperationChoices

from .genotypes import PRIMITIVES
from .operations import OPS, ConvBnRelu, ReLUConvBN

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MixedOp(nn.Module):
    def __init__(self, C: int, stride: int):
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

    def forward(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.float:
        return sum(w * op(x) for w, op in zip(weights, self._ops))  # type: ignore


class ChoiceBlock(nn.Module):
    """Adapted to match Figure 3 in:
    Bender, Gabriel, et al. "Understanding and simplifying one-shot
    architecture search." International Conference on Machine Learning. 2018.
    """

    def __init__(self, C_in: int):
        super().__init__()
        # Pre-processing 1x1 convolution at the beginning of each choice block.
        ops = MixedOp(C_in, stride=1)._ops
        self.mixed_op = OperationChoices(ops)

    def forward(
        self,
        inputs: list,
        input_weights: torch.Tensor,
        weights: list[torch.Tensor],
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
        self,
        steps: int,
        C_prev: int,
        C: int,
        layer: int,
        search_space_info: dict,
    ):
        super().__init__()
        # All cells are normal cells in NASBench case.
        self._steps = steps

        self._choice_blocks = nn.ModuleList()
        self._bns = nn.ModuleList()
        self.search_space_info = search_space_info

        self._input_projections = nn.ModuleList()
        # Number of input channels is dependent on whether it is the first layer or not.
        # Any subsequent layer has C_in * (steps + 1) input channels because the output
        # is a concatenation of the input tensor and all choice block outputs
        C_in = C_prev if layer == 0 else C_prev * steps

        # Create the choice block and the input
        for _ in range(self._steps):
            choice_block = ChoiceBlock(C_in=C)
            self._choice_blocks.append(choice_block)
            self._input_projections.append(
                ConvBnRelu(C_in=C_in, C_out=C, kernel_size=1, stride=1, padding=0)
            )

        # Add one more input preprocessing for edge from input to output of the cell
        self._input_projections.append(
            ConvBnRelu(
                C_in=C_in,
                C_out=C * self._steps,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

    def forward(
        self,
        s0: torch.Tensor,
        weights: list,
        output_weights: list,
        input_weights: list,
    ) -> torch.Tensor:
        # Adaption to NASBench
        # Only use a single input, from the previous cell
        states: list = []

        # Loop through the choice blocks of each cell
        for choice_block_idx in range(self._steps):
            # Select the current weighting for input edges to each choice block
            if input_weights is not None:
                # Node 1 has no choice with respect to its input
                if (choice_block_idx == 0) or (
                    choice_block_idx == 1
                    and self.search_space_info["search_space_type"] == "S1"
                ):
                    input_weight = None
                else:
                    input_weight = input_weights.pop(0)

            # Iterate over the choice blocks
            # Apply 1x1 projection only to edges from input of the cell
            # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L289
            s = self._choice_blocks[choice_block_idx](
                inputs=[
                    self._input_projections[choice_block_idx](s0),
                    *states,
                ],
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
        else:
            # Create weighted concatenation at the output of the cell
            tensor_list = [w * t for w, t in zip(output_weights[0][1:], states)]

        # Concatenate to form output tensor
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L325
        return output_weights[0][0] * input_to_output_edge + torch.cat(
            tensor_list, dim=1
        )


class Network(nn.Module):
    def __init__(
        self,
        search_space_info: dict,
        C: int = 16,
        num_classes: int = 10,
        layers: int = 8,
        criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss,
        output_weights: bool = True,
        steps: int = 4,
    ):
        """Initialize a neural network architecture of the search space.

        Args:
            search_space_info (dict): Information about the search space.
            C (int): Number of channels in the architecture (default: 16).
            num_classes (int): Number of output classes (default: 10).
            layers (int): Number of layers in the architecture (default: 8).
            criterion (nn.modules.loss._Loss): Loss criterion
            (default: CrossEntropyLoss).
            output_weights (bool): Whether to include output weights (default: True).
            steps (int): Number of steps or intermediate nodes (default: 4).

        Returns:
            None
        """
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._output_weights = output_weights
        self.search_space_info = search_space_info

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
                search_space_info=search_space_info,
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

    def new(self) -> Network:
        """Create a new Network with the same architecture parameters as the current
        instance.

        Returns:
            Network: A new instance of the Network class.
        """
        model_new = Network(
            C=self._C,
            num_classes=self._num_classes,
            layers=self._layers,
            criterion=self._criterion,
            steps=self.search_space_info["num_intermediate_nodes"],
            output_weights=self._output_weights,
            search_space_info=self.search_space_info,
        ).to(DEVICE)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _preprocess_op(
        self, x: torch.Tensor, discrete: bool, normalize: bool
    ) -> torch.Tensor:
        """Preprocess architecture weights.

        Args:
            x (torch.Tensor): The input tensor representing architecture weights.
            discrete (bool): Whether to use discrete architecture weights.
            normalize (bool): Whether to normalize weights.

        Returns:
            torch.Tensor: original or normalized or softmax of x
        """
        if discrete and normalize:
            raise ValueError("architecture can't be discrete and normalized")
        # If using discrete architecture from random_ws search with weight sharing
        # then pass through architecture weights directly.
        if discrete or (hasattr(self, "discretized") and self.discretized):
            return x
        elif normalize:  # noqa: RET505
            arch_sum = torch.sum(x, dim=-1)
            if arch_sum > 0:
                return x / arch_sum

            return x
        else:
            # Normal search softmax over the inputs and mixed ops.
            return F.softmax(x, dim=-1)

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        discrete: bool = False,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input (torch.Tensor): The input tensor.
            discrete (bool, optional): Whether to use discrete weights. Defaults to
            False.
            normalize (bool, optional): Whether to normalize weights. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing two tensors,
            the intermediate output and the final logits.
        """
        # NASBench only has one input to each cell
        s0 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                # Perform down-sampling by factor 1/2
                # Equivalent to https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L68
                s0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(s0)

            # Normalize mixed_op weights for the choice blocks in the graph
            mixed_op_weights = self._preprocess_op(
                self._arch_parameters[0], discrete=discrete, normalize=False
            )

            # Normalize the output weights
            output_weights = (
                self._preprocess_op(self._arch_parameters[1], discrete, normalize)
                if self._output_weights
                else None
            )
            # Normalize the input weights for the nodes in the cell
            input_weights = [
                self._preprocess_op(alpha, discrete, normalize)
                for alpha in self._arch_parameters[2:]
            ]
            s0 = cell(s0, mixed_op_weights, output_weights, input_weights)

        # Include one more preprocessing step here
        s0 = self.postprocess(s0)  # [N, C_max * (steps + 1), w, h] -> [N, C_max, w, h]

        # Global Average Pooling by averaging over last two remaining spatial dimensions
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L92
        out = s0.view(*s0.shape[:2], -1).mean(-1)
        logits = self.classifier(out.view(out.size(0), -1))
        return out, logits

    def _loss(
        self, input: torch.Tensor, target: torch.Tensor  # noqa: A002
    ) -> torch.Tensor:
        """Compute the loss for a given input and target.

        Args:
            input (torch.Tensor): The input data.
            target (torch.Tensor): The target data.

        Returns:
            torch.Tensor: The computed loss.
        """
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self) -> None:
        """Initialize the architecture parameters.

        This method initializes the weights for the mixed operations, the alphas on
        the output node, and the weights for the inputs to each choice block. It also
        sets the total architecture parameters.

        Note:
            The initial weights are generated with random values.
            In comparison to the original paper of https://arxiv.org/pdf/2001.10422.pdf
            alphas_mixed_op is equivalent to beta
            alphas_output is equivalent to alpha
            alphas_input is equivalent to gama

        Returns:
            None
        """
        # Initializes the weights for the mixed ops.
        num_ops = len(PRIMITIVES)
        self.alphas_mixed_op = nn.Parameter(
            1e-3 * torch.randn(self._steps, num_ops).to(DEVICE),
            requires_grad=True,
        )

        # For the alphas on the output node initialize a weighting vector for all choice
        # blocks and the input edge.
        self.alphas_output = nn.Parameter(
            1e-3 * torch.randn(1, self._steps + 1).to(DEVICE),
            requires_grad=True,
        )

        begin = 3 if self.search_space_info["search_space_type"] == "S1" else 2
        # Initialize the weights for the inputs to each choice block.
        self.alphas_inputs = [
            nn.Parameter(1e-3 * torch.randn(1, n_inputs).to(DEVICE), requires_grad=True)
            for n_inputs in range(begin, self._steps + 1)
        ]

        # Total architecture parameters
        self._arch_parameters = [
            self.alphas_mixed_op,
            self.alphas_output,
            *self.alphas_inputs,
        ]

    def arch_parameters(self) -> list[nn.Parameter]:
        """Get architecture parameters.

        Returns:
            list[nn.Parameter]: A list of architecture parameters.
        """
        return self._arch_parameters

    def _discretize(self, op_sparsity: float) -> None:
        """Discretize architecture parameters to enforce sparsity.

        Args:
            op_sparsity (float): The desired sparsity level, represented as a
            fraction of operations to keep.

        Note:
            This method enforces sparsity in the architecture parameters by zeroing out
            a fraction of the smallest values, as specified by the `op_sparsity`
            parameter.
            It modifies the architecture parameters in-place to achieve the desired
            sparsity.
        """
        # self.edge_normalization = False
        self.discretized = True

        for p in self._arch_parameters:
            top_k = max(int(op_sparsity * p.data.shape[-1]), 1)
            sorted_arch_params, _ = torch.sort(p.data, dim=1, descending=True)
            thresholds = sorted_arch_params[:, :top_k]
            mask = p.data >= thresholds

            p.data *= mask.float()
            p.data[~mask].requires_grad = False
            if p.data[~mask].grad:
                p.data[~mask].grad.zero_()
