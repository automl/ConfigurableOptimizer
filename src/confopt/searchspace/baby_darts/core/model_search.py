from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.distributions import Dirichlet
import torch.nn.functional as F  # noqa: N812

from confopt.searchspace.baby_darts.core.operations import OLES_OPS, OPS, Identity
from confopt.searchspace.common.mixop import OperationBlock, OperationChoices
from confopt.searchspace.darts.core.genotypes import (
    PRIMITIVES,
    DARTSGenotype,
)
from confopt.searchspace.darts.core.model import NetworkCIFAR, NetworkImageNet
from confopt.utils import (
    preserve_gradients_in_module,
    prune,
    set_ops_to_prune,
)
from confopt.utils.normalize_params import normalize_params

NUM_CIFAR_CLASSES = 10
NUM_CIFAR100_CLASSES = 100
NUM_IMAGENET_CLASSES = 120
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MixedOp(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int,
        primitives: list[str] = PRIMITIVES,
        k: int = 1,
    ):
        super().__init__()
        self.C = C_in
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C_in // k, C_out // k, stride, False)
            self._ops.append(op)

    def forward(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        return sum(w * op(x) for w, op in zip(weights, self._ops))  # type: ignore


class Cell(nn.Module):
    def __init__(
        self,
        steps: int,
        multiplier: int,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,  # noqa: ARG002
        primitives: list[str] = PRIMITIVES,
        k: int = 1,
    ):
        """Neural Cell for DARTS.

        Represents a neural cell used in DARTS.

        Args:
            steps (int): Number of steps in the cell.
            multiplier (int): Multiplier for channels in the cell.
            C_prev_prev (int): Number of channels in the previous-previous cell.
            C_prev (int): Number of channels in the previous cell.
            C (int): Number of channels in the current cell.
            reduction (bool): Whether the cell is a reduction cell.
            reduction_prev (bool): Whether the previous cell is a reduction cell.
            primitives (list): The list of primitives to use for generating cell.
            k (int): Shows how much of the channel widths should be used. Defaults to 1.

        Attributes:
            preprocess0(nn.Module): Preprocess for input from previous-previous cell.
            preprocess1(nn.Module): Preprocess for input from previous cell.
            _ops(nn.ModuleList): List of operations in the cell.
            reduction(bool): Whether the cell is a reduction cell (True) or
                             a normal cell (False).
        """
        super().__init__()
        self.reduction = reduction
        assert not self.reduction
        self.C = C
        self.C_prev = C_prev
        self.C_prev_prev = C_prev_prev

        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        stride = 1
        ops = MixedOp(3, C, stride, primitives, k)._ops
        op = OperationChoices(ops, is_reduction_cell=reduction)
        self._ops.append(op)

    def forward(
        self,
        s0: torch.Tensor,
        weights: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of the cell.

        Args:
            s0 (torch.Tensor): First input tensor to the model.
            s1 (torch.Tensor): Second input tensor to the model.
            weights (list[torch.Tensor]): Alpha weights to the edges.
            beta_weights (torch.Tensor): Beta weights for the edge.
            drop_prob: (float|None): the droping probability of a path (for discrete).

        Returns:
            torch.Tensor: state ouptut from the cell
        """
        return self._ops[0](s0, weights[0])

    def prune_ops(self, mask: torch.Tensor) -> None:
        assert len(self._ops) == mask.shape[0]
        for edge, edge_mask in zip(self._ops, mask):
            set_ops_to_prune(edge, edge_mask)

    def change_op_stride_size(self, new_stride: int, is_reduction_cell: bool) -> None:
        idx = 0
        for i in range(self._steps):
            for j in range(2 + i):
                op = self._ops[idx]
                op.is_reduction_cell = is_reduction_cell
                if j < 2:
                    op.change_op_stride_size(new_stride)
                idx += 1

    def get_weighted_flops(self, alphas: torch.Tensor) -> torch.Tensor:
        flops = 0
        for idx, op in enumerate(self._ops):
            flops += op.get_weighted_flops(alphas[idx])
        return flops


class Network(nn.Module):
    def __init__(
        self,
        C: int = 16,
        num_classes: int = 10,
        layers: int = 8,
        criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss,
        steps: int = 4,
        multiplier: int = 4,
        stem_multiplier: int = 3,
        edge_normalization: bool = False,
        discretized: bool = False,
        primitives: list[str] = PRIMITIVES,
        k: int = 1,
    ) -> None:
        """Implementation of DARTS search space's network model.

        Args:
            C (int): Number of channels. Defaults to 16.
            num_classes (int): Number of output classes. Defaults to 10.
            layers (int): Number of layers in the network. Defaults to 8.
            criterion (nn.modules.loss._Loss): Loss function. Defaults to nn.CrossEntropyLoss.
            steps (int): Number of steps in the search space cell. Defaults to 4.
            multiplier (int): Multiplier for channels in the cells. Defaults to 4.
            stem_multiplier (int): Stem multiplier for channels. Defaults to 3.
            edge_normalization (bool): Flag to  use edge normalization. Defaults to False.
            discretized (bool): Whether supernet is discretized to only have one operation on
            each edge or not.
            primitives (list): The list of primitives to use for generating cell.
            k (int): how much of the channel width should be used in the forward pass. Defaults to 1 which mean the whole channel width.

        Attributes:
            stem (nn.Sequential): Stem network composed of Conv2d and BatchNorm2d layers.
            cells (nn.ModuleList): List of cells in the search space.
            global_pooling (nn.AdaptiveAvgPool2d): Global pooling layer.
            classifier (nn.Linear): Linear classifier layer.
            alphas_normal (nn.Parameter): Parameter for normal cells' alpha values.
            alphas_reduce (nn.Parameter): Parameter for reduction cells' alpha values.
            arch_parameters (list[nn.Parameter]): List of parameter for architecture alpha values.
            discretized (bool): Whether the network is dicretized or not

        Note:
            This is a custom neural network model with various hyperparameters and
            architectural choices.
        """  # noqa: E501
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.edge_normalization = edge_normalization
        self.discretized = discretized
        self.mask: list[torch.Tensor] | None = None
        self.last_mask: list[torch.Tensor] = []
        C_curr = stem_multiplier * C
        self.stem = Identity()
        self.primitives = primitives

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for _ in range(layers):
            # there are no reduction cells
            reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                self.primitives,
                k,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.weights_grad: dict[str, list[torch.Tensor]] = {}
        self.grad_hook_handlers: list[torch.utils.hooks.RemovableHandle] = []

        # Multi-head attention for architectural parameters
        self.is_arch_attention_enabled = False  # disabled by default
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=len(self.primitives), num_heads=1
        )

        # mask for pruning
        self._initialize_parameters()

    def new(self) -> Network:
        """Get a new object with same arch and beta parameters.

        Return:
            Network: A torch module with same arch and beta parameters as this model.
        """
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion
        ).to(DEVICE)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        for x, y in zip(model_new.beta_parameters(), self.beta_parameters()):
            x.data.copy_(y.data)
        return model_new

    def sample(self, alphas: torch.Tensor) -> torch.Tensor:
        # Replace this function on the fly to change the sampling method
        return F.softmax(alphas, dim=-1)

    def save_weight_grads(
        self, weights: torch.Tensor, cell_type: Literal["reduce", "normal"]
    ) -> None:
        assert cell_type in ["reduce", "normal"]
        if not self.training:
            return
        grad_hook = weights.register_hook(self.save_gradient(cell_type=cell_type))
        self.grad_hook_handlers.append(grad_hook)

    def sample_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights_normal_to_sample = self.alphas_normal
        weights_normal = self.sample(weights_normal_to_sample)

        if self.mask is not None:
            weights_normal = normalize_params(weights_normal, self.mask[0])

        return weights_normal, None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network model.

        Args:
            x (torch.Tensor): Input x tensor to the model.
            drop_prob (float|None): the droping probability of a path.


        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - The output tensor after the forward pass.
            - The logits tensor produced by the model.
        """
        if self.edge_normalization:
            return self.edge_normalization_forward(x)

        s0 = s1 = self.stem(x)
        weights_normal, _ = self.sample_weights()
        self.sampled_weights = [weights_normal]

        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                print("For this toy searchspace, there is only 1 normal")
            else:
                weights = weights_normal.clone()
            s0, s1 = s1, cell(s0, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return torch.squeeze(out, dim=(-1, -2)), logits

    def _loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss for the given input data and target.

        Args:
            x (torch.Tensor): Input data.
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Computed loss value.

        """
        logits = self(x)
        return self._criterion(logits, target)  # type: ignore

    def _initialize_parameters(self) -> None:
        """Initialize architectural and beta parameters for the cell.

        This function initializes the architectural and beta parameters required for
        the neural cell.
        """
        k = 1
        self.num_ops = len(self.primitives)
        self.num_edges = k

        self.num_nodes = self._steps - 1

        self.alphas_normal = nn.Parameter(
            1e-3 * torch.randn(k, self.num_ops).to(DEVICE)
        )
        self._arch_parameters = [
            self.alphas_normal,
        ]

        self.anchor_normal = Dirichlet(torch.ones_like(self.alphas_normal).to(DEVICE))

    def arch_parameters(self) -> list[torch.nn.Parameter]:
        """Get a list containing the architecture parameters or alphas.

        Returns:
            list[torch.Tensor]: A list containing the architecture parameters, such as
            alpha values.
        """
        return self._arch_parameters  # type: ignore

    def beta_parameters(self) -> list[torch.nn.Parameter]:
        """Get a list containing the beta parameters of partial connection used for
        edge normalization.

        Returns:
            list[torch.Tensor]: A list containing the beta parameters for the model.
        """
        return []

    def genotype(self) -> DARTSGenotype:
        """Get the genotype of the model, representing the architecture.

        Returns:
            Structure: An object representing the genotype of the model, which describes
            the architectural choices in terms of operations and connections between
            nodes.
        """
        # return always a default genotype
        genotype = DARTSGenotype(
            normal=[
                ("skip_connect", 0),
                ("sep_conv_5x5", 1),
                ("sep_conv_5x5", 0),
                ("sep_conv_5x5", 2),
                ("sep_conv_5x5", 2),
                ("dil_conv_3x3", 3),
                ("sep_conv_5x5", 2),
                ("dil_conv_3x3", 4),
            ],
            normal_concat=range(2, 6),
            reduce=[
                ("max_pool_3x3", 1),
                ("skip_connect", 0),
                ("dil_conv_5x5", 1),
                ("skip_connect", 0),
                ("skip_connect", 3),
                ("skip_connect", 2),
                ("skip_connect", 2),
                ("skip_connect", 3),
            ],
            reduce_concat=range(2, 6),
        )
        return genotype

    def prune(self, prune_fraction: float) -> None:
        """Discretize architecture parameters to enforce sparsity.

        Args:
            prune_fraction (float): Fraction of operations to remove.
        """
        assert prune_fraction < 1, "Prune fraction should be less than 1"
        assert prune_fraction >= 0, "Prune fraction greater or equal to 0"

        num_ops = len(self.primitives)
        top_k = int(num_ops * (1 - prune_fraction))

        if self.mask is None:
            self.mask = []

        # Calculate masks
        for i, p in enumerate(self._arch_parameters):
            if i < len(self.mask):
                self.mask[i] = prune(p, top_k, self.mask[i])
            else:
                mask = prune(p, top_k, None)
                self.mask.append(mask)

        for cell in self.cells:
            if cell.reduction:
                cell.prune_ops(self.mask[1])
            else:
                cell.prune_ops(self.mask[0])

    def discretize(self) -> NetworkCIFAR | NetworkImageNet:
        genotype = self.genotype()
        if self._num_classes in {NUM_CIFAR_CLASSES, NUM_CIFAR100_CLASSES}:
            discrete_model = NetworkCIFAR(
                C=self._C,
                num_classes=self._num_classes,
                layers=self._layers,
                auxiliary=False,
                genotype=genotype,
            )
        elif self._num_classes == NUM_IMAGENET_CLASSES:
            discrete_model = NetworkImageNet(
                C=self._C,
                num_classes=self._num_classes,
                layers=self._layers,
                auxiliary=False,
                genotype=genotype,
            )
        else:
            raise ValueError(
                "number of classes is not a valid number of any of the datasets"
            )

        discrete_model.to(next(self.parameters()).device)

        return discrete_model

    def model_weight_parameters(self) -> list[nn.Parameter]:
        params = set(self.parameters())
        params -= set(self._betas)
        if self._arch_parameters != [None]:
            params -= set(self.alphas_normal)
        return list(params)


def preserve_grads(m: nn.Module) -> None:
    ignored_modules = (
        OperationBlock,
        OperationChoices,
        Cell,
        MixedOp,
        Network,
    )

    preserve_gradients_in_module(m, ignored_modules, OLES_OPS)
