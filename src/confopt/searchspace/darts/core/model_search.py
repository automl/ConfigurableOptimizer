from __future__ import annotations

import copy
import math
from typing import Callable, Literal
import warnings

import torch
from torch import nn
from torch.distributions import Dirichlet
import torch.nn.functional as F  # noqa: N812

from confopt.searchspace.common.mixop import OperationBlock, OperationChoices
from confopt.utils import (
    calc_layer_alignment_score,
    get_pos_new_cell_darts,
    get_pos_reductions_darts,
    preserve_gradients_in_module,
    prune,
    set_ops_to_prune,
)
from confopt.utils.normalize_params import normalize_params

from .genotypes import (
    BABY_PRIMITIVES,
    PRIMITIVES,
    DARTSGenotype,
    get_skip_connection_index,
)
from .model import NetworkCIFAR, NetworkImageNet
from .operations import OLES_OPS, OPS, FactorizedReduce, Identity, ReLUConvBN

NUM_CIFAR_CLASSES = 10
NUM_CIFAR100_CLASSES = 100
NUM_IMAGENET_CLASSES = 120
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MixedOp(nn.Module):
    def __init__(
        self, C: int, stride: int, primitives: list[str] = PRIMITIVES, k: int = 1
    ):
        super().__init__()
        self.C = C
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C // k, stride, False)
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
        reduction_prev: bool,
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
        self.C = C
        self.C_prev = C_prev
        self.C_prev_prev = C_prev_prev

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(
                C_prev_prev, C, 1, 1, 0, affine=False
            )  # type: ignore
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                ops = MixedOp(C, stride, primitives, k)._ops
                op = OperationChoices(ops, is_reduction_cell=reduction)
                self._ops.append(op)

    def forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights: list[torch.Tensor],
        beta_weights: torch.Tensor | None = None,
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
        if beta_weights is not None:
            return self.edge_normalization_forward(s0, s1, weights, beta_weights)

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)

    def edge_normalization_forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights: list[torch.Tensor],
        beta_weights: torch.Tensor,
    ) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _i in range(self._steps):
            s = sum(
                beta_weights[offset + j] * self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)

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

    def change_preprocessing0_type(
        self,
        prev_reduction: bool = False,
    ) -> None:
        if prev_reduction:
            self.preprocess0 = FactorizedReduce(self.C_prev_prev, self.C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(
                self.C_prev_prev, self.C, 1, 1, 0, affine=False
            )  # type: ignore

    def change_op_channel_size(
        self,
        new_cell: bool = True,
    ) -> None:
        # if pre0 was not changed
        if self.preprocess0.C_in != self.C_prev_prev:
            num_channels_to_add = self.C_prev_prev - self.preprocess0.C_in
            self.preprocess0.increase_in_channel_size(
                num_channels_to_add=num_channels_to_add
            )
        if self.preprocess0.C_out != self.C:
            num_channels_to_add = self.C - self.preprocess0.C_out
            self.preprocess0.increase_out_channel_size(
                num_channels_to_add=num_channels_to_add
            )
        if self.preprocess1.C_in != self.C_prev:
            num_channels_to_add = self.C_prev - self.preprocess1.C_in
            self.preprocess1.increase_in_channel_size(
                num_channels_to_add=num_channels_to_add
            )
        if self.preprocess1.C_out != self.C:
            num_channels_to_add = self.C - self.preprocess1.C_out
            self.preprocess1.increase_out_channel_size(
                num_channels_to_add=num_channels_to_add
            )

        if self.reduction and new_cell:
            for op in self._ops:
                op.change_op_channel_size(k=1 / 2, new_cell=True)

    def change_skip_connection_type(self, reduction: bool) -> None:
        idx = 0
        skip_connection_index = get_skip_connection_index()
        for i in range(self._steps):
            for j in range(2 + i):
                if j < 2:
                    if reduction:
                        self._ops[idx].ops[skip_connection_index] = FactorizedReduce(
                            self.C, self.C, affine=False
                        )
                    else:
                        self._ops[idx].ops[skip_connection_index] = Identity()
                idx += 1

    def increase_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
    ) -> None:
        for op in self._ops:
            op.change_op_channel_size(
                k=k, num_channels_to_add=num_channels_to_add, new_cell=True
            )


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
        is_baby_darts: bool = False,
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
            edge_normalization (bool): Whether to use edge normalization. Defaults to False.
            discretized (bool): Whether supernet is discretized to only have one operation on
            each edge or not.
            is_baby_darts (bool): Controls which primitive list to use
            k (int): how much of the channel width should be used in the forward pass. Defaults to 1 which mean the whole channel width.

        Attributes:
            stem (nn.Sequential): Stem network composed of Conv2d and BatchNorm2d layers.
            cells (nn.ModuleList): List of cells in the search space.
            global_pooling (nn.AdaptiveAvgPool2d): Global pooling layer.
            classifier (nn.Linear): Linear classifier layer.
            alphas_normal (nn.Parameter): Parameter for normal cells' alpha values.
            alphas_reduce (nn.Parameter): Parameter for reduction cells' alpha values.
            arch_parameters (list[nn.Parameter]): List of parameter for architecture alpha values.
            betas_normal (nn.Parameter): Parameter for normal cells' beta values.
            betas_reduce (nn.Parameter): Parameter for normal cells' beta values.
            beta_parameters (list[nn.Parameter]): List of parameter for architecture alpha values.
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
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        self.is_baby_darts = is_baby_darts
        if is_baby_darts:
            self.primitives = BABY_PRIMITIVES
        else:
            self.primitives = PRIMITIVES

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in get_pos_reductions_darts(layers=layers):
                C_curr *= 2
                reduction = True
            else:
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

    def reset_hooks(self) -> None:
        for hook in self.grad_hook_handlers:
            hook.remove()
        self.grad_hook_handlers = []

    def save_gradient(self, cell_type: Literal["normal", "reduce"]) -> Callable:
        def hook(grad: torch.Tensor) -> None:
            self.weights_grad[cell_type].append(grad)

        return hook

    def save_weight_grads(
        self, weights: torch.Tensor, cell_type: Literal["reduce", "normal"]
    ) -> None:
        assert cell_type in ["reduce", "normal"]
        if not self.training:
            return
        grad_hook = weights.register_hook(self.save_gradient(cell_type=cell_type))
        self.grad_hook_handlers.append(grad_hook)

    def sample_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.projection_mode:
            weights_normal, weights_reduce = (
                self.get_projected_weights("normal"),
                self.get_projected_weights("reduce"),
            )
            return weights_normal, weights_reduce

        weights_normal_to_sample = self.alphas_normal
        weights_reduce_to_sample = self.alphas_reduce

        if self.is_arch_attention_enabled:
            (
                weights_normal_to_sample,
                weights_reduce_to_sample,
            ) = self._compute_arch_attention(
                weights_normal_to_sample, weights_reduce_to_sample
            )

        weights_normal = self.sample(weights_normal_to_sample)
        weights_reduce = self.sample(weights_reduce_to_sample)

        if self.mask is not None:
            weights_normal = normalize_params(weights_normal, self.mask[0])
            weights_reduce = normalize_params(weights_reduce, self.mask[1])

        return weights_normal, weights_reduce

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
        self.reset_hooks()
        if self.edge_normalization:
            return self.edge_normalization_forward(x)

        s0 = s1 = self.stem(x)
        self.weights_grad["normal"] = []
        self.weights_grad["reduce"] = []

        weights_normal, weights_reduce = self.sample_weights()
        self.sampled_weights = [weights_normal, weights_reduce]

        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce.clone()
                self.save_weight_grads(weights, cell_type="reduce")
            else:
                weights = weights_normal.clone()
                self.save_weight_grads(weights, cell_type="normal")

            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return torch.squeeze(out, dim=(-1, -2)), logits

    def edge_normalization_forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: normalization of alphas

        s0 = s1 = self.stem(inputs)
        self.weights_grad["normal"] = []
        self.weights_grad["reduce"] = []

        weights_normal, weights_reduce = self.sample_weights()
        self.sampled_weights = [weights_normal, weights_reduce]

        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce.clone()
                self.save_weight_grads(weights, cell_type="reduce")
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
                for _i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                weights = weights_normal.clone()
                self.save_weight_grads(weights, cell_type="normal")
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
                for _i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)

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
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.num_ops = len(self.primitives)
        self.num_edges = k

        self.num_nodes = self._steps - 1

        self.alphas_normal = nn.Parameter(
            1e-3 * torch.randn(k, self.num_ops).to(DEVICE)
        )
        self.alphas_reduce = nn.Parameter(
            1e-3 * torch.randn(k, self.num_ops).to(DEVICE)
        )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

        self.betas_normal = nn.Parameter(1e-3 * torch.randn(k).to(DEVICE))
        self.betas_reduce = nn.Parameter(1e-3 * torch.randn(k).to(DEVICE))
        self._betas = [
            self.betas_normal,
            self.betas_reduce,
        ]

        self.anchor_normal = Dirichlet(torch.ones_like(self.alphas_normal).cuda())
        self.anchor_reduce = Dirichlet(torch.ones_like(self.alphas_reduce).cuda())
        self._initialize_projection_params()

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
        return self._betas

    def genotype(self) -> DARTSGenotype:
        """Get the genotype of the model, representing the architecture.

        Returns:
            Structure: An object representing the genotype of the model, which describes
            the architectural choices in terms of operations and connections between
            nodes.
        """

        def _parse(weights: list[torch.Tensor]) -> list[tuple[str, int]]:
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k]
                        for k in range(len(W[x]))  # type: ignore
                        if k != self.primitives.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != self.primitives.index("none") and (
                            k_best is None or W[j][k] > W[j][k_best]
                        ):
                            k_best = k
                    gene.append((self.primitives[k_best], j))  # type: ignore
                start = end
                n += 1
            return gene

        if self.is_arch_attention_enabled:
            alphas_normal, alphas_reduce = self._compute_arch_attention(
                self.alphas_normal, self.alphas_reduce
            )
        else:
            alphas_normal = self.alphas_normal
            alphas_reduce = self.alphas_reduce

        gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = DARTSGenotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype

    def get_arch_grads(
        self, only_first_and_last: bool = False
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        def get_grads(alphas_grads_list: list[torch.Tensor]) -> list[torch.Tensor]:
            grads = []
            if only_first_and_last:
                grads.append(alphas_grads_list[0].reshape(-1))
                grads.append(alphas_grads_list[-1].reshape(-1))
            else:
                for alphas_grad in alphas_grads_list:
                    grads.append(alphas_grad.reshape(-1))

            return grads

        grads_normal = get_grads(self.weights_grad["normal"])
        grads_reduce = get_grads(self.weights_grad["reduce"])
        return grads_normal, grads_reduce

    def get_mean_layer_alignment_score(
        self, only_first_and_last: bool = False
    ) -> tuple[float, float]:
        grads_normal, grads_reduce = self.get_arch_grads(only_first_and_last)
        mean_score_normal = calc_layer_alignment_score(grads_normal)
        mean_score_reduce = calc_layer_alignment_score(grads_reduce)

        if math.isnan(mean_score_normal):
            mean_score_normal = 0
        if math.isnan(mean_score_reduce):
            mean_score_reduce = 0
        return mean_score_normal, mean_score_reduce

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
            params -= set(self.alphas_reduce)
            params -= set(self.alphas_normal)
        return list(params)

    def _initialize_projection_params(self) -> None:
        self.proj_weights = {  # for hard/soft assignment after project
            "normal": torch.zeros_like(self.alphas_normal),
            "reduce": torch.zeros_like(self.alphas_reduce),
        }

        self.candidate_flags = {
            "normal": torch.tensor(
                self.num_edges * [True], requires_grad=False, dtype=torch.bool
            ).to(DEVICE),
            "reduce": torch.tensor(
                self.num_edges * [True], requires_grad=False, dtype=torch.bool
            ).to(DEVICE),
        }
        self.candidate_flags_edge = {
            "normal": torch.tensor(
                3 * [True], requires_grad=False, dtype=torch.bool
            ).to(DEVICE),
            "reduce": torch.tensor(
                3 * [True], requires_grad=False, dtype=torch.bool
            ).to(DEVICE),
        }

        # for outgoing edges
        self.nid2eids: dict[int, list[int]] = {}
        offset = 2  # 2 inital edges to node 0
        num_states = 3  # 2 initial states and 1 incoming edge to node 0

        for i in range(self.num_nodes):
            for j in range(num_states):
                if i not in self.nid2eids:
                    self.nid2eids[i] = [offset + j]
                else:
                    self.nid2eids[i].append(offset + j)
            offset += num_states
            num_states += 1

        self.nid2selected_eids: dict[str, dict[int, list[int]]] = {
            "normal": {},
            "reduce": {},
        }
        for i in range(self.num_nodes):
            self.nid2selected_eids["normal"][i] = []
            self.nid2selected_eids["reduce"][i] = []

        self.projection_mode = False
        self.projection_evaluation = False
        self.removal_cell_type: Literal["normal", "reduce"] | None = None
        self.removed_projected_weights = {
            "normal": None,
            "reduce": None,
        }

    def remove_from_projected_weights(
        self,
        cell_type: Literal["normal", "reduce"],
        selected_edge: int,
        selected_op: int | None,
        topology: bool = False,
    ) -> None:
        weights = self.get_projected_weights(cell_type)
        proj_mask = torch.ones_like(weights[selected_edge])
        if topology:
            if selected_op is not None:
                warnings.warn(
                    "selected_op should be set to None in case of topology search",
                    stacklevel=1,
                )
            weights[selected_edge].data.fill_(0)
        else:
            proj_mask[selected_op] = 0

        weights[selected_edge] = weights[selected_edge] * proj_mask

        self.removed_projected_weights = {
            "normal": None,
            "reduce": None,
        }
        self.removal_cell_type = cell_type
        self.removed_projected_weights[cell_type] = weights

    def mark_projected_op(
        self, eid: int, opid: int, cell_type: Literal["normal", "reduce"]
    ) -> None:
        self.proj_weights[cell_type][eid][opid] = 1
        self.candidate_flags[cell_type][eid] = False

    def mark_projected_edges(
        self, nid: int, eids: list[int], cell_type: Literal["normal", "reduce"]
    ) -> None:
        for eid in self.nid2eids[nid]:
            if eid not in eids:  # not top2
                self.proj_weights[cell_type][eid].data.fill_(0)
        self.nid2selected_eids[cell_type][nid] = copy.deepcopy(eids)
        self.candidate_flags_edge[cell_type][nid] = False

    def get_projected_weights(
        self, cell_type: Literal["normal", "reduce"]
    ) -> torch.Tensor:
        assert cell_type in ["normal", "reduce"]

        if self.projection_evaluation and self.removal_cell_type == cell_type:
            return self.removed_projected_weights[cell_type]

        if self.is_arch_attention_enabled:
            alphas_normal, alphas_reduce = self._compute_arch_attention(
                self.alphas_normal, self.alphas_reduce
            )
        else:
            alphas_normal = self.alphas_normal
            alphas_reduce = self.alphas_reduce

        if cell_type == "normal":
            weights = torch.softmax(alphas_normal, dim=-1)
        else:
            weights = torch.softmax(alphas_reduce, dim=-1)

        ## proj op
        for eid in range(self.num_edges):
            if not self.candidate_flags[cell_type][eid]:
                weights[eid].data.copy_(self.proj_weights[cell_type][eid])

        ## proj edge
        for nid in self.nid2eids:
            if not self.candidate_flags_edge[cell_type][nid]:  ## projected node
                for eid in self.nid2eids[nid]:
                    if eid not in self.nid2selected_eids[cell_type][nid]:
                        weights[eid].data.copy_(self.proj_weights[cell_type][eid])

        return weights

    def _compute_arch_attention(
        self, normal_alphas: nn.Parameter, reduce_alphas: nn.Parameter
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_arch_params = torch.concat((normal_alphas, reduce_alphas))
        all_arch_attn, _ = self.multihead_attention(
            all_arch_params, all_arch_params, all_arch_params, need_weights=False
        )
        num_edges_normal = normal_alphas.shape[0]
        attn_normal = all_arch_attn[:num_edges_normal]
        attn_reduce = all_arch_attn[num_edges_normal:]

        return attn_normal, attn_reduce

    def create_new_cell(self, pos: int) -> Cell:
        prev_cell = self.cells[pos - 1]
        new_cell = copy.deepcopy(prev_cell)

        new_cell.C = prev_cell.C
        new_cell.C_prev = prev_cell.C * self._multiplier
        new_cell.C_prev_prev = prev_cell.C_prev

        # reduction->normal
        if prev_cell.reduction is True:
            # update cell type
            new_cell.reduction = False
            # update stride
            new_cell.change_op_stride_size(new_stride=1, is_reduction_cell=False)
            # update skip connection
            new_cell.change_skip_connection_type(reduction=False)
            # have FactorizedReduce as pre0
            new_cell.change_preprocessing0_type(prev_reduction=True)

        # normal->reduction
        elif self._layers == 3:
            # update cell type
            new_cell.reduction = True
            # update skip connection
            new_cell.change_skip_connection_type(reduction=True)
            # update channel size
            new_cell.C = prev_cell.C * 2  # holds when reduction or normal
            # update stride
            new_cell.change_op_stride_size(new_stride=2, is_reduction_cell=True)
            # have ReLUConvBN as pre0
            new_cell.change_preprocessing0_type()

        # reduction-normal->noraml
        elif self._layers in (6, 7):
            # have ReLUConvBN as pre0
            new_cell.change_preprocessing0_type()

        # normal->normal-reduction-normal-reduction-normal
        elif self._layers == 5:
            # update channel size of reduction at pos=2
            post_cell = self.cells[1]
            post_cell.C_prev = new_cell.C * self._multiplier
            post_cell.C_prev_prev = new_cell.C_prev
            post_cell.change_op_channel_size(new_cell=False)

        # else:
        # normal->normal->normal

        # change
        if self._layers < 9:
            new_cell.change_op_channel_size()

        return new_cell

    def insert_new_cells(self, num_of_cells: int) -> None:
        for i in range(self._layers, self._layers + num_of_cells):
            pos = get_pos_new_cell_darts(i)
            self.cells.insert(pos, self.create_new_cell(pos))
            self._layers += 1

    def increase_channel_size(
        self,
        k: float | None = None,
        num_channels_to_add: int | None = None,
    ) -> None:
        for cell in self.cells:
            cell.increase_channel_size(k=k, num_channels_to_add=num_channels_to_add)


def preserve_grads(m: nn.Module) -> None:
    ignored_modules = (
        OperationBlock,
        OperationChoices,
        Cell,
        MixedOp,
        Network,
    )

    preserve_gradients_in_module(m, ignored_modules, OLES_OPS)
