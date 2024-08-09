from __future__ import annotations

import copy
import math
from typing import Literal
import warnings

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from confopt.searchspace.common.mixop import OperationBlock, OperationChoices
from confopt.utils import (
    AverageMeter,
    calc_layer_alignment_score,
    freeze,
    preserve_gradients_in_module,
    set_ops_to_prune,
)
from confopt.utils.normalize_params import normalize_params

from .genotypes import BABY_PRIMITIVES, PRIMITIVES, DARTSGenotype
from .model import NetworkCIFAR, NetworkImageNet
from .operations import OLES_OPS, OPS, FactorizedReduce, ReLUConvBN

NUM_CIFAR_CLASSES = 10
NUM_CIFAR100_CLASSES = 100
NUM_IMAGENET_CLASSES = 120
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MixedOp(nn.Module):
    def __init__(self, C: int, stride: int, primitives: list[str] = PRIMITIVES):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C, stride, False)
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

        Attributes:
            preprocess0(nn.Module): Preprocess for input from previous-previous cell.
            preprocess1(nn.Module): Preprocess for input from previous cell.
            _ops(nn.ModuleList): List of operations in the cell.
            reduction(bool): Whether the cell is a reduction cell (True) or
                             a normal cell (False).
        """
        super().__init__()
        self.reduction = reduction

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
                ops = MixedOp(C, stride, primitives)._ops
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
            if i in [layers // 3, 2 * layers // 3]:
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
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.weights: dict[str, list[torch.Tensor]] = {}

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

    def retain_weight_grad(
        self, weights: torch.Tensor, weight_type: Literal["reduce", "normal"]
    ) -> None:
        # Retain grads for calculating layer alignment score
        assert weight_type in ["reduce", "normal"]
        if self.training:
            if isinstance(weights, list):
                for weight in weights:
                    weight.retain_grad()
            else:
                assert isinstance(weights, torch.Tensor)
                weights.retain_grad()
        self.weights[weight_type].append(weights)

    def remove_pruned_alphas(
        self, weights_normal: torch.Tensor, weights_reduce: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            self.mask is not None
        ), "This function requires a prior call to prune function"

        def _prune_alpha(weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            weight = weight[mask]
            weight = weight.reshape(mask.shape[0], mask[0].sum())
            return weight

        weights_normal = _prune_alpha(weights_normal, self.mask[0])
        weights_reduce = _prune_alpha(weights_reduce, self.mask[1])

        return weights_normal, weights_reduce

    def restore_pruned_alpha_shape(
        self, weights_normal: torch.Tensor, weights_reduce: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            self.mask is not None
        ), "This function requires a prior call to prune function"

        def _restore_weight_shape(
            weight: list[torch.Tensor] | torch.Tensor,
            reference_weight: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            if isinstance(weight, list):
                weight = torch.stack(weight)
            weight_full = torch.zeros_like(reference_weight)
            weight_full[mask] = weight.view(-1)
            return weight_full

        weights_normal = _restore_weight_shape(
            weights_normal, self.alphas_normal, self.mask[0]
        )
        weights_reduce = _restore_weight_shape(
            weights_reduce, self.alphas_reduce, self.mask[1]
        )

        return weights_normal, weights_reduce

    def sample_with_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        weights_normal_to_sample = self.alphas_normal
        weights_reduce_to_sample = self.alphas_reduce

        if self.mask is not None:
            # The shape of weights will change
            # If num_keep was 6
            # The shape should be (14, 6)
            (
                weights_normal_to_sample,
                weights_reduce_to_sample,
            ) = self.remove_pruned_alphas(
                weights_normal_to_sample, weights_reduce_to_sample
            )

        weights_normal = self.sample(weights_normal_to_sample)
        weights_reduce = self.sample(weights_reduce_to_sample)

        if self.mask is not None:
            # Revert back to original shape of (14, 8)
            weights_normal, weights_reduce = self.restore_pruned_alpha_shape(
                weights_normal, weights_reduce
            )

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
        if self.edge_normalization:
            return self.edge_normalization_forward(x)

        s0 = s1 = self.stem(x)
        self.weights["normal"] = []
        self.weights["reduce"] = []

        if self.projection_mode:
            weights_normal, weights_reduce = (
                self.get_projected_weights("normal"),
                self.get_projected_weights("reduce"),
            )
        else:
            weights_normal, weights_reduce = self.sample_with_mask()

        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
                self.retain_weight_grad(weights, "reduce")
                if self.mask is not None:
                    weights = normalize_params(weights, self.mask[1])
            else:
                weights = weights_normal
                self.retain_weight_grad(weights, "normal")
                if self.mask is not None:
                    weights = normalize_params(weights, self.mask[0])
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
        self.weights["normal"] = []
        self.weights["reduce"] = []

        if self.projection_mode:
            weights_normal, weights_reduce = self.get_projected_weights(
                "normal"
            ), self.get_projected_weights("reduce")
        else:
            weights_normal, weights_reduce = self.sample_with_mask()

        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
                self.retain_weight_grad(weights, "reduce")
                if self.mask is not None:
                    weights = normalize_params(weights, self.mask[1])
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
                weights = weights_normal
                self.retain_weight_grad(weights, "normal")
                if self.mask is not None:
                    weights = normalize_params(weights, self.mask[0])
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
        num_ops = len(self.primitives)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops).to(DEVICE))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops).to(DEVICE))
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

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = DARTSGenotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype

    def get_arch_grads(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        def get_grads(alphas_list: list[torch.Tensor]) -> list[torch.Tensor]:
            grads = []
            for alphas in alphas_list:
                if isinstance(alphas, list):
                    alphas_grads = [
                        alpha.grad.data.clone().detach() for alpha in alphas
                    ]
                    grads.append(torch.stack(alphas_grads).detach().reshape(-1))
                else:
                    grads.append(alphas.grad.data.clone().detach().reshape(-1))

            return grads

        grads_normal = get_grads(self.weights["normal"])
        grads_reduce = get_grads(self.weights["reduce"])
        return grads_normal, grads_reduce

    def _get_mean_layer_alignment_score(self) -> tuple[float, float]:
        grads_normal, grads_reduce = self.get_arch_grads()
        mean_score_normal = calc_layer_alignment_score(grads_normal)
        mean_score_reduce = calc_layer_alignment_score(grads_reduce)

        if math.isnan(mean_score_normal):
            mean_score_normal = 0
        if math.isnan(mean_score_reduce):
            mean_score_reduce = 0
        return mean_score_normal, mean_score_reduce

    def prune(self, num_keep: int) -> None:
        """Discretize architecture parameters to enforce sparsity.

        Args:
            num_keep (float): Number of operations to keep.
        """
        self.mask = []

        top_k = num_keep

        # Calculate masks
        for i, p in enumerate(self._arch_parameters):
            data_to_sort = p.data.clone()
            if len(self.last_mask) != 0:
                last_mask = self.last_mask[i]
                temp = float("-inf") * torch.ones_like(data_to_sort)
                data_to_sort[~last_mask] = temp[~last_mask]

            sorted_arch_params, _ = torch.sort(data_to_sort, dim=1, descending=True)
            thresholds = sorted_arch_params[:, top_k - 1].unsqueeze(1)
            mask = data_to_sort >= thresholds
            self.mask.append(mask)

        for cell in self.cells:
            if cell.reduction:
                cell.prune_ops(self.mask[1])
            else:
                cell.prune_ops(self.mask[0])

        self.last_mask = self.mask

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
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.num_edges = k
        self.num_ops = len(self.alphas_normal[0])
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
        self.nid2eids = {0: [2, 3, 4], 1: [5, 6, 7, 8], 2: [9, 10, 11, 12, 13]}
        self.nid2selected_eids = {  # type: ignore
            "normal": {0: [], 1: [], 2: []},
            "reduce": {0: [], 1: [], 2: []},
        }
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

        if cell_type == "normal":
            weights = torch.softmax(self.alphas_normal, dim=-1)
        else:
            weights = torch.softmax(self.alphas_reduce, dim=-1)

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


def preserve_grads(m: nn.Module) -> None:
    ignored_modules = (
        OperationBlock,
        OperationChoices,
        Cell,
        MixedOp,
        Network,
    )

    preserve_gradients_in_module(m, ignored_modules, OLES_OPS)


# TODO: break function from OLES paper to have less branching.
def check_grads_cosine(m: nn.Module, oles: bool = False) -> None:  # noqa: C901
    if (
        isinstance(m, (OperationBlock, OperationChoices, Cell, MixedOp, Network))
        or not isinstance(m, tuple(OLES_OPS))
        or not hasattr(m, "pre_grads")
        or not m.pre_grads
    ):
        return

    i = 0
    true_i = 0
    temp = 0

    for param in m.parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.detach().cpu()
            if len(g) != 0:
                temp += torch.cosine_similarity(g, m.pre_grads[i], dim=0).mean()
                true_i += 1
            i += 1

    m.pre_grads.clear()
    if true_i == 0:
        return
    sim_avg = temp / true_i

    if not hasattr(m, "avg"):
        m.avg = 0
    m.avg += sim_avg

    if not hasattr(m, "running_sim"):
        m.running_sim = AverageMeter()
    m.running_sim.update(sim_avg)

    if not hasattr(m, "count"):
        m.count = 0

    if m.count == 20:
        if m.avg / m.count < 0.4 and oles:
            freeze(m)
        m.count = 0
        m.avg = 0
    else:
        m.count += 1
