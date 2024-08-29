##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
from __future__ import annotations

from copy import deepcopy
import math
from typing import Callable

import torch
from torch import nn

from confopt.searchspace.common.mixop import OperationBlock, OperationChoices
from confopt.utils import (
    calc_layer_alignment_score,
    preserve_gradients_in_module,
    prune,
)
from confopt.utils.normalize_params import normalize_params

from .cells import NAS201SearchCell as SearchCell
from .genotypes import Structure
from .model import NASBench201Model
from .operations import NAS_BENCH_201, OLES_OPS, ResNetBasicblock

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NB201SearchModel(nn.Module):
    """Implementation of Nasbench201 search space.

    Args:
        C (int, optional): Number of channels. Defaults to 16.
        N (int, optional): Number of layers. Defaults to 5.
        max_nodes (int, optional): Maximum number of nodes in a cell. Defaults to 4.
        num_classes (int, optional): Number of classes of the dataset. Defaults to 10.
        steps (int, optional): Number of steps. Defaults to 3.
        search_space (list[str], optional): List of search space options. Defaults to
        NAS_BENCH_201.
        affine (bool, optional): Whether to use affine transformations in BatchNorm in
        cells. Defaults to False.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm in cells. Defaults to False.
        edge_normalization (bool, optional): Whether to enable edge normalization for
        partial connection. Defaults to False.
        with one operation on each edge

    Attributes:
        stem (nn.Sequential): Stem network composed of Conv2d and BatchNorm2d layers.
        cells (nn.ModuleList): List of cells in the search space.
        op_names (list[str]): List of operation names.
        edge2index (dict[str, int]): Dictionary mapping edge names to indices.
        lastact (nn.Sequential): Sequential layer consisting of BatchNorm2d and ReLU.
        global_pooling (nn.AdaptiveAvgPool2d): Global pooling layer.
        classifier (nn.Linear): Linear classifier layer.
        arch_parameters (nn.Parameter): Parameter for architecture alpha values.
        beta_parameters (nn.Parameter): Parameter for beta values.

    Note:
        This is a custom neural network model with various hyperparameters and
        architectural choices.
    """

    def __init__(
        self,
        C: int = 16,
        N: int = 5,
        max_nodes: int = 4,
        num_classes: int = 10,
        steps: int = 3,
        search_space: list[str] = NAS_BENCH_201,
        affine: bool = False,
        track_running_stats: bool = False,
        edge_normalization: bool = False,
        k: int = 1,
    ):
        super().__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self._steps = steps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.edge_normalization = edge_normalization
        self.mask: None | torch.Tensor = None
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for _index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(  # type: ignore
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                    k=k,
                )

                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges
                    ), f"invalid {num_edge} vs. {cell.num_edges}."
                    assert (
                        edge2index == cell.edge2index
                    ), f"invalid {num_edge} vs. {cell.num_edges}."
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index: dict[str, int] = edge2index  # type: ignore
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU())
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_parameters: None | nn.Parameter = nn.Parameter(
            1e-3 * torch.randn(num_edge, len(search_space))  # type: ignore
        )
        self.beta_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge)  # type: ignore
        )
        self.weights_grad: list[torch.Tensor] = []
        self.num_edges = num_edge
        self.num_nodes = max_nodes - 1
        self.num_ops = len(search_space)
        self._initialize_projection_params()

        # Multi-head attention for architectural parameters
        self.is_arch_attention_enabled = False  # disabled by default
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=len(self.op_names), num_heads=1
        )

    def get_weights(self) -> list[nn.Parameter]:
        """Get a list of learnable parameters in the model. (does not include alpha or
        beta parameters).

        Returns:
            list[nn.Parameter]: A list of learnable parameters in the model, including
            stem, cells, lastact, global_pooling, and classifier parameters.
        """
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self) -> list[torch.Tensor]:
        """Get a list containing the architecture parameters or alphas.

        Returns:
            list[torch.Tensor]: A list containing the architecture parameters, such as
            alpha values.
        """
        return [self.arch_parameters]  # type: ignore

    def get_betas(self) -> list[torch.Tensor]:
        """Get a list containing the beta parameters of partial connection used for
        edge normalization.

        Returns:
            list[torch.Tensor]: A list containing the beta parameters for the model.
        """
        return [self.beta_parameters]

    def show_alphas(self) -> str:
        """Gets a string representation of the architecture parameters
        (alphas).

        Returns:
            str: A string representing the architecture parameters after softmax
            operation.
        """
        with torch.no_grad():
            return "arch-parameters :\n{:}".format(
                nn.functional.softmax(
                    self.arch_parameters, dim=-1
                ).cpu()  # type: ignore
            )

    def show_betas(self) -> str:
        """Gets a string representation of the beta parameters.

        Returns:
            str: A string representing the beta parameters after softmax operation.
        """
        with torch.no_grad():
            return "beta-parameters:\n{:}".format(
                nn.functional.softmax(self.beta_parameters, dim=-1).cpu()
            )

    def get_message(self) -> str:
        """Gets a message describing the model and its cells.

        Returns:
            str: A string message containing information about the model and its cells.
        """
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += f"\n {i:02d}/{len(self.cells):02d} :: {cell.extra_repr()}"
        return string

    def extra_repr(self) -> str:
        """Return a string containing extra information about the model.

        Returns:
            str: A string representation containing information about the model's class
            name, number of channels (C), maximum nodes (Max-Nodes), number of layers
            (N), and number of cells (L).
        """
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def genotype(self) -> Structure:
        """Get the genotype of the model, representing the architecture.

        Returns:
            Structure: An object representing the genotype of the model, which describes
            the architectural choices in terms of operations and connections between
            nodes.
        """
        genotypes = []

        if self.is_arch_attention_enabled:
            arch_parameters = self._compute_arch_attention(self.arch_parameters)
        else:
            arch_parameters = self.arch_parameters

        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                with torch.no_grad():
                    weights = arch_parameters[self.edge2index[node_str]]  # type: ignore
                    # betas = self.beta_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]  # type: ignore
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def sample(self, alphas: torch.Tensor) -> torch.Tensor:
        # Replace this function on the fly to change the sampling method
        return torch.nn.functional.softmax(alphas, dim=-1)

    def sample_weights(self) -> torch.Tensor:
        if self.projection_mode:
            weights = self.get_projected_weights()
            return weights

        weights_to_sample = self.arch_parameters

        if self.is_arch_attention_enabled:
            weights_to_sample = self._compute_arch_attention(weights_to_sample)

        weights = self.sample(weights_to_sample)

        if self.mask is not None:
            weights = normalize_params(weights, self.mask)

        return weights

    def save_gradient(self) -> Callable:
        def hook(grad: torch.Tensor) -> None:
            self.weights_grad.append(grad)

        return hook

    def save_weight_grads(
        self,
        weights: torch.Tensor,
    ) -> None:
        if not self.training:
            return
        weights.register_hook(self.save_gradient())

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor to the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - The output tensor after the forward pass.
            - The logits tensor produced by the model.
        """
        if self.edge_normalization:
            return self.edge_normalization_forward(inputs)

        self.weights_grad = []

        weights = self.sample_weights()

        feature = self.stem(inputs)
        for _i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                alphas = weights.clone()
                self.save_weight_grads(alphas)
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

    def edge_normalization_forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.weights_grad = []

        weights = self.sample_weights()

        feature = self.stem(inputs)
        for _i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                alphas = weights.clone()
                self.save_weight_grads(alphas)
                betas = torch.empty((0,)).to(alphas[0].device)  # type: ignore
                for v in range(1, self.max_nodes):
                    idx_nodes = []
                    for u in range(v):
                        node_str = f"{v}<-{u}"
                        idx_nodes.append(cell.edge2index[node_str])
                    beta_node_v = nn.functional.softmax(
                        self.beta_parameters[idx_nodes], dim=-1
                    )
                    betas = torch.cat([betas, beta_node_v], dim=0)

                feature = cell(feature, alphas, betas)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

    def get_arch_grads(self, only_first_and_last: bool = False) -> list[torch.Tensor]:
        grads = []
        if only_first_and_last:
            grads.append(self.weights_grad[0].reshape(-1))
            grads.append(self.weights_grad[1].reshape(-1))
        else:
            for alphas in self.weights_grad:
                grads.append(alphas.reshape(-1))

        return grads

    def get_mean_layer_alignment_score(
        self, only_first_and_last: bool = False
    ) -> float:
        grads = self.get_arch_grads(only_first_and_last)
        mean_score = calc_layer_alignment_score(grads)

        if math.isnan(mean_score):
            mean_score = 0

        return mean_score

    def prune(self, prune_fraction: float) -> None:
        """Masks architecture parameters to enforce sparsity.

        Args:
            prune_fraction (float): Number of operations to keep.
        """
        assert prune_fraction < 1, "Prune fraction should be less than 1"
        assert prune_fraction >= 0, "Prune fraction greater or equal to 0"

        num_ops = len(self.op_names)
        top_k = num_ops - int(num_ops * prune_fraction)

        self.mask = prune(self.arch_parameters, top_k, self.mask)

        for cell in self.cells:
            if isinstance(cell, SearchCell):
                cell.prune_ops(self.mask)

    def _discretize(self) -> NASBench201Model:
        genotype = self.genotype()

        discrete_model = NASBench201Model(
            C=self._C,
            N=self._layerN,
            genotype=genotype,
            num_classes=self.classifier.out_features,
        ).to(next(self.parameters()).device)

        return discrete_model

    def model_weight_parameters(self) -> list[nn.Parameter]:
        params = set(self.parameters())
        params -= set(self.beta_parameters)
        if self.arch_parameters is not None:
            params -= set(self.arch_parameters)
        return list(params)

    def _initialize_projection_params(self) -> None:
        self.candidate_flags = torch.tensor(
            self.num_edges * [True],  # type: ignore
            requires_grad=False,
            dtype=torch.bool,
        ).to(DEVICE)
        self.proj_weights = torch.zeros_like(self.arch_parameters)

        self.projection_mode = False
        self.projection_evaluation = False
        self.removed_projected_weights = None

    def mark_projected_op(self, eid: int, opid: int) -> None:
        self.proj_weights[eid][opid] = 1
        self.candidate_flags[eid] = False

    def get_projected_weights(self) -> torch.Tensor:
        if self.projection_evaluation:
            return self.removed_projected_weights

        if self.is_arch_attention_enabled:
            arch_parameters = self._compute_arch_attention(self.arch_parameters)
        else:
            arch_parameters = self.arch_parameters

        weights = torch.softmax(arch_parameters, dim=-1)
        for eid in range(len(arch_parameters)):  # type: ignore
            if not self.candidate_flags[eid]:
                weights[eid].data.copy_(self.proj_weights[eid])

        return weights

    def remove_from_projected_weights(
        self, selected_edge: int, selected_op: int
    ) -> None:
        weights = self.get_projected_weights()
        proj_mask = torch.ones_like(weights[selected_edge])
        proj_mask[selected_op] = 0

        weights[selected_edge] = weights[selected_edge] * proj_mask
        self.removed_projected_weights = weights

    def _compute_arch_attention(self, alphas: nn.Parameter) -> torch.Tensor:
        attn_alphas, _ = self.multihead_attention(alphas, alphas, alphas)
        return attn_alphas


def preserve_grads(m: nn.Module) -> None:
    ignored_modules = (
        OperationBlock,
        OperationChoices,
        SearchCell,
        NB201SearchModel,
    )

    preserve_gradients_in_module(m, ignored_modules, OLES_OPS)
