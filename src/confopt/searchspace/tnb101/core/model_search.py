from __future__ import annotations

from copy import deepcopy
import math
from typing import Callable

import torch
from torch import nn

from confopt.searchspace.common import OperationBlock, OperationChoices
from confopt.utils import (
    calc_layer_alignment_score,
    preserve_gradients_in_module,
    prune,
    set_ops_to_prune,
)
from confopt.utils.normalize_params import normalize_params

from . import operations as ops
from .genotypes import TNB101Genotype
from .operations import OLES_OPS, OPS, TRANS_NAS_BENCH_101

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TNB101SearchModel(nn.Module):
    def __init__(
        self,
        C: int = 16,
        stride: int = 1,
        max_nodes: int = 4,
        num_classes: int = 10,
        op_names: list[str] = TRANS_NAS_BENCH_101,
        affine: bool = False,
        track_running_stats: bool = False,
        dataset: str = "cifar10",
        edge_normalization: bool = False,
    ):
        """Initialize a TransNasBench-101 network consisting of one cell
        Args:
            C_in: in channel
            C_out: out channel
            stride: 1 or 2
            max_nodes: total amount of nodes in one cell
            num_classes: classes
            op_names: operations for cell structure
            affine: used for torch.nn.BatchNorm2D
            track_running_stats: used for torch.nn.BatchNorm2D.
        """
        super().__init__()
        assert stride in (1, 2), f"invalid stride {stride}"

        self.C = C
        self.stride = stride
        self.edge_normalization = edge_normalization

        self.op_names = deepcopy(op_names)
        self.max_nodes = max_nodes
        self.n_modules = 5
        self.blocks_per_module = [2] * self.n_modules

        # initialize other arguments for intializing a new model
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.dataset = dataset

        self.module_stages = [
            "r_stage_1",
            "n_stage_1",
            "r_stage_2",
            "n_stage_2",
            "r_stage_3",
        ]

        self.cells = nn.ModuleList()
        C_in, C_out = C, C
        for idx, stage in enumerate(self.module_stages):
            for i in range(self.blocks_per_module[idx]):
                downsample = self._is_reduction_stage(stage) and i % 2 == 0
                if downsample:
                    C_out *= 2
                cell = TNB101SearchCell(
                    C_in,
                    C_out,
                    stride,
                    max_nodes,
                    op_names,
                    affine,
                    track_running_stats,
                    downsample,
                ).to(DEVICE)
                self.cells.append(cell)
                C_in = C_out
        self.num_edge = len(self.cells[0].edges)

        if dataset == "jigsaw":
            self.num_classes = 1000
        elif dataset == "class_object":
            self.num_classes = 100
        elif dataset == "class_scene":
            self.num_classes = 63
        else:
            self.num_classes = num_classes

        self.stem = self._get_stem_for_task(dataset)
        self.decoder = self._get_decoder_for_task(dataset, C_out)
        self.op_names = deepcopy(op_names)
        self.max_nodes = max_nodes

        self.lastact = nn.Sequential(nn.BatchNorm2d(num_classes), nn.ReLU())
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.num_classes, self.num_classes)

        self.mask: None | torch.Tensor = None
        self._arch_parameters = nn.Parameter(
            1e-3 * torch.randn(self.num_edge, len(op_names))  # type: ignore
        )
        self._beta_parameters = nn.Parameter(1e-3 * torch.randn(self.num_edge))

        self.weights_grad: list[torch.Tensor] = []
        self.grad_hook_handlers: list[torch.utils.hooks.RemovableHandle] = []

        # Multi-head attention for architectural parameters
        self.is_arch_attention_enabled = False  # disabled by default
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=len(self.op_names), num_heads=1
        )

    def arch_parameters(self) -> nn.Parameter:
        return self._arch_parameters

    def beta_parameters(self) -> nn.Parameter:
        return self._beta_parameters

    def sample(self, alphas: torch.Tensor) -> torch.Tensor:
        # Replace this function on the fly to change the sampling method
        return torch.nn.functional.softmax(alphas, dim=-1)

    def sample_weights(self) -> torch.Tensor:
        weights_to_sample = self._arch_parameters

        if self.is_arch_attention_enabled:
            weights_to_sample = self._compute_arch_attention(weights_to_sample)

        weights = self.sample(weights_to_sample)

        return weights

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.reset_hooks()

        if self._arch_parameters is None:
            return self.discrete_model_forward(inputs)

        if self.edge_normalization:
            return self.edge_normalization_forward(inputs)

        self.weights_grad = []

        # alphas = self.sample(self._arch_parameters)
        alphas = self.sample_weights()

        if self.mask is not None:
            alphas = normalize_params(alphas, self.mask)

        self.sampled_weights = [alphas]

        feature = self.stem(inputs)

        for cell in self.cells:
            weights = alphas.clone()
            self.save_weight_grads(weights)
            feature = cell(feature, weights)

        out = self.decoder(feature)

        # out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        # logits = self.classifier(out)

        return out, out

    def discrete_model_forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.stem(inputs)
        for _i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.decoder(feature)

        # out = self.global_pooling(out)
        out = out.view(out.size(0), -1)

        return out, out

    def edge_normalization_forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # alphas = self.sample(self._arch_parameters)
        self.weights_grad = []

        alphas = self.sample_weights()

        if self.mask is not None:
            alphas = normalize_params(alphas, self.mask)

        self.sampled_weights = [alphas]

        feature = self.stem(inputs)
        for cell in self.cells:
            betas = torch.empty((0,)).to(self._arch_parameters.device)
            for v in range(1, self.max_nodes):
                idx_nodes = []
                for u in range(v):
                    node_str = f"{v}<-{u}"
                    idx_nodes.append(cell.edge2index[node_str])
                beta_node_v = nn.functional.softmax(
                    self._beta_parameters[idx_nodes], dim=-1
                )
                betas = torch.cat([betas, beta_node_v], dim=0)
            weights = alphas.clone()
            self.save_weight_grads(weights)
            feature = cell(feature, alphas, betas)

        out = self.decoder(feature)

        # out = self.global_pooling(out)
        out = out.view(out.size(0), -1)

        return out, out

    def _get_stem_for_task(self, task: str) -> nn.Module:
        if task == "jigsaw":
            return ops.StemJigsaw(C_out=self.C)
        if task in ["class_object", "class_scene"]:
            return ops.Stem(C_out=self.C)
        if task == "autoencoder":
            return ops.Stem(C_out=self.C)
        return ops.Stem(C_in=3, C_out=self.C)

    def _get_decoder_for_task(self, task: str, n_channels: int) -> nn.Module:
        if task == "jigsaw":
            return ops.SequentialJigsaw(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels * 9, self.num_classes),
            )
        if task in ["class_object", "class_scene"]:
            return ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels, self.num_classes),
            )
        if task == "autoencoder":
            if self.use_small_model:
                return ops.GenerativeDecoder((64, 32), (256, 2048))  # Short
            return ops.GenerativeDecoder((512, 32), (512, 2048))  # Full TNB

        return ops.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_channels, self.num_classes),
        )

    def _is_reduction_stage(self, stage: str) -> bool:
        return "r_stage" in stage

    def prune(self, prune_fraction: float) -> None:
        """Discretize architecture parameters to enforce sparsity.

        Args:
            prune_fraction (float): Number of operations to keep.
        """
        assert prune_fraction < 1, "Prune fraction should be less than 1"
        assert prune_fraction >= 0, "Prune fraction greater or equal to 0"

        num_ops = len(self.op_names)
        top_k = num_ops - int(num_ops * prune_fraction)

        self.mask = prune(self._arch_parameters, top_k, self.mask)

        for cell in self.cells:
            cell.prune_ops(self.mask)

    def _discretize(self) -> TNB101SearchModel:
        dicrete_model = TNB101SearchModel(
            C=self.C,
            stride=self.stride,
            max_nodes=self.max_nodes,
            num_classes=self.num_classes,
            op_names=self.op_names,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            dataset=self.dataset,
            edge_normalization=False,
        ).to(next(self.parameters()).device)

        if self.is_arch_attention_enabled:
            arch_parameters = self._compute_arch_attention(self._arch_parameters)
        else:
            arch_parameters = self._arch_parameters

        for cell in dicrete_model.cells:
            cell._discretize(arch_parameters)
        dicrete_model._arch_parameters = None

        return dicrete_model

    def model_weight_parameters(self) -> list[nn.Parameter]:
        params = set(self.parameters())
        params -= set(self._beta_parameters)
        if self._arch_parameters is not None:
            params -= set(self._arch_parameters)
        return list(params)

    def _compute_arch_attention(self, alphas: nn.Parameter) -> torch.Tensor:
        attn_alphas, _ = self.multihead_attention(alphas, alphas, alphas)
        return attn_alphas

    def genotype(self) -> TNB101Genotype:
        node_edge_dict: dict[int, list[tuple[str, int]]] = {}
        op_idx_list = []

        if self.is_arch_attention_enabled:
            arch_parameters = self._compute_arch_attention(self._arch_parameters)
        else:
            arch_parameters = self._arch_parameters
        edge2index = self.cells[0].edge2index

        for i in range(1, self.max_nodes):
            for j in range(i):
                node_str = f"{i}<-{j}"
                max_idx = torch.argmax(arch_parameters[edge2index[node_str]])
                if i in node_edge_dict:
                    node_edge_dict[i].append((self.op_names[max_idx], j))
                else:
                    node_edge_dict[i] = [(self.op_names[max_idx], j)]
                op_idx_list.append(max_idx.item())

        return TNB101Genotype(node_edge_dict=node_edge_dict, op_idx_list=op_idx_list)

    ### Layer alignment score support  methods ###
    def reset_hooks(self) -> None:
        for hook in self.grad_hook_handlers:
            hook.remove()

        self.grad_hook_handlers = []

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
        grad_hook = weights.register_hook(self.save_gradient())
        self.grad_hook_handlers.append(grad_hook)

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

    ### End of Layer alignment score support methods ###


class TNB101SearchCell(nn.Module):
    expansion = 1

    def __init__(
        self,
        C_in: int = 16,
        C_out: int = 16,
        stride: int = 1,
        max_nodes: int = 4,
        op_names: list[str] = TRANS_NAS_BENCH_101,
        affine: bool = True,
        track_running_stats: bool = True,
        downsample: bool = True,
    ):
        """Initialize a TransNasBench-101 cell
        Args:
            C_in: in channel
            C_out: out channel
            stride: 1 or 2
            max_nodes: total amount of nodes in one cell
            op_names: operations for cell structure
            affine: used for torch.nn.BatchNorm2D
            track_running_stats: used for torch.nn.BatchNorm2D.
        """
        super().__init__()
        assert stride in (1, 2), f"invalid stride {stride}"

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = f"{i}<-{j}"
                if j == 0:
                    stride = 2 if downsample else 1
                    xlists = nn.ModuleList(
                        [
                            OPS[op_name](
                                C_in,
                                C_out,
                                stride,
                                affine,
                                track_running_stats,
                            )  # type: ignore
                            for op_name in op_names
                        ]
                    )
                else:
                    xlists = nn.ModuleList(
                        [
                            OPS[op_name](
                                C_out, C_out, 1, affine, track_running_stats
                            )  # type: ignore
                            for op_name in op_names
                        ]
                    )
                self.edges[node_str] = OperationChoices(
                    ops=xlists, is_reduction_cell=downsample
                )
        self.edge_keys = sorted(self.edges.keys())
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges: int = len(self.edges)

    def forward(
        self,
        inputs: torch.Tensor,
        alphas: torch.Tensor | None = None,
        betas: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if alphas is None:
            return self.discrete_model_forward(inputs)
        if betas is not None:
            return self.edge_normalization_forward(inputs, alphas, betas)

        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                weights = alphas[self.edge2index[node_str]]
                inter_nodes.append(self.edges[node_str](nodes[j], weights))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    def discrete_model_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                inter_nodes.append(self.edges[node_str](nodes[j]))
            nodes.append(sum(inter_nodes))  # type: ignore
        return nodes[-1]

    def edge_normalization_forward(
        self,
        inputs: torch.Tensor,
        alphas: list[torch.Tensor],
        betas: list[torch.Tensor],
    ) -> torch.Tensor:
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                weights = alphas[self.edge2index[node_str]]
                beta_weights = betas[self.edge2index[node_str]]
                inter_nodes.append(
                    beta_weights * self.edges[node_str](nodes[j], weights)
                )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    def _discretize(self, alphas: list[torch.Tensor]) -> None:
        for i in range(1, self.max_nodes):
            for j in range(i):
                node_str = f"{i}<-{j}"
                max_idx = torch.argmax(alphas[self.edge2index[node_str]], dim=-1)
                self.edges[node_str] = (self.edges[node_str].ops)[  # type: ignore
                    max_idx
                ]

    def prune_ops(self, mask: torch.Tensor) -> None:
        for i in range(1, self.max_nodes):
            for j in range(i):
                node_str = f"{i}<-{j}"
                edge_mask = mask[self.edge2index[node_str]]
                set_ops_to_prune(self.edges[node_str], edge_mask)


def preserve_grads(m: nn.Module) -> None:
    ignored_modules = (
        OperationBlock,
        OperationChoices,
        TNB101SearchCell,
        TNB101SearchModel,
    )

    preserve_gradients_in_module(m, ignored_modules, OLES_OPS)
