from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from confopt.searchspace.common import OperationChoices
from confopt.utils.normalize_params import normalize_params

from . import operations as ops
from .operations import OPS, TRANS_NAS_BENCH_101

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
        assert stride == 1 or stride == 2, f"invalid stride {stride}"

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

    def arch_parameters(self) -> nn.Parameter:
        return self._arch_parameters

    def beta_parameters(self) -> nn.Parameter:
        return self._beta_parameters

    def sample(self, alphas: torch.Tensor) -> torch.Tensor:
        # Replace this function on the fly to change the sampling method
        return torch.nn.functional.softmax(alphas, dim=-1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._arch_parameters is None:
            return self.discrete_model_forward(inputs)

        if self.edge_normalization:
            return self.edge_normalization_forward(inputs)

        alphas = self.sample(self._arch_parameters)

        if self.mask is not None:
            alphas = normalize_params(alphas, self.mask)

        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature, alphas)

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
        alphas = self.sample(self._arch_parameters)

        if self.mask is not None:
            alphas = normalize_params(alphas, self.mask)

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

    def prune(self, num_keep: int) -> None:
        """Discretize architecture parameters to enforce sparsity.

        Args:
            num_keep (float): Number of operations to keep.
        """
        sorted_arch_params, _ = torch.sort(
            self._arch_parameters, dim=1, descending=True
        )
        top_k = num_keep
        thresholds = sorted_arch_params[:, top_k - 1].unsqueeze(1)
        self.mask = self._arch_parameters >= thresholds

        self._arch_parameters.data *= self.mask.float()  # type: ignore
        self._arch_parameters.data[self.mask].requires_grad = True  # type: ignore
        self._arch_parameters.data[~self.mask].requires_grad = False  # type: ignore
        if self._arch_parameters.data[~self.mask].grad:  # type: ignore
            self._arch_parameters.data[~self.mask].grad.zero_()  # type: ignore

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

        for cell in dicrete_model.cells:
            cell._discretize(self._arch_parameters)
        dicrete_model._arch_parameters = None

        return dicrete_model

    def model_weight_parameters(self) -> list[nn.Parameter]:
        params = set(self.parameters())
        params -= set(self._beta_parameters)
        if self._arch_parameters is not None:
            params -= set(self._arch_parameters)
        return list(params)


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
        assert stride == 1 or stride == 2, f"invalid stride {stride}"

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
