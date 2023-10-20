from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from confopt.searchspace.common import OperationChoices

from . import operations as ops
from .operations import OPS, TRANS_NAS_BENCH_101

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TNB101SearchModel(nn.Module):
    """Implementation of a TransNASBench-101 network.

    Args:
        C (int, optional): Number of channels. Defaults to 16.
        stride (int, optional): Stride for the deconvolutional operation.
        Defaults to 1.
        max_nodes (int, optional): Total amount of nodes in one cell.
        Defaults to 4.
        num_classes (int, optional): Number of classes for classification.
        Defaults to 10.
        op_names (list[str], optional): List of operations names used for
        cell structure.
        Defaults to TRANS_NAS_BENCH_101.
        affine (bool, optional): Whether to use affine transformations in BatchNorm in
        cells. Defaults to False.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm in cells. Defaults to False.
        dataset (str, optional): Name of the dataset used. Defaults to "cifar10".
        edge_normalization (bool, optional): Whether to enable edge normalization for
        partial connection. Defaults to False.
        discretized (bool, optional): Shows if we have a supernet or a discretized
        search space with one operation on each edge. Defaults to False.

    Attributes:
        cells (nn.ModuleList): List of cells in the search space.
        stem (nn.Module): Stem network tailored to the dataset at hand.
        decoder (nn.Module): Decoder network tailored to the dataset at hand.

    """

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
        discretized: bool = False,
    ):
        super().__init__()
        assert stride == 1 or stride == 2, f"invalid stride {stride}"

        self.C = C
        self.stride = stride
        self.edge_normalization = edge_normalization
        self.discretized = discretized

        self.op_names = deepcopy(op_names)
        self.max_nodes = max_nodes
        self.n_modules = 5
        self.blocks_per_module = [2] * self.n_modules

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

        self._arch_parameters = nn.Parameter(
            1e-3 * torch.randn(self.num_edge, len(op_names))  # type: ignore
        )
        self._beta_parameters = nn.Parameter(1e-3 * torch.randn(self.num_edge))

    def arch_parameters(self) -> nn.Parameter:
        """Getter function for the private architecture parameters.

        Returns:
            nn.Parameter: Alpha parameters of the model.
        """
        return self._arch_parameters

    def beta_parameters(self) -> nn.Parameter:
        """Getter function for the private beta parameters used for edge normalization.

        Returns:
            nn.Parameter: Beta parameters of the model.
        """
        return self._beta_parameters

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor to the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the output of the
            model twice.
        """
        if self.discretized:
            alphas = self._arch_parameters
        else:
            alphas = nn.functional.softmax(self._arch_parameters, dim=-1)

        feature = self.stem(inputs)
        for cell in self.cells:
            betas = torch.empty((0,)).to(self._arch_parameters.device)
            if self.edge_normalization:
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
            else:
                feature = cell(feature, alphas)

        out = self.decoder(feature)

        # out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        # logits = self.classifier(out)

        return out, out

    def _get_stem_for_task(self, task: str) -> nn.Module:
        """Builds a task dependant stem network.

        Args:
            task (str): Name of the dataset/task.

        Returns:
            nn.Module: Stem network
        """
        if task == "jigsaw":
            return ops.StemJigsaw(C_out=self.C)
        if task in ["class_object", "class_scene"]:
            return ops.Stem(C_out=self.C)
        if task == "autoencoder":
            return ops.Stem(C_out=self.C)
        return ops.Stem(C_in=3, C_out=self.C)

    def _get_decoder_for_task(self, task: str, n_channels: int) -> nn.Module:
        """Builds a task dependant decoder network.

        Args:
            task (str): Name of the dataset/task.
            n_channels (int): Number of channels.


        Returns:
            nn.Module: Decoder network.
        """
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
        self.edge_normalization = False
        self.discretized = True
        sorted_arch_params, _ = torch.sort(
            self._arch_parameters, dim=1, descending=True
        )
        top_k = int(op_sparsity * len(self.op_names))
        thresholds = sorted_arch_params[:, :top_k]
        mask = self._arch_parameters >= thresholds

        self._arch_parameters.data *= mask.float()
        self._arch_parameters.data[~mask].requires_grad = False
        if self._arch_parameters.data[~mask].grad:
            self._arch_parameters.data[~mask].grad.zero_()


class TNB101SearchCell(nn.Module):
    """Initialize a TransNasBench-101 cell.

    Args:
        C_in (int, optional): Number of input channels.
        C_out (int, optional): Number of output channels.
        stride (int, optional): Stride for the deconvolutional operation. Defaults to 1.
        max_nodes (int, optional): Total amount of nodes in one cell. Defaults to 4.
        op_names (list[str], optional): List of operations names used for
        cell structure. Defaults to TRANS_NAS_BENCH_101.
        affine (bool, optional): Whether to use affine transformations in BatchNorm in
        cells. Defaults to False.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm in cells. Defaults to False
        downsample (bool, optional):

    Attributes:
        edges (nn.ModuleDict): Contains OperationChoices for every edge in the cell.
        Keys are formatted as edge from node j to node i: "{i}<-{j}"
        edge_keys (Iterable[str]): Sorted Iterable over all possible keys in the cell.
        edge2index (dict(str, int)): Dictionary from edge keys to indices of the
        edge_keys iterable.
        num_edges (int): Number of edges in a cell.
    """

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
        alphas: torch.Tensor,
        betas: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor to the model.
            alphas (torch.Tensor): Alpha parameters of the model
            betas (list[torch.Tensor], optional): Beta parameters of the model.
            Defaults to None.

        Returns:
            torch.Tensor:
        """
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                weights = alphas[self.edge2index[node_str]]
                if betas is not None:
                    beta_weights = betas[self.edge2index[node_str]]
                    inter_nodes.append(
                        beta_weights * self.edges[node_str](nodes[j], weights)
                    )
                else:
                    inter_nodes.append(self.edges[node_str](nodes[j], weights))
            nodes.append(sum(inter_nodes))
        return nodes[-1]
