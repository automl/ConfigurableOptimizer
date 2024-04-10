##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from confopt.searchspace.common import OperationChoices

from .genotypes import Structure
from .operations import OPS

__all__ = ["NAS201SearchCell", "InferCell"]


class NAS201SearchCell(nn.Module):
    """NAS-Bench-201 Search Cell Class.

    Args:
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int): Stride for convolutional operations.
        max_nodes (int): Maximum number of nodes in the cell.
        op_names (list[str]): List of operation names to choose from.
        affine (bool, optional): Whether to use affine transformations in BatchNorm.
        Defaults to False.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm. Defaults to True.

    Attributes:
        op_names (list[str]): List of operation names.
        edges (nn.ModuleDict): Module dictionary representing edges between nodes.
        max_nodes (int): Maximum number of nodes in the cell.
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        edge_keys (list[str]): Sorted list of edge keys.
        edge2index (dict[str, int]): Mapping of edge keys to their indices.
        num_edges (int): Number of edges in the cell.

    Note:
        This class represents a search cell for NAS-Bench-201 with customizable
        architecture choices.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int,
        max_nodes: int,
        op_names: list[str],
        affine: bool = False,
        track_running_stats: bool = True,
    ):
        super().__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = f"{i}<-{j}"
                if j == 0:
                    xlists = nn.ModuleList(
                        [
                            OPS[op_name](
                                C_in,
                                C_out,
                                stride,
                                affine,
                                track_running_stats,
                            )
                            for op_name in op_names
                        ]
                    )
                else:
                    xlists = nn.ModuleList(
                        [
                            OPS[op_name](C_in, C_out, 1, affine, track_running_stats)
                            for op_name in op_names
                        ]
                    )
                self.edges[node_str] = OperationChoices(
                    ops=xlists, is_reduction_cell=False
                )
        self.edge_keys = sorted(self.edges.keys())
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges: int = len(self.edges)

    def extra_repr(self) -> str:
        """Return a string representation of the NAS201SearchCell.

        Returns:
            str: A string containing information about the cell's number of nodes, input
            and output dimensions.

        Note:
            This method constructs a human-readable string that includes details about
            the cell's architecture.
        """
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        return string

    def forward(
        self,
        inputs: torch.Tensor,
        weightss: list[torch.Tensor] | None = None,
        beta_weightss: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass through the NAS201SearchCell.

        Args:
            inputs (torch.Tensor): Input tensor to the cell.
            weightss (list[torch.Tensor]): List of weights tensors for the cell's edges.
            (alphas)
            beta_weightss (list[torch.Tensor] | None, optional): List of beta weights
            tensors for the cell's edges. Defaults to None assuming we do not have
            partial connection.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the NAS201SearchCell, applying
            operations to input tensors based on the provided weights and beta weights
            (if edge normalization is required) for each edge.
        """
        if weightss is None:
            return self.discrete_model_forward(inputs)

        if beta_weightss is not None:
            return self.edge_normalization_forward(inputs, weightss, beta_weightss)

        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(self.edges[node_str](nodes[j], weights))
            nodes.append(sum(inter_nodes))  # type: ignore
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
        weightss: list[torch.Tensor],
        beta_weightss: list[torch.Tensor],
    ) -> torch.Tensor:
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                weights = weightss[self.edge2index[node_str]]
                beta_weights = beta_weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    beta_weights * self.edges[node_str](nodes[j], weights)
                )
            nodes.append(sum(inter_nodes))  # type: ignore
        return nodes[-1]

    def _discretize(self, weightss: list[torch.Tensor]) -> None:
        for i in range(1, self.max_nodes):
            for j in range(i):
                node_str = f"{i}<-{j}"
                max_idx = torch.argmax(weightss[self.edge2index[node_str]], dim=-1)
                self.edges[node_str] = (self.edges[node_str].ops)[  # type: ignore
                    max_idx
                ]


class InferCell(nn.Module):
    """Inference Cell Class.

    Args:
        genotype (Structure): Genotype structure describing the architecture.
        C_in (int): Number of input channels.
        C_out (int): Number of output channels.
        stride (int): Stride for convolutional operations.
        affine (bool, optional): Whether to use affine transformations in BatchNorm.
        Defaults to True.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm. Defaults to True.

    Attributes:
        layers (nn.ModuleList): List of layers in the cell.
        node_IN (list[list[int]]): List of input nodes for each node in the cell.
        node_IX (list[list[int]]): List of indices of layers connected to each node
        in the cell.
        genotype (Structure): Genotype structure describing the cell's architecture.
        nodes (int): Number of nodes in the cell.
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.

    Note:
        This class represents an inference cell with a specified architecture defined by
         the genotype.
    """

    def __init__(
        self,
        genotype: Structure,
        C_in: int,
        C_out: int,
        stride: int,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for op_name, op_in in node_info:
                if op_in == 0:
                    layer = OPS[op_name](
                        C_in, C_out, stride, affine, track_running_stats
                    )
                else:
                    layer = OPS[op_name](C_out, C_out, 1, affine, track_running_stats)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def extra_repr(self) -> str:
        """Return an informative string representation of the custom search model.

        Returns:
            str: A string containing information about the model's nodes, input and
            output dimensions, and genotype.

        Note:
            This method constructs a human-readable string that includes details about
            the model's nodes, input and output dimensions, and genotype.
        """
        string = "info :: nodes={nodes}, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        laystr = []
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = [f"I{_ii}-L{_il}" for _il, _ii in zip(node_layers, node_innods)]
            x = "{:}<-({:})".format(i + 1, ",".join(y))
            laystr.append(x)
        return (
            string + ", [{:}]".format(" | ".join(laystr)) + f", {self.genotype.tostr()}"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the InferCell of custom search model NB201SearchModel.

        Args:
            inputs (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: The output tensor of the forward pass.

        Note:
            This method performs a forward pass through the custom search model. It
            processes the input tensor through the model's nodes and layers and
            returns the final output tensor.
        """
        nodes = [inputs]
        for _i, (node_layers, node_innods) in enumerate(
            zip(self.node_IX, self.node_IN)
        ):
            node_feature = sum(
                self.layers[_il](nodes[_ii])
                for _il, _ii in zip(node_layers, node_innods)
            )
            nodes.append(node_feature)
        return nodes[-1]
