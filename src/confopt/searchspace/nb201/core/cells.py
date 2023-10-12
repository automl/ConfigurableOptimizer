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
                                C_in, C_out, stride, affine, track_running_stats
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
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        return string

    def forward(
        self,
        inputs: torch.Tensor,
        weightss: list[torch.Tensor],
        beta_weightss: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                weights = weightss[self.edge2index[node_str]]
                if beta_weightss is not None:
                    beta_weights = beta_weightss[self.edge2index[node_str]]
                    inter_nodes.append(
                        beta_weights * self.edges[node_str](nodes[j], weights)
                    )
                else:
                    inter_nodes.append(self.edges[node_str](nodes[j], weights))
            nodes.append(sum(inter_nodes))
        return nodes[-1]


class InferCell(nn.Module):
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
