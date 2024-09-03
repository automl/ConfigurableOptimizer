from __future__ import annotations

from .operations import TRANS_NAS_BENCH_101


class TNB101Genotype:
    def __init__(self, node_edge_dict: dict, op_idx_list: list) -> None:
        self.node_edge_dict = node_edge_dict
        self.op_idx_list = op_idx_list
        self.node_edge_list = []
        self._verify_ops()
        for node in sorted(node_edge_dict.keys()):
            # of the form (op_name, incoming node)
            self.node_edge_list.append(tuple(node_edge_dict[node]))

    def _verify_ops(self) -> None:
        ops_idx_from_dict = []
        for node in sorted(self.node_edge_dict.keys()):
            for edge_name, _ in self.node_edge_dict[node]:
                ops_idx_from_dict.append(TRANS_NAS_BENCH_101.index(edge_name))

        assert self.op_idx_list == ops_idx_from_dict

    def __str__(self) -> str:
        strings = []
        for node_info in self.node_edge_list:
            string = "|".join([x[0] + f"~{x[1]}" for x in node_info])
            string = f"|{string}|"
            strings.append(string)
        return "+".join(strings)

    def tostr(self) -> str:
        return str(self)

    def get_arch_str(self) -> str:
        return "64-41414-{}_{}{}_{}{}{}".format(*self.op_idx_list)
