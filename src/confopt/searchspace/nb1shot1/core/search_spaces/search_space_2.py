from __future__ import annotations

import itertools
from typing import Any, Generator

import ConfigSpace
from nasbench import api
import numpy as np

from .search_space import NB1Shot1Space
from .utils import (
    CONV1X1,
    INPUT,
    OUTPUT,
    OUTPUT_NODE,
    Architecture,
    Model,
    upscale_to_nasbench_format,
)


class NB1Shot1Space2(NB1Shot1Space):
    def __init__(self) -> None:
        self.search_space_number = 2
        self.num_intermediate_nodes = 4
        super().__init__(
            search_space_number=self.search_space_number,
            num_intermediate_nodes=self.num_intermediate_nodes,
        )
        """
        SEARCH SPACE 2
        """
        self.num_parents_per_node = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2, "5": 3}
        if sum(self.num_parents_per_node.values()) > 9:
            raise ValueError("Each nasbench cell has at most 9 edges.")

        self.test_min_error = 0.057592153549194336
        self.valid_min_error = 0.051582515239715576

    def create_nasbench_adjacency_matrix(self, parents: dict[str, Any]) -> np.ndarray:
        adjacency_matrix = self._create_adjacency_matrix(
            parents, adjacency_matrix=np.zeros([6, 6]), node=OUTPUT_NODE - 1
        )
        # Create nasbench compatible adjacency matrix
        return upscale_to_nasbench_format(adjacency_matrix)

    def create_nasbench_adjacency_matrix_with_loose_ends(
        self, parents: dict[str, Any]
    ) -> np.ndarray:
        return upscale_to_nasbench_format(
            self._create_adjacency_matrix_with_loose_ends(parents)
        )

    def generate_adjacency_matrix_without_loose_ends(
        self,
    ) -> Generator[np.ndarray, None, None]:
        for adjacency_matrix in self._generate_adjacency_matrix(
            adjacency_matrix=np.zeros([6, 6]), node=OUTPUT_NODE - 1
        ):
            yield upscale_to_nasbench_format(adjacency_matrix)

    def objective_function(
        self,
        nasbench: api.NASBench,
        config: ConfigSpace.configuration_space.ConfigurationSpace,
        budget: int = 108,
    ) -> tuple:
        adjacency_matrix, node_list = super().convert_config_to_nasbench_format(config)
        # adjacency_matrix = upscale_to_nasbench_format(adjacency_matrix)
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = nasbench.query(model_spec, epochs=budget)

        # record the data to history
        architecture = Model()
        arch = Architecture(adjacency_matrix=adjacency_matrix, node_list=node_list)
        architecture.update_data(arch, nasbench_data, budget)
        self.run_history.append(architecture)

        return nasbench_data["validation_accuracy"], nasbench_data["training_time"]

    def generate_with_loose_ends(self) -> Generator[np.ndarray, None, None]:
        for (
            parent_node_2,
            parent_node_3,
            parent_node_4,
            output_parents,
        ) in itertools.product(
            *[
                itertools.combinations(list(range(int(node))), num_parents)
                for node, num_parents in self.num_parents_per_node.items()
            ][2:]
        ):
            parents = {
                "0": [],
                "1": [0],
                "2": parent_node_2,
                "3": parent_node_3,
                "4": parent_node_4,
                "5": output_parents,
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(
                parents
            )
            yield adjacency_matrix
