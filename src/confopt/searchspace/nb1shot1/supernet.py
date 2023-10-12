from __future__ import annotations

import copy
import itertools
import random
from typing import Any, Generator

import ConfigSpace
from nasbench import api
import numpy as np
import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace
from confopt.searchspace.nb1shot1.core.genotypes import PRIMITIVES

from .core import NasBench1Shot1SearchModel
from .core.util import (
    CONV1X1,
    CONV3X3,
    INPUT,
    MAXPOOL3X3,
    OUTPUT,
    OUTPUT_NODE,
    Architecture,
    Model,
    upscale_to_nasbench_format,
)
from .core.util import parent_combinations as parent_combinations_old

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parent_combinations(node: int, num_parents: int) -> Any:
    if node == 1 and num_parents == 1:
        return [(0,)]
    else:  # noqa: RET505
        return list(itertools.combinations(list(range(int(node))), num_parents))


class NASBench1Shot1SearchSpace(SearchSpace):
    def __init__(
        self,
        num_intermediate_nodes: int = 4,
        search_space_type: str = "S1",
        *args: Any,
        **kwargs: Any,
    ):
        self.search_space_type = search_space_type
        self.num_intermediate_nodes = num_intermediate_nodes
        self.num_parents_per_node = {}

        self.run_history: list = []
        if self.search_space_type == "S1":
            self.num_parents_per_node = {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 2,
                "4": 2,
                "5": 2,
            }

            if sum(self.num_parents_per_node.values()) > 9:
                raise ValueError("Each nasbench cell has at most 9 edges.")

            self.test_min_error = 0.05448716878890991
            self.valid_min_error = 0.049278855323791504

        elif self.search_space_type == "S2":
            self.num_parents_per_node = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2, "5": 3}
            if sum(self.num_parents_per_node.values()) > 9:
                raise ValueError("Each nasbench cell has at most 9 edges.")

            self.test_min_error = 0.057592153549194336
            self.valid_min_error = 0.051582515239715576

        elif self.search_space_type == "S3":
            self.num_parents_per_node = {
                "0": 0,
                "1": 1,
                "2": 1,
                "3": 1,
                "4": 2,
                "5": 2,
                "6": 2,
            }
            if sum(self.num_parents_per_node.values()) > 9:
                raise ValueError("Each nasbench cell has at most 9 edges.")

            self.test_min_error = 0.05338543653488159
            self.valid_min_error = 0.04847759008407593

        else:
            raise ValueError(
                "The search_space_type argument received an incorrect \
                             value. Please select between S1, S2, and S3"
            )

        search_space_info = {
            "search_space_type": self.search_space_type,
            "num_intermediate_nodes": self.num_intermediate_nodes,
        }
        kwargs["search_space_info"] = search_space_info
        model = NasBench1Shot1SearchModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return self.model.arch_parameters()  # type: ignore

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        (
            self.model.alphas_mixed_op.data,
            self.model.alphas_output.data,
            _,
            _,
        ) = arch_parameters
        self.model._arch_parameters = [
            self.model.alphas_mixed_op,
            self.model.alphas_output,
            *self.model.alphas_inputs,
        ]

    def create_nasbench_adjacency_matrix(self, parents: dict[str, Any]) -> np.ndarray:
        """Based on given connectivity pattern create the corresponding adjacency
        matrix.
        """
        if self.search_space_type == "S1" or self.search_space_type == "S2":
            adjacency_matrix = self._create_adjacency_matrix(
                parents, adjacency_matrix=np.zeros([6, 6]), node=OUTPUT_NODE - 1
            )
            # Create nasbench compatible adjacency matrix
            return upscale_to_nasbench_format(adjacency_matrix)

        adjacency_matrix = self._create_adjacency_matrix(
            parents, adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE
        )
        return adjacency_matrix

    def create_nasbench_adjacency_matrix_with_loose_ends(
        self, parents: dict[str, Any]
    ) -> np.ndarray:
        if self.search_space_type == "S1" or self.search_space_type == "S2":
            return upscale_to_nasbench_format(
                self._create_adjacency_matrix_with_loose_ends(parents)
            )

        return self._create_adjacency_matrix_with_loose_ends(parents)

    def sample(self, with_loose_ends: bool, upscale: bool = True) -> tuple:
        if with_loose_ends:
            adjacency_matrix_sample = self._sample_adjacency_matrix_with_loose_ends()
        else:
            adjacency_matrix_sample = self._sample_adjacency_matrix_without_loose_ends(
                adjacency_matrix=np.zeros(
                    [self.num_intermediate_nodes + 2, self.num_intermediate_nodes + 2]
                ),
                node=self.num_intermediate_nodes + 1,
            )
            assert self._check_validity_of_adjacency_matrix(
                adjacency_matrix_sample
            ), "Incorrect graph"

        if upscale and self.search_space_type in ["S1", "S2"]:
            adjacency_matrix_sample = upscale_to_nasbench_format(
                adjacency_matrix_sample
            )
        return adjacency_matrix_sample, random.choices(
            PRIMITIVES, k=self.num_intermediate_nodes
        )

    def _sample_adjacency_matrix_with_loose_ends(self) -> np.ndarray:
        parents_per_node = [
            random.sample(
                list(itertools.combinations(list(range(int(node))), num_parents)), 1
            )
            for node, num_parents in self.num_parents_per_node.items()
        ][2:]
        parents = {"0": [], "1": [0]}
        for node, node_parent in enumerate(parents_per_node, 2):
            parents[str(node)] = node_parent
        adjacency_matrix = self._create_adjacency_matrix_with_loose_ends(parents)
        return adjacency_matrix

    def _sample_adjacency_matrix_without_loose_ends(
        self, adjacency_matrix: np.ndarray, node: int
    ) -> np.ndarray:
        req_num_parents = self.num_parents_per_node[str(node)]
        current_num_parents = np.sum(adjacency_matrix[:, node], dtype=np.int)
        num_parents_left = req_num_parents - current_num_parents
        sampled_parents = random.sample(
            list(
                parent_combinations_old(
                    adjacency_matrix, node, n_parents=num_parents_left
                )
            ),
            1,
        )[0]
        for parent in sampled_parents:
            adjacency_matrix[parent, node] = 1
            adjacency_matrix = self._sample_adjacency_matrix_without_loose_ends(
                adjacency_matrix, parent
            )
        return adjacency_matrix

    def generate_adjacency_matrix_without_loose_ends(self) -> np.ndarray:
        """Returns every adjacency matrix in the search space without loose ends."""
        if self.search_space_type == "S1" or self.search_space_type == "S2":
            for adjacency_matrix in self._generate_adjacency_matrix(
                adjacency_matrix=np.zeros([6, 6]), node=OUTPUT_NODE - 1
            ):
                yield upscale_to_nasbench_format(adjacency_matrix)
        else:
            for adjacency_matrix in self._generate_adjacency_matrix(
                adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE
            ):
                yield adjacency_matrix

    def convert_config_to_nasbench_format(
        self, config: ConfigSpace.configuration_space.ConfigurationSpace
    ) -> tuple:
        parents = {
            node: config[f"choice_block_{node}_parents"]
            for node in list(self.num_parents_per_node.keys())[1:]
        }
        parents["0"] = []
        adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(
            parents
        )
        ops = [
            config[f"choice_block_{node}_op"]
            for node in list(self.num_parents_per_node.keys())[1:-1]
        ]
        return adjacency_matrix, ops

    def get_configuration_space(self) -> Any:
        cs = ConfigSpace.ConfigurationSpace()

        for node in list(self.num_parents_per_node.keys())[1:-1]:
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter(
                    f"choice_block_{node}_op", [CONV1X1, CONV3X3, MAXPOOL3X3]
                )
            )

        for choice_block_index, num_parents in list(self.num_parents_per_node.items())[
            1:
        ]:
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter(
                    f"choice_block_{choice_block_index}_parents",
                    parent_combinations(
                        node=int(choice_block_index), num_parents=num_parents
                    ),
                )
            )
        return cs

    def generate_search_space_without_loose_ends(self) -> Generator:
        # Create all possible connectivity patterns
        for iter, adjacency_matrix in enumerate(  # noqa: A001
            self.generate_adjacency_matrix_without_loose_ends()
        ):
            print(iter)
            # Print graph
            # Evaluate every possible combination of node ops.
            n_repeats = int(np.sum(np.sum(adjacency_matrix, axis=1)[1:-1] > 0))
            for combination in itertools.product(
                [CONV1X1, CONV3X3, MAXPOOL3X3], repeat=n_repeats
            ):
                # Create node labels
                # Add some op as node 6 which isn't used, here conv1x1
                ops = [INPUT]
                combination_list = list(combination)
                for i in range(5):
                    if np.sum(adjacency_matrix, axis=1)[i + 1] > 0:
                        ops.append(combination_list.pop())
                    else:
                        ops.append(CONV1X1)
                assert len(combination_list) == 0, "Something is wrong"
                ops.append(OUTPUT)

                # Create nested list from numpy matrix
                nasbench_adjacency_matrix = adjacency_matrix.astype(np.int).tolist()

                # Assemble the model spec
                model_spec = api.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=nasbench_adjacency_matrix,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=ops,
                )

                yield adjacency_matrix, ops, model_spec

    def _generate_adjacency_matrix(
        self, adjacency_matrix: np.ndarray, node: int
    ) -> np.ndarray:
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            yield adjacency_matrix
        else:
            req_num_parents = self.num_parents_per_node[str(node)]
            current_num_parents = np.sum(adjacency_matrix[:, node], dtype=np.int)
            num_parents_left = req_num_parents - current_num_parents

            for parents in parent_combinations_old(
                adjacency_matrix, node, n_parents=num_parents_left
            ):
                # Make copy of adjacency matrix so that when it returns to this stack
                # it can continue with the unmodified adjacency matrix
                adjacency_matrix_copy = copy.copy(adjacency_matrix)
                for parent in parents:
                    adjacency_matrix_copy[parent, node] = 1
                    for graph in self._generate_adjacency_matrix(
                        adjacency_matrix=adjacency_matrix_copy, node=parent
                    ):
                        yield graph

    def _create_adjacency_matrix(
        self, parents: dict[str, Any], adjacency_matrix: np.ndarray, node: int
    ) -> np.ndarray:
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            return adjacency_matrix
        else:  # noqa: RET505
            for parent in parents[str(node)]:
                adjacency_matrix[parent, node] = 1
                if parent != 0:
                    adjacency_matrix = self._create_adjacency_matrix(
                        parents=parents, adjacency_matrix=adjacency_matrix, node=parent
                    )
            return adjacency_matrix

    def _create_adjacency_matrix_with_loose_ends(
        self, parents: dict[str, Any]
    ) -> np.ndarray:
        # Create the adjacency_matrix on a per node basis
        adjacency_matrix = np.zeros([len(parents), len(parents)])
        for node, node_parents in parents.items():
            for parent in node_parents:
                adjacency_matrix[parent, int(node)] = 1
        return adjacency_matrix

    def _check_validity_of_adjacency_matrix(self, adjacency_matrix: np.ndarray) -> bool:
        """Checks whether a graph is a valid graph in the search space.
        1. Checks that the graph is non empty
        2. Checks that every node has the correct number of inputs
        3. Checks that if a node has outgoing edges then it should also have incoming
           edges
        4. Checks that input node is connected
        5. Checks that the graph has no more than 9 edges
        :param adjacency_matrix:
        :return:.
        """
        # Check that the graph contains nodes
        num_intermediate_nodes = sum(
            np.array(np.sum(adjacency_matrix, axis=1) > 0, dtype=int)[1:-1]
        )
        if num_intermediate_nodes == 0:
            return False

        # Check that every node has exactly the right number of inputs
        col_sums = np.sum(adjacency_matrix[:, :], axis=0)
        for col_idx, col_sum in enumerate(col_sums):
            if col_sum > 0 and col_sum != self.num_parents_per_node[str(col_idx)]:
                return False

        # Check that if a node has outputs then it should also have incoming edges
        # (apart from zero)
        col_sums = np.sum(np.sum(adjacency_matrix, axis=0) > 0)
        row_sums = np.sum(np.sum(adjacency_matrix, axis=1) > 0)
        if col_sums != row_sums:
            return False

        # Check that the input node is always connected. Otherwise the graph is
        # disconnected.
        row_sum = np.sum(adjacency_matrix, axis=1)
        if row_sum[0] == 0:
            return False

        # Check that the graph returned has no more than 9 edges.
        num_edges = np.sum(adjacency_matrix.flatten())
        if num_edges > 9:
            return False

        return True

    def generate_with_loose_ends(self) -> Any:
        if self.search_space_type == "S1" or self.search_space_type == "S2":
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
                if self.search_space_type == "S1":
                    parents = {
                        "0": [],
                        "1": [0],
                        "2": [0, 1],
                        "3": parent_node_3,
                        "4": parent_node_4,
                        "5": output_parents,
                    }
                elif self.search_space_type == "S2":
                    parents = {
                        "0": [],
                        "1": [0],
                        "2": parent_node_2,
                        "3": parent_node_3,
                        "4": parent_node_4,
                        "5": output_parents,
                    }
                adjacency_matrix = (
                    self.create_nasbench_adjacency_matrix_with_loose_ends(parents)
                )

                yield adjacency_matrix
        else:
            for (
                parent_node_2,
                parent_node_3,
                parent_node_4,
                parent_node_5,
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
                    "5": parent_node_5,
                    "6": output_parents,
                }
                adjacency_matrix = (
                    self.create_nasbench_adjacency_matrix_with_loose_ends(parents)
                )
                yield adjacency_matrix

    def objective_function(
        self,
        nasbench: api.NASBench,
        config: ConfigSpace.configuration_space.ConfigurationSpace,
        budget: int = 108,
    ) -> tuple:
        adjacency_matrix, node_list = self.convert_config_to_nasbench_format(config)
        # adjacency_matrix = upscale_to_nasbench_format(adjacency_matrix)
        if self.search_space_type == "S3":
            node_list = [INPUT, *node_list, OUTPUT]
        else:
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = nasbench.query(model_spec, epochs=budget)

        # record the data to history
        if self.search_space_type == "S2":
            architecture = Model()
            arch = Architecture(adjacency_matrix=adjacency_matrix, node_list=node_list)
            architecture.update_data(arch, nasbench_data, budget)
            self.run_history.append(architecture)

        return nasbench_data["validation_accuracy"], nasbench_data["training_time"]
