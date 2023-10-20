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
    """Generate parent combinations for a given node.

    Args:
        node (int): Node index.
        num_parents (int): Number of parents for the node.

    Returns:
        Any: List of parent combinations.
    """
    if node == 1 and num_parents == 1:
        return [(0,)]
    else:  # noqa: RET505
        return list(itertools.combinations(list(range(int(node))), num_parents))


class NASBench1Shot1SearchSpace(SearchSpace):
    """Initialize the NASBench1Shot1SearchSpace.

    Args:
        num_intermediate_nodes (int): Number of intermediate nodes in the cell.
        search_space_type (str): Type of search space, should be "S1", "S2", or "S3".
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If the given search_space_type is invalid or the total number
            of parents per node exceeds 9.
    """

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
            self.num_parents_per_node = {
                "0": 0,
                "1": 1,
                "2": 1,
                "3": 2,
                "4": 2,
                "5": 3,
            }
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
        """Return the architecture parameters of the search space."""
        return self.model.arch_parameters()  # type: ignore

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        """Set the architecture parameters of the search space.

        Args:
            arch_parameters (list[nn.Parameter]): The architecture parameters to set.

        Returns:
            None
        """
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

        Args:
            parents (dict[str, Any]): Parent nodes for each node.

        Returns:
            np.ndarray: The created adjacency matrix.
        """
        if self.search_space_type == "S1" or self.search_space_type == "S2":
            adjacency_matrix = self._create_adjacency_matrix(
                parents,
                adjacency_matrix=np.zeros([6, 6]),
                node=OUTPUT_NODE - 1,
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
        """Create the NASBench adjacency matrix based on given parent nodes.

        Args:
            parents (dict[str, Any]): Parent nodes for each node.

        Returns:
            np.ndarray: The created adjacency matrix.
        """
        if self.search_space_type == "S1" or self.search_space_type == "S2":
            return upscale_to_nasbench_format(
                self._create_adjacency_matrix_with_loose_ends(parents)
            )

        return self._create_adjacency_matrix_with_loose_ends(parents)

    def sample(self, with_loose_ends: bool, upscale: bool = True) -> tuple:
        """Sample an adjacency matrix and node operations.

        Args:
            with_loose_ends (bool): Whether to allow loose ends in the adjacency matrix.
            upscale (bool, optional): Whether to upscale the adjacency matrix for
            certain search space types (default is True).

        Returns:
            tuple: A tuple containing the sampled adjacency matrix and node operations.
        """
        if with_loose_ends:
            adjacency_matrix_sample = self._sample_adjacency_matrix_with_loose_ends()
        else:
            adjacency_matrix_sample = self._sample_adjacency_matrix_without_loose_ends(
                adjacency_matrix=np.zeros(
                    [
                        self.num_intermediate_nodes + 2,
                        self.num_intermediate_nodes + 2,
                    ]
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
        """Sample an adjacency matrix with loose ends.

        Returns:
            np.ndarray: The sampled adjacency matrix.
        """
        parents_per_node = [
            random.sample(
                list(itertools.combinations(list(range(int(node))), num_parents)),
                1,
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
        """Sample an adjacency matrix without loose ends recursively.

        Args:
            adjacency_matrix (np.ndarray): The adjacency matrix to be modified.
            node (int): The current node being processed.

        Returns:
            np.ndarray: The modified adjacency matrix.
        """
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
        """Convert a configuration to the NASBench-compatible format.

        This method takes a ConfigurationSpace object representing an architectural
        configuration and converts it to the format compatible with NASBench, which
        includes the adjacency matrix and operations for each choice block.

        Args:
            config (ConfigurationSpace): A ConfigurationSpace object containing
            architectural hyperparameters.

        Returns:
            tuple: A tuple containing the adjacency matrix and a list of operations for
            each choice block.

        The method extracts parent combinations and choice block operations from the
        input configuration and uses them to construct an adjacency matrix and a list
        of operations that define the neural architecture.

        Example:
            This method is essential for converting a configuration generated during the
            architecture search process into a format that can be queried using
            NASBench.

        Returns:
            tuple: A tuple containing the adjacency matrix and operations.
        """
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
        """Get the configuration space for the search space.

        This method constructs a configuration space representing the search space of
        neural architectures. The configuration space includes hyperparameters for
        choosing node operations and specifying parent combinations for each choice
        block.

        Returns:
            Any: A ConfigurationSpace object that defines the search space.

        The configuration space is used for hyperparameter optimization and searching
        within the defined neural architecture search space. It includes categorical
        hyperparameters for selecting the operations of choice blocks and specifying
        combinations of parent nodes for each choice block.

        Example:
            This method can be used to define the configuration space for architecture
            search algorithms, enabling the exploration of different architectural
            configurations within the search space.

        Returns:
            ConfigurationSpace: A ConfigurationSpace object that defines the search
            space.

        """
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
        """Generate the search space without loose ends.

        This method generates the search space for the neural architecture without loose
        ends. It explores different connectivity patterns and node operations within the
        search space, ensuring that all generated architectures adhere to the search
        space constraints.

        Yields:
            Tuple: A tuple containing the adjacency matrix, node operations, and the
            corresponding ModelSpec for architectures within the search space.

        The search space is constructed by iteratively generating valid connectivity
        patterns and evaluating all possible combinations of node operations. The
        resulting architectures are represented as adjacency matrices and their
        corresponding node operations.

        Example:
            This method can be used to explore the search space for neural architectures
            without loose ends, generating various valid architectures.

        Yields the generated architectures within the search space.

        """
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
        """Generate valid adjacency matrices for a given node in the search space.

        This method generates valid adjacency matrices for a specific node in the search
        space. It explores different combinations of parent nodes for the given node
        while ensuring that the resulting graphs adhere to the search space constraints.

        Args:
            adjacency_matrix (np.ndarray): The current adjacency matrix representing the
                graph.
            node (int): The node for which adjacency matrices are generated.

        Yields:
            np.ndarray: Valid adjacency matrices that adhere to the search space
            constraints.

        The adjacency matrix represents the current state of the graph. The method
        explores various parent node combinations for the specified node to create valid
        graphs within the search space.

        Example:
            Given a node with the requirement of 2 parents, calling this method for that
            node generates valid adjacency matrices with different parent combinations.

        Yields the valid adjacency matrices.

        Notes:
            This method utilizes a depth-first approach and may yield multiple valid
            adjacency matrices.

        """
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
        """Create an adjacency matrix for a given node's parents and its ancestors.

        This method constructs an adjacency matrix based on the provided parents for a
        specific node and its ancestors. It ensures that the generated graph adheres to
        the search space constraints.

        Args:
            parents (dict[str, Any]): A dictionary specifying parent nodes for each node
            in the graph.
            adjacency_matrix (np.ndarray): The current adjacency matrix of the graph.
            node (int): The node for which the adjacency matrix is being constructed.

        Returns:
            np.ndarray: An updated adjacency matrix.

        The adjacency matrix is initially provided, and it represents the current state
        of the graph. The method adds connections based on the specified parents for the
        given node and its ancestors while ensuring that the graph remains valid within
        the search space.

        Example:
            For a node with parents {'0': [0], '1': [0], '2': [1, 2]}, and an initial
            adjacency matrix of
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            calling this method for node 2 will update the adjacency matrix to:
            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 1]]

        Returns the updated adjacency matrix.
        """
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            return adjacency_matrix
        else:  # noqa: RET505
            for parent in parents[str(node)]:
                adjacency_matrix[parent, node] = 1
                if parent != 0:
                    adjacency_matrix = self._create_adjacency_matrix(
                        parents=parents,
                        adjacency_matrix=adjacency_matrix,
                        node=parent,
                    )
            return adjacency_matrix

    def _create_adjacency_matrix_with_loose_ends(
        self, parents: dict[str, Any]
    ) -> np.ndarray:
        """Create an adjacency matrix with loose ends for the specified parent nodes.

        This method generates an adjacency matrix that represents connections between
        nodes in a graph, considering the provided parents for each node. The resulting
        adjacency matrix accounts for loose ends (nodes without incoming connections).

        Args:
            parents (dict[str, Any]): A dictionary specifying parent nodes for each node
            in the graph.

        Returns:
            np.ndarray: An adjacency matrix representing the graph with loose ends.

        The adjacency matrix is constructed based on the provided parents for each node.
        A '1' in the matrix indicates a connection between nodes, while '0' represents
        no connection.

        Example:
            If parents = {'0': [], '1': [0], '2': [0, 1]}, the resulting adjacency
            matrix will have connections as follows:
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]

        Returns the generated adjacency matrix.
        """
        # Create the adjacency_matrix on a per node basis
        adjacency_matrix = np.zeros([len(parents), len(parents)])
        for node, node_parents in parents.items():
            for parent in node_parents:
                adjacency_matrix[parent, int(node)] = 1
        return adjacency_matrix

    def _check_validity_of_adjacency_matrix(self, adjacency_matrix: np.ndarray) -> bool:
        """Check the validity of an adjacency matrix representing a graph in the search
        space.

        This method performs several checks to determine whether the provided adjacency
        matrix is a valid graph in the specified search space.

        Args:
            adjacency_matrix (np.ndarray): The adjacency matrix to be checked.

        Returns:
            bool: True if the graph is valid; otherwise, False.

        The method performs the following checks:
        1. Checks that the graph is non-empty.
        2. Verifies that every node has the correct number of inputs.
        3. Ensures that if a node has outgoing edges, it also has incoming edges (apart
           from the input node).
        4. Validates that the input node is connected.
        5. Confirms that the graph has no more than 9 edges.
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
        """Generate adjacency matrices with loose ends for the specified search space.

        This method generates adjacency matrices with loose ends for the given search
        space type. The generated adjacency matrices are yielded one by one.

        Yields:
            Any: Adjacency matrix with loose ends based on the search space type.
        """
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
        """Calculate the objective function (validation_accuracy and training_time)
        value for a given NASBench configuration.

        This method computes the objective function value for a given NASBench
        configuration by querying the NASBench dataset. It records data to the history
        if the search space type is "S2".

        Args:
            nasbench (api.NASBench): The NASBench API for querying architecture data.
            config (ConfigSpace.configuration_space.ConfigurationSpace): The
                configuration to evaluate.
            budget (int, optional): The budget for evaluating the configuration.
                Defaults to 108.

        Returns:
            tuple: A tuple containing the validation accuracy and training time for the
                evaluated configuration.
        """
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

        return (
            nasbench_data["validation_accuracy"],
            nasbench_data["training_time"],
        )

    def discretize(self) -> None:
        """Discretize the model's architecture parameters to enforce sparsity.

        Note:
            This method discretizes the model's architecture parameters to enforce
            sparsity. It sets the sparsity level to 0.2 (20% of operations will be kept)
            and calls the `_discretize` method to apply the discretization.
        """
        sparsity = 0.2
        self.model._discretize(sparsity)
