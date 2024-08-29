from __future__ import annotations

from collections import namedtuple
import itertools
import os
import re
import typing

import ConfigSpace
from nasbench import api
import networkx as nx
import numpy as np
import seaborn as sns

from confopt.searchspace.common.base_search import SearchSpace

from .search_space_3 import SearchSpace3

sns.set_style("whitegrid")

INPUT = "input"
OUTPUT = "output"
CONV1X1 = "conv1x1-bn-relu"
CONV3X3 = "conv3x3-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OUTPUT_NODE = 6

Architecture = namedtuple("Architecture", ["adjacency_matrix", "node_list"])


class NasbenchWrapper(api.NASBench):
    """Small modification to the NASBench class, to return all three architecture
    evaluations at the same time, instead of samples.
    """

    def query(
        self, model_spec: api.ModelSpec, epochs: int = 108, stop_halfway: bool = False
    ) -> list:
        """Fetch one of the evaluations for this model spec.

        Each call will sample one of the config['num_repeats'] evaluations of the
        model. This means that repeated queries of the same model (or isomorphic
        models) may return identical metrics.

        This function will increment the budget counters for benchmarking purposes.
        See self.training_time_spent, and self.total_epochs_spent.

        This function also allows querying the evaluation metrics at the halfway
        point of training using stop_halfway. Using this option will increment the
        budget counters only up to the halfway point.

        Args:
          model_spec: ModelSpec object.
          epochs: number of epochs trained. Must be one of the evaluated number of
            epochs, [4, 12, 36, 108] for the full dataset.
          stop_halfway: if True, returned dict will only contain the training time
            and accuracies at the halfway point of training (num_epochs/2).
            Otherwise, returns the time and accuracies at the end of training
            (num_epochs).

        Returns:
          dict containing the evaluated darts for this object.

        Raises:
          OutOfDomainError: if model_spec or num_epochs is outside the search space.
        """
        if epochs not in self.valid_epochs:
            raise api.OutOfDomainError(
                "invalid number of epochs, must be one of %s" % self.valid_epochs
            )

        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
        trainings = []
        for index in range(self.config["num_repeats"]):
            computed_stat_at_epoch = computed_stat[epochs][index]

            data = {}
            data["module_adjacency"] = fixed_stat["module_adjacency"]
            data["module_operations"] = fixed_stat["module_operations"]
            data["trainable_parameters"] = fixed_stat["trainable_parameters"]

            if stop_halfway:
                data["training_time"] = computed_stat_at_epoch["halfway_training_time"]
                data["train_accuracy"] = computed_stat_at_epoch[
                    "halfway_train_accuracy"
                ]
                data["validation_accuracy"] = computed_stat_at_epoch[
                    "halfway_validation_accuracy"
                ]
                data["test_accuracy"] = computed_stat_at_epoch["halfway_test_accuracy"]
            else:
                data["training_time"] = computed_stat_at_epoch["final_training_time"]
                data["train_accuracy"] = computed_stat_at_epoch["final_train_accuracy"]
                data["validation_accuracy"] = computed_stat_at_epoch[
                    "final_validation_accuracy"
                ]
                data["test_accuracy"] = computed_stat_at_epoch["final_test_accuracy"]

            self.training_time_spent += data["training_time"]
            if stop_halfway:
                self.total_epochs_spent += epochs // 2
            else:
                self.total_epochs_spent += epochs
            trainings.append(data)

        return trainings


def get_top_k(array: np.ndarray, k: int) -> list:
    return list(np.argpartition(array[0], -k)[-k:])


@typing.no_type_check
def parent_combinations(
    adjacency_matrix: np.ndarray, node: int, n_parents: int = 2
) -> list:
    """Get all possible parent combinations for the current node."""
    if node != 1:
        # Parents can only be nodes which have an index that is lower than the
        # current index, because of the upper triangular adjacency matrix and
        # because the index is also a topological ordering in our case.
        return itertools.combinations(
            np.argwhere(adjacency_matrix[:node, node] == 0).flatten(), n_parents
        )  # (e.g. (0, 1), (0, 2), (1, 2), ...

    return [[0]]


def draw_graph_to_adjacency_matrix(graph: dict) -> None:
    """Draws the graph in circular format for easier debugging.
    :param graph:
    :return:
    """  # noqa: D415
    dag = nx.DiGraph(graph)
    nx.draw_circular(dag, with_labels=True)


def upscale_to_nasbench_format(adjacency_matrix: np.ndarray) -> np.ndarray:
    """The search space uses only 4 intermediate nodes, rather than 5 as used
    in nasbench. This method adds a dummy node to the graph which is never used to
    be compatible with nasbench.
    :param adjacency_matrix:
    :return:
    """  # noqa: D415
    return np.insert(
        np.insert(adjacency_matrix, 5, [0, 0, 0, 0, 0, 0], axis=1),
        5,
        [0, 0, 0, 0, 0, 0, 0],
        axis=0,
    )


@typing.no_type_check
def parse_log(path: str) -> tuple:
    f = open(os.path.join(path, "log.txt"))  # noqa: SIM115
    # Read in the relevant information
    train_accuracies = []
    valid_accuracies = []
    for line in f:
        if "train_acc" in line:
            train_accuracies.append(line)
        elif "valid_acc" in line:
            valid_accuracies.append(line)

    valid_error = [
        [
            1
            - 1
            / 100
            * float(re.search("valid_acc ([-+]?[0-9]*\\.?[0-9]+)", line).group(1))
        ]
        for line in valid_accuracies
    ]
    train_error = [
        [
            1
            - 1
            / 100
            * float(re.search("train_acc ([-+]?[0-9]*\\.?[0-9]+)", line).group(1))
        ]
        for line in train_accuracies
    ]

    return valid_error, train_error


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text: str) -> int | str:
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> list:
    """alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments).
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


class Model:
    """A class representing a model.

    It holds two attributes: `arch` (the simulated architecture) and `accuracy`
    (the simulated accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:  (as in the original code)
      arch: the architecture as an int representing a bit-string of length `DIM`.
          As a result, the integers are required to be less than `2**DIM`. They
          can be visualized as strings of 0s and 1s by calling `print(model)`,
          where `model` is an instance of this class.
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self) -> None:
        self.arch: None | Architecture = None
        self.validation_accuracy: None | float = None
        self.test_accuracy: None | float = None
        self.training_time: None | float = None
        self.budget: None | int = None

    def update_data(
        self,
        arch: typing.Any,
        nasbench_data: dict[str, float],
        budget: int | None = None,
    ) -> None:
        self.arch = arch
        self.validation_accuracy = nasbench_data["validation_accuracy"]
        self.test_accuracy = nasbench_data["test_accuracy"]
        self.training_time = nasbench_data["training_time"]
        self.budget = budget

    def query_nasbench(
        self, search_space: SearchSpace, nasbench: api.NASBench, sample: bool
    ) -> None:
        config = ConfigSpace.Configuration(
            search_space.get_configuration_space(), vector=sample
        )
        (
            adjacency_matrix,
            node_list,
        ) = search_space.convert_config_to_nasbench_format(config)
        if type(search_space) == SearchSpace3:
            node_list = [INPUT, *node_list, OUTPUT]
        else:
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)

        nasbench_data = nasbench.query(model_spec)
        self.arch = Architecture(adjacency_matrix=adjacency_matrix, node_list=node_list)
        self.validation_accuracy = nasbench_data["validation_accuracy"]
        self.test_accuracy = nasbench_data["test_accuracy"]
        self.training_time = nasbench_data["training_time"]
