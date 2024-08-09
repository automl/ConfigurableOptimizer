from __future__ import annotations

from functools import partial

import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace

from .core.genotypes import Structure as NB201Gynotype
from .core.model_search import NB201SearchModel, check_grads_cosine, preserve_grads

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NASBench201SearchSpace(SearchSpace):
    def __init__(self, *args, **kwargs):  # type: ignore
        """Initialize the custom search model of NASBench201SearchSpace.

        Args:
            *args: Positional arguments to pass to the NB201SearchModel constructor.
            **kwargs: Keyword arguments to pass to the NB201SearchModel constructor.

        Note:
            This constructor initializes the custom search model by creating an instance
            of NB201SearchModel with the provided arguments and keyword arguments.
            The resulting model is then moved to the specified device (DEVICE).
        """
        model = NB201SearchModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.

        Note:
            This method sets the architectural parameters of the model to the provided
            values.
        """
        return [self.model.arch_parameters]  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the beta parameters of the model.

        Returns:
            list[nn.Parameter]: A list containing the beta parameters for the model.
        """
        return [self.model.beta_parameters]  # type: ignore

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.

        Note:
            This method sets the architectural parameters of the model to the provided
            values.
        """
        self.model.arch_parameters.data = arch_parameters[0]

    def prune(self, num_keep: int) -> None:
        """Prune the model's architecture parameters."""
        self.model.prune(num_keep=num_keep)  # type: ignore

    def discretize(self) -> nn.Module:
        return self.model._discretize()  # type: ignore

    def get_genotype(self) -> NB201Gynotype:
        return self.model.genotype()  # type: ignore

    def preserve_grads(self) -> None:
        self.model.apply(preserve_grads)

    def check_grads_cosine(self, oles: bool = False) -> None:
        check_grads_cosine_part = partial(check_grads_cosine, oles=oles)
        self.model.apply(check_grads_cosine_part)

    def calc_avg_gm_score(self) -> float:
        sim_avg = []
        for module in self.model.modules():
            if hasattr(module, "running_sim"):
                sim_avg.append(module.running_sim.avg)
        if len(sim_avg) == 0:
            return 0
        avg_gm_score = sum(sim_avg) / len(sim_avg)
        return avg_gm_score

    def reset_gm_scores(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "running_sim"):
                module.running_sim.reset()

    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        return self.model._get_mean_layer_alignment_score(), 0

    def get_num_skip_ops(self) -> tuple[int, int]:
        alphas_normal = self.model.arch_parameters
        count_skip = lambda alphas: sum(alphas.argmax(dim=-1) == 1)
        return count_skip(alphas_normal), -1

    def get_num_ops(self) -> int:
        return self.model.num_ops

    def get_num_edges(self) -> int:
        return self.model.num_edges

    def get_num_nodes(self) -> int:
        raise NotImplementedError(
            "get_num_nodes is not implemented for NB201SearchSpace"
        )

    def get_candidate_flags(self, topology: bool = False) -> list:
        assert topology is False
        return self.model.candidate_flags

    def get_nodes_to_edge_mapping(self, selected_node: int) -> dict:  # type: ignore
        raise NotImplementedError(
            "get_nodes_to_edge_mapping is not implemented for NB201SearchSpace"
        )

    def remove_from_projected_weights(
        self,
        selected_edge: int,
        selected_op: int | None,
        cell_type: str | None = None,
        topology: bool = False,
    ) -> None:
        assert topology is False
        assert cell_type is None
        assert selected_op is not None
        self.model.remove_from_projected_weights(selected_edge, selected_op)

    def mark_projected_operation(
        self,
        selected_edge: int,
        selected_op: int,
        cell_type: str | None = None,
    ) -> None:
        assert cell_type is None
        self.model.mark_projected_op(selected_edge, selected_op)

    def mark_projected_edge(  # type: ignore
        self,
        selected_node: int,
        selected_edges: list[int],
        cell_type: str | None = None,
    ) -> None:
        raise NotImplementedError(
            "mark_projected_edge is not implemented for NB201SearchSpace"
        )

    def set_projection_mode(self, value: bool) -> None:
        self.model.projection_mode = value

    def set_projection_evaluation(self, value: bool) -> None:
        self.model.projection_evaluation = value
