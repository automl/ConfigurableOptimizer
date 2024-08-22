from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import torch
import torch.nn as nn  # noqa: PLR0402

from confopt.oneshot.base_component import OneShotComponent
from confopt.utils import reset_gm_score_attributes


class ModelWrapper(nn.Module, ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


class SearchSpace(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.components: list[OneShotComponent] = []

    @property
    @abstractmethod
    def arch_parameters(self) -> list[nn.Parameter]:
        pass

    @property
    @abstractmethod
    def beta_parameters(self) -> list[nn.Parameter] | None:
        pass

    @abstractmethod
    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        pass

    def get_cell_types(self) -> list[str]:
        return ["normal"]

    def set_sample_function(self, sample_function: Callable) -> None:
        self.model.sample = sample_function

    def model_weight_parameters(self) -> list[nn.Parameter]:
        arch_param_ids = {id(p) for p in getattr(self, "arch_parameters", [])}
        beta_param_ids = {id(p) for p in getattr(self, "beta_parameters", [])}

        all_parameters = [
            p
            for p in self.model.parameters()
            if id(p) not in arch_param_ids and id(p) not in beta_param_ids
        ]

        return all_parameters

    def prune(self, num_keep: int) -> None:
        """Prune the candidates operations of the supernet."""
        self.model.prune(num_keep=num_keep)  # type: ignore

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)  # type: ignore

    def new_epoch(self) -> None:
        for component in self.components:
            component.new_epoch()

    def new_step(self) -> None:
        for component in self.components:
            component.new_step()


class ArchAttentionSupport(ModelWrapper):
    def set_arch_attention(self, enabled: bool) -> None:
        """Enable or disable attention between architecture parameters."""
        self.model.is_arch_attention_enabled = enabled


class GradientMatchingScoreSupport(ModelWrapper):
    @abstractmethod
    def preserve_grads(self) -> None:
        """Preserve the gradients of the model for gradient matching later."""
        ...

    @abstractmethod
    def update_gradient_matching_scores(
        self,
        early_stop: bool = False,
        early_stop_frequency: int = 20,
        early_stop_threshold: float = 0.4,
    ) -> None:
        """Update the gradient matching scores of the model."""
        ...

    def calc_avg_gm_score(self) -> float:
        """Calculate the average gradient matching score of the model.

        Returns:
            float: The average gradient matching score of the model.
        """
        sim_avg = []
        for module in self.model.modules():
            if hasattr(module, "running_sim"):
                sim_avg.append(module.running_sim.avg)
        if len(sim_avg) == 0:
            return 0
        avg_gm_score = sum(sim_avg) / len(sim_avg)
        return avg_gm_score

    def reset_gm_score_attributes(self) -> None:
        """Reset the gradient matching score attributes of the model."""
        for module in self.modules():
            reset_gm_score_attributes(module)

    def reset_gm_scores(self) -> None:
        """Reset the gradient matching scores of the model."""
        for module in self.model.modules():
            if hasattr(module, "running_sim"):
                module.running_sim.reset()


class LayerAlignmentScoreSupport(ModelWrapper):
    @abstractmethod
    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        """Get the mean layer alignment score of the model.

        Returns:
            tuple[float, float]: The mean layer alignment score of the normal
            and reduction cell.

        """


class OperationStatisticsSupport(ModelWrapper):
    @abstractmethod
    def get_num_skip_ops(self) -> dict[str, int]:
        """Get the number of skip operations in the model.

        Returns:
            dict[str, int]: A dictionary containing the number of skip operations
            in different types of cells. E.g., for DARTS, the dictionary would
            contain the keys "skip_connections/normal" and "skip_connections/reduce"
            with the number of skip operations.
            In NB201, the dictionary would contain only "skip_connections/normal".
        """

    def get_op_stats(self) -> dict[str, Any]:
        """Get the all the candidate operation statistics of the model."""
        skip_ops_stats = self.get_num_skip_ops()

        all_stats = {}
        all_stats.update(skip_ops_stats)
        # all_stats.update(other_stats) # Add other stats here

        return all_stats

    @abstractmethod
    def get_num_ops(self) -> int:
        """Get number of operations in an edge of a cell of the model.

        Returns:
            int: Number of operations
        """

    @abstractmethod
    def get_num_edges(self) -> int:
        """Get number of edges in a cell of the model.

        Returns:
            int: Number of rdges
        """

    @abstractmethod
    def get_num_nodes(self) -> int:
        """Get number of nodes in a cell of the model.

        Returns:
            int: Number of nodes
        """


class ArchSelectionSupport(ModelWrapper):
    @abstractmethod
    def is_topology_supported(self) -> bool:
        """Returns:
        bool: Whether topology search is supported or not for the SearchSpace.
        """

    def set_topology(self, value: bool) -> None:
        """Set whether toplogy search is active for model or not."""
        self.topology = value

    @abstractmethod
    def get_candidate_flags(self, cell_type: Literal["normal", "reduce"]) -> list:
        """Get a list of candidate flags for selecting architecture.

        The candidate flags can be for edges or operations depending on whether
        topology is active or not.

        Returns:
            list: list of candidate flags
        """

    def get_edges_at_node(  # type: ignore
        self, selected_node: int  # noqa: ARG002
    ) -> list:
        """Get a list of edges at a node.

        Returns:
            list: list of outgoing edges from the selected node.
        """
        assert (
            self.is_topology_supported()
        ), "Topology should be supported for this function"

    @abstractmethod
    def remove_from_projected_weights(
        self,
        selected_edge: int,
        selected_op: int | None,
        cell_type: Literal["normal", "reduce"] = "normal",
    ) -> None:
        """Remove an operation or a edge (depending on topology) from the
        projected weights.
        """

    @abstractmethod
    def mark_projected_operation(
        self,
        selected_edge: int,
        selected_op: int,
        cell_type: Literal["normal", "reduce"],
    ) -> None:
        """Mark an operation on a given edge (of the cell type) in the candidate flags
        and projected weights to be already projected.
        """

    def mark_projected_edge(  # type: ignore
        self,
        selected_node: int,  # noqa: ARG002
        selected_edges: list[int],  # noqa: ARG002
        cell_type: str | None = None,  # noqa: ARG002
    ) -> None:
        """Mark an operation on a given edge (of the cell type) in the candidate flags
        and projected weights to be already projected.
        """
        assert (
            self.is_topology_supported()
        ), "Topology should be supported for this function"

    @abstractmethod
    def set_projection_mode(self, value: bool) -> None:
        """Set the model into projection mode.

        When projection mode is True, the weights used in forward are candidate weights.
        """

    @abstractmethod
    def set_projection_evaluation(self, value: bool) -> None:
        """Set the model into projection mode.

        When projection mode is True, the weights used in forward are the
        projected weights.
        """
