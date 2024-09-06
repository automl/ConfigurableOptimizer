from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import torch
import torch.nn as nn  # noqa: PLR0402

from confopt.oneshot.base_component import OneShotComponent
from confopt.utils import AverageMeter, reset_gm_score_attributes


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

    def prune(self, prune_fraction: float) -> None:
        """Prune the candidates operations of the supernet."""
        self.model.prune(prune_fraction=prune_fraction)  # type: ignore

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)  # type: ignore

    def new_epoch(self) -> None:
        for component in self.components:
            component.new_epoch()

    def new_step(self) -> None:
        for component in self.components:
            component.new_step()

    def get_num_ops(self) -> int:
        """Get number of operations in an edge of a cell of the model.

        Returns:
            int: Number of operations
        """
        raise NotImplementedError("get_num_ops is not implemented for this searchpace")

    def get_num_edges(self) -> int:
        """Get number of edges in a cell of the model.

        Returns:
            int: Number of edges
        """
        raise NotImplementedError(
            "get_num_edges is not implemented for this searchpace"
        )

    def get_num_nodes(self) -> int:
        """Get number of nodes in a cell of the model.

        Returns:
            int: Number of nodes
        """
        raise NotImplementedError(
            "get_num_nodes is not implemented for this searchpace"
        )


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
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.score_types = ["mean", "first_last"]
        self.cell_types = ["normal", "reduce"]
        self.layer_alignment_meters: dict[str, dict] = {
            score_type: {} for score_type in self.score_types
        }

        for score_type in self.score_types:
            for cell_type in self.cell_types:
                self.layer_alignment_meters[score_type][cell_type] = AverageMeter()

    def get_layer_alignment_scores_as_strings(self) -> list[str]:
        """Get the layer alignment scores of the model as strings.

        Returns:
            list[str]: A list containing the layer alignment scores of the model
            as strings.
        """
        layer_alignment_scores = []

        for score_type in self.score_types:
            for cell_type in self.cell_types:
                layer_alignment_scores.append(
                    f"Layer Alignment Score ({score_type}) for cell type: {cell_type}: "
                    + f"{self.layer_alignment_meters[score_type][cell_type].avg:.4f}"
                )

        return layer_alignment_scores

    def reset_layer_alignment_scores(self) -> None:
        """Reset the layer alignment scores of the model."""
        for score_type in self.score_types:
            for cell_type in self.cell_types:
                self.layer_alignment_meters[score_type][cell_type].reset()

    def update_layer_alignment_scores(self) -> None:
        """Update the layer alignment scores of the model."""
        # Update the "mean" scores
        score_normal, score_reduce = self.get_mean_layer_alignment_score()
        self.layer_alignment_meters["mean"]["normal"].update(val=score_normal)
        self.layer_alignment_meters["mean"]["reduce"].update(val=score_reduce)

        # Update the "first_last" scores
        (
            score_normal_first,
            score_normal_last,
        ) = self.get_first_and_last_layer_alignment_score()
        self.layer_alignment_meters["first_last"]["normal"].update(
            val=score_normal_first
        )
        self.layer_alignment_meters["first_last"]["reduce"].update(
            val=score_normal_last
        )

    def get_layer_alignment_scores(self) -> dict[str, Any]:
        """Get the layer alignment scores of the model.

        Returns:
            dict[str, Any]: A dictionary containing the layer alignment scores of
            the model.
        """
        layer_alignment_scores = {}
        for score_type in self.score_types:
            for cell_type in self.cell_types:
                layer_alignment_scores[
                    f"layer_alignment_scores/{score_type}/{cell_type}"
                ] = self.layer_alignment_meters[score_type][cell_type].avg

        return layer_alignment_scores

    @abstractmethod
    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        """Get the mean layer alignment score of the model.

        Returns:
            tuple[float, float]: The mean layer alignment score of the normal
            and reduction cell.

        """

    @abstractmethod
    def get_first_and_last_layer_alignment_score(self) -> tuple[float, float]:
        """Get the layer alignment score for the first and last layer of the model.

        Returns:
            tuple[float, float]: The layer alignment score of the first and last layer
            of normal and reduction cells.

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


class PerturbationArchSelectionSupport(ModelWrapper):
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


class DrNASRegTermSupport(ModelWrapper):
    @abstractmethod
    def get_drnas_anchors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the anchors used in DrNAS.

        Returns:
            torch.Tensor: The DrNAS regularization term of the model.
        """
        ...


class FLOPSRegTermSupport(ModelWrapper):
    @abstractmethod
    def get_weighted_flops(self) -> torch.Tensor:
        """Get the FLOPS regularization term of the model.

        Returns:
            torch.Tensor: The FLOPS regularization term of the model.
        """
        ...


class InsertCellSupport(ModelWrapper):
    @abstractmethod
    def insert_new_cells(self, num_cells: int) -> None:
        """Insert new cells in the model.

        Args:
            num_cells (int): Number of cells to insert.
        """
        ...

    @abstractmethod
    def create_new_cell(self, pos: int) -> nn.Module:
        """Create a new cell in the model.

        Args:
            pos (int): Position to insert the new cell.

        Returns:
            nn.Module: The new cell.
        """
        ...


class GradientStatsSupport(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.model.is_gradient_stats_enabled = True
        self.n_cells = len(self.model.cells)
        self.cell_grads_meters = {idx: AverageMeter() for idx in range(self.n_cells)}
        self.arch_grads_meters = {
            idx: AverageMeter() for idx in range(len(self.arch_parameters))
        }
        self.arch_row_grads_meters: dict[str, AverageMeter] = {}

        for idx, param in enumerate(self.arch_parameters):
            for row_idx in range(param.size(0)):
                self.arch_row_grads_meters[
                    f"arch_param_{idx}_row_{row_idx}"
                ] = AverageMeter()

    def reset_grad_stats(self) -> None:
        for cell_grad_meter in self.cell_grads_meters.values():
            cell_grad_meter.reset()

        for arch_grad_meter in self.arch_grads_meters.values():
            arch_grad_meter.reset()

        for row_grad_meter in self.arch_row_grads_meters.values():
            row_grad_meter.reset()

    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        total_norm = 0.0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        return total_norm

    def get_grad_stats(self) -> dict[str, Any]:
        """Get the gradient statistics of the model.

        Returns:
            dict[str, Any]: A dictionary containing the gradient statistics
            of the model.
        """
        cell_grad_stats = self.get_cell_grad_stats()
        arch_params_grad_stats = self.get_arch_params_grad_norm()

        all_stats = {}
        all_stats.update(cell_grad_stats)
        all_stats.update(arch_params_grad_stats)
        # all_stats.update(other_stats) # Add other stats here

        return all_stats

    def get_cell_grad_stats(self) -> dict[str, Any]:
        """Get the gradient statistics of the cells in the model.

        Returns:
            dict[str, Any]: A dictionary containing the gradient statistics of
            the cells in the model.
        """
        cell_grad_stats = {}
        for idx, cell_grad_meter in self.cell_grads_meters.items():
            cell_grad_stats[
                f"gradient_stats/cell_{idx}_grad_norm"
            ] = cell_grad_meter.avg

        return cell_grad_stats

    def get_arch_params_grad_norm(self) -> dict[str, Any]:
        """Get the gradient norm of the architecture parameters of the model.

        Returns:
            dict[str, Any]: A dictionary containing the gradient norm of the
            architecture parameters of the model.
        """
        grad_stats: dict[str, Any] = {}

        for idx, param in enumerate(self.arch_parameters):
            grad_stats[
                f"gradient_stats/total_arch_param_{idx}_grad_norm"
            ] = self.arch_grads_meters[idx].avg

            for row_idx, _row in enumerate(param):
                grad_stats[
                    f"gradient_stats/arch_param_{idx}_row_{row_idx}_grad_norm"
                ] = self.arch_row_grads_meters[f"arch_param_{idx}_row_{row_idx}"].avg

        return grad_stats

    def update_cell_grad_stats(self) -> None:
        """Compute the gradient statistics of the cells in the model."""
        for idx, cell in enumerate(self.model.cells):
            self.cell_grads_meters[idx].update(self._calculate_gradient_norm(cell))

    def update_arch_params_grad_stats(self) -> None:
        """Compute the gradient norm of the architecture parameters of the model."""
        for idx, param in enumerate(self.arch_parameters):
            if param.grad is None:
                continue

            self.arch_grads_meters[idx].update(param.grad.data.norm(2).item())

            for row_idx, row in enumerate(param.grad.data):
                self.arch_row_grads_meters[f"arch_param_{idx}_row_{row_idx}"].update(
                    row.norm(2).item()
                )
