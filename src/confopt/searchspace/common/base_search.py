from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn  # noqa: PLR0402

from confopt.utils import AverageMeter, reset_gm_score_attributes

if TYPE_CHECKING:
    from confopt.oneshot.base import OneShotComponent


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

    def get_arch_parameters_as_dict(self) -> dict[str, np.ndarray]:
        return {
            f"arch_parameters/{idx}": p.detach().cpu().numpy()
            for idx, p in enumerate(self.arch_parameters)
        }

    def get_sampled_weights(self) -> list[nn.Parameter]:
        return self.model.sampled_weights

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

    def get_mask(self) -> list[torch.Tensor] | None:
        """Get the mask of the model.

        Returns:
            list[torch.Tensor]: The mask of the model.
        """
        if hasattr(self.model, "mask"):
            return self.model.mask
        return None


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


@dataclass
class LambdaReg:
    epsilon_base: float = 0.001
    epsilon: float = 0.0
    corr_type: str = "corr"
    strength: float = 0.125
    enabled: bool = True


class LambdaDARTSSupport(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._assert_model_has_implementation()
        self.lambda_reg = LambdaReg()

    def get_cells(self, cell_type: str | None = None) -> list[torch.nn.Module] | None:
        return self.model.get_cells(cell_type)

    def _assert_model_has_implementation(self) -> None:
        base_error = "LambdaDARTSSupport implementation missing"

        def assert_is_function(fn_name: str) -> None:
            assert hasattr(
                self.model, fn_name
            ), f"{base_error}: {fn_name} method not found in {type(self.model)}"
            assert callable(
                self.model.get_arch_grads
            ), f"'{fn_name}' should be a method"

        assert_is_function("get_arch_grads")
        assert_is_function("get_cells")
        assert_is_function("set_lambda_perturbations")

    def set_lambda_darts_params(self, lambda_reg: LambdaReg) -> None:
        self.lambda_reg = lambda_reg

    def enable_lambda_darts(self) -> None:
        self.lambda_reg.enabled = True

    def disable_lambda_darts(self) -> None:
        self.lambda_reg.enabled = False

    def get_perturbations(self) -> list[torch.Tensor]:
        grads_normal, grads_reduce = self.model.get_arch_grads()
        alpha_normal = self.arch_parameters[0]

        def get_perturbation_for_cell(
            layer_gradients: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            with torch.no_grad():
                weight = 1 / ((len(layer_gradients) * (len(layer_gradients) - 1)) / 2)
                if self.lambda_reg.corr_type == "corr":
                    u = [g / g.norm(p=2.0) for g in layer_gradients]
                    sum_u = sum(u)
                    identity_matrix = torch.eye(sum_u.shape[0]).cuda()
                    P = [
                        (1 / g.norm(p=2.0)) * (identity_matrix - torch.ger(u_l, u_l))
                        for g, u_l in zip(layer_gradients, u)
                    ]
                    perturbations = [
                        weight * (P_l @ sum_u).reshape(alpha_normal.shape) for P_l in P
                    ]
                elif self.lambda_reg.corr_type == "signcorr":
                    perturbations = []
                    for i in range(len(layer_gradients)):
                        _dir: torch.Tensor = 0
                        for j in range(len(layer_gradients)):
                            if i == j:
                                continue
                            g, g_ = layer_gradients[i], layer_gradients[j]
                            dot, abs_dot = torch.dot(g, g_), torch.dot(
                                torch.abs(g), torch.abs(g_)
                            )
                            _dir += (
                                (
                                    torch.ones_like(g_)
                                    - (dot / abs_dot) * torch.sign(g) * torch.sign(g_)
                                )
                                * g_
                                / abs_dot
                            )
                        perturbations.append(weight * _dir.reshape(alpha_normal.shape))
            return perturbations

        pert_normal = get_perturbation_for_cell(grads_normal)
        pert_reduce = (
            get_perturbation_for_cell(grads_reduce)
            if grads_reduce is not None
            else None
        )
        pert_denom = (
            pert_normal + pert_reduce if pert_reduce is not None else pert_normal
        )

        self.lambda_reg.epsilon = (
            self.lambda_reg.epsilon_base
            / torch.cat(pert_denom, dim=0).norm(p=2.0).item()
        )

        idx_normal = 0
        idx_reduce = 0
        pert = []

        cells = self.get_cells()

        if cells is not None:
            for cell in cells:
                if pert_reduce is not None and cell.reduction:
                    pert.append(pert_reduce[idx_reduce] * self.lambda_reg.epsilon)
                    idx_reduce += 1
                else:
                    pert.append(pert_normal[idx_normal] * self.lambda_reg.epsilon)
                    idx_normal += 1

        return pert

    def add_lambda_regularization(
        self, data: torch.Tensor, target: torch.Tensor, criterion: nn.modules.loss._Loss
    ) -> None:
        if not self.lambda_reg.enabled:
            return

        pert = self.get_perturbations()

        loss_fn = criterion
        # Calculate forward and backward gradients to compute finite difference
        self.model.set_lambda_perturbations(pert)
        forward_grads = torch.autograd.grad(
            loss_fn(self.model(data)[1], target),
            self.model_weight_parameters(),
            allow_unused=True,
        )
        self.model.set_lambda_perturbations([-p for p in pert])
        backward_grads = torch.autograd.grad(
            loss_fn(self.model(data)[1], target),
            self.model_weight_parameters(),
            allow_unused=True,
        )

        reg_grad = [
            (f - b).div_(2 * self.lambda_reg.epsilon)
            if (f is not None and b is not None)
            else 0.0
            for f, b in zip(forward_grads, backward_grads)
        ]
        for param, grad in zip(self.model_weight_parameters(), reg_grad):
            if param.grad is not None:
                param.grad.data.add_(self.lambda_reg.strength * grad)


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
        bool: Flag showing topology search is supported or not for the SearchSpace.
        """

    def set_topology(self, value: bool) -> None:
        """Set flag showing toplogy search is active for model or not."""
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
        cell_type: Any | None = None,  # noqa: ARG002
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

    @abstractmethod
    def get_max_input_edges_at_node(self, selected_node: int) -> int:
        """Gets the number of edges allowed on a node after discretization.

        Returns:
            int: max number of edges from the nodes after discretization
        """

    @abstractmethod
    def get_projected_arch_parameters(self) -> list[torch.Tensor]:
        """Gets the projected arch parameters used in the forward pass.

        Returns:
            list[torch.Tensor] : list of projected arch patameters
        """


class DrNASRegTermSupport(ModelWrapper):
    @abstractmethod
    def get_drnas_anchors(self) -> list[torch.Tensor]:
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


class FairDARTSRegTermSupport(ModelWrapper):
    @abstractmethod
    def get_fair_darts_arch_parameters(self) -> list[torch.Tensor]:
        """Get the arch parameters used in FairDARTS.

        Returns:
            torch.Tensor: The FairDARTS regularization term of the model.
        """
        ...
