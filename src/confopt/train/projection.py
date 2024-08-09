from __future__ import annotations

from copy import deepcopy
from enum import Enum

import numpy as np
import torch
import torch.utils

from confopt.oneshot.archsampler.darts.sampler import DARTSSampler
from confopt.searchspace import SearchSpace
from confopt.train import ConfigurableTrainer, SearchSpaceHandler


# rewrote SearchSpaceType to avoid circular import
class SearchSpaceType(Enum):
    DARTS = "darts"
    NB201 = "nb201"
    NB1SHOT1 = "nb1shot1"
    TNB101 = "tnb101"
    BABYDARTS = "baby_darts"
    RobustDARTS = "robust_darts"


class PerturbationArchSelection:
    def __init__(
        self,
        trainer: ConfigurableTrainer,
        projection_criteria: str | dict,
        projection_interval: int,
        edge_decision: str = "random",
        searchspace_type: str = "nb201",
        is_wandb_log: bool = False,
    ) -> None:
        self.trainer = trainer
        self.searchspace_type = SearchSpaceType(searchspace_type)
        self.projection_criteria = projection_criteria
        self.projection_interval = projection_interval
        self.edge_decision = edge_decision
        self.is_wandb_log = is_wandb_log

        if self.projection_criteria == "loss":
            self.crit_idx = 1
            self.compare = lambda x, y: x > y
        if self.projection_criteria == "acc":
            self.crit_idx = 0
            self.compare = lambda x, y: x < y

    def project_edge(
        self,
        model: SearchSpace,
        valid_queue: torch.utils.data.DataLoader,
        cell_type: str | None = None,
    ) -> tuple[int, list[int]]:
        # only called for DARTS
        candidate_flags = model.get_candidate_flags(topology=True)[cell_type]

        #### select an edge randomly
        remain_nids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
        if self.edge_decision == "random":
            selected_nid = np.random.choice(remain_nids, size=1)[0]
            self.trainer.logger.log(f"selected node: {selected_nid} {cell_type}")

        eids = deepcopy(model.get_nodes_to_edge_mapping(selected_node=selected_nid))
        while len(eids) > 2:
            eid_to_del = None
            crit_extrema = None
            for eid in eids:
                model.remove_from_projected_weights(eid, None, cell_type, topology=True)

                model.set_projection_evaluation(True)
                ## proj evaluation
                valid_stats = self.evaluate(model, valid_queue)
                model.set_projection_evaluation(False)

                crit = valid_stats[self.crit_idx]

                if crit_extrema is None or not self.compare(
                    crit, crit_extrema
                ):  # find out bad edges
                    crit_extrema = crit
                    eid_to_del = eid
                # self.trainer.logger.log("valid_acc %f", valid_stats[0])
                # self.trainer.logger.log("valid_loss %f", valid_stats[1])
            eids.remove(eid_to_del)

        self.trainer.logger.log(f"Found top2 edges: ({eids[0]}, {eids[1]})")
        return selected_nid, eids

    def project_op(
        self,
        model: SearchSpace,
        valid_queue: torch.utils.data.DataLoader,
        cell_type: str | None = None,
        selected_eid: int | None = None,
    ) -> tuple[int, int]:
        if self.searchspace_type == SearchSpaceType.DARTS:
            assert cell_type is not None
            candidate_flags = model.get_candidate_flags()[cell_type]
        else:
            candidate_flags = model.get_candidate_flags()
        num_ops = model.get_num_ops()

        if selected_eid is None:
            remain_eids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
            if self.edge_decision == "random":
                selected_eid = np.random.choice(remain_eids, size=1)[0]
                self.trainer.logger.log(f"selected Edge: {selected_eid}")

        ## select the best operation
        best_opid = 0
        crit_extrema = None
        for opid in range(num_ops):
            # remove operation
            model.remove_from_projected_weights(selected_eid, opid, cell_type)

            # proj evaluation
            model.set_projection_evaluation(True)
            valid_stats = self.evaluate(model, valid_queue)
            model.set_projection_evaluation(False)
            crit = valid_stats[self.crit_idx]

            if crit_extrema is None or self.compare(crit, crit_extrema):
                crit_extrema = crit
                best_opid = opid
            # self.trainer.logger.log("valid_acc %f", valid_stats[0])
            # self.trainer.logger.log("valid_loss %f", valid_stats[1])

        self.trainer.logger.log(f"Found best op id: {best_opid}")
        return selected_eid, best_opid  # type: ignore

    def select_operation(
        self, model: SearchSpace, proj_queue: torch.utils.DataLoader
    ) -> None:
        self.trainer.logger.log("Selecting Operation")
        if self.searchspace_type == SearchSpaceType.DARTS:
            selected_eid_normal, best_opid_normal = self.project_op(
                model, proj_queue, cell_type="normal"
            )
            model.mark_projected_operation(
                selected_eid_normal, best_opid_normal, cell_type="normal"
            )
            selected_eid_reduce, best_opid_reduce = self.project_op(
                model, proj_queue, cell_type="reduce"
            )
            model.mark_projected_operation(
                selected_eid_reduce, best_opid_reduce, cell_type="normal"
            )
        else:
            selected_eid, best_opid = self.project_op(model, proj_queue)
            model.mark_projected_operation(selected_eid, best_opid)

    def select_topology(
        self, model: SearchSpace, proj_queue: torch.utils.DataLoader
    ) -> None:
        # Select Topology
        assert (
            self.searchspace_type != SearchSpaceType.NB201
        ), "NB201SearchSpace does not support topolgy selection"
        self.trainer.logger.log("Selecting Topology")
        selected_nid_normal, eids_normal = self.project_edge(
            model, proj_queue, cell_type="normal"
        )
        model.mark_projected_edge(selected_nid_normal, eids_normal, cell_type="normal")
        selected_nid_reduce, eids_reduce = self.project_edge(
            model, proj_queue, cell_type="reduce"
        )
        model.mark_projected_edge(selected_nid_reduce, eids_reduce, cell_type="reduce")

    def select_architecture(self) -> None:  # noqa: C901, PLR0912
        self.model = self.trainer.model

        if self.trainer.use_data_parallel:
            network, self.criterion = self.trainer._load_onto_data_parallel(
                self.model, self.trainer.criterion
            )
        else:
            network = self.model  # type: ignore
            self.criterion = self.criterion

        train_queue, valid_queue, _ = self.trainer.data.get_dataloaders(
            batch_size=self.trainer.batch_size,
            n_workers=0,
        )
        proj_queue = valid_queue

        network.train()

        if self.trainer.use_data_parallel:
            network.module.set_projection_mode(True)
        else:
            network.set_projection_mode(True)

        # Initial Evaluation
        train_acc, train_obj = self.evaluate(network, train_queue)
        valid_acc, valid_obj = self.evaluate(network, valid_queue)

        self.trainer.logger.log(
            "[DARTS-PT-Tuning] Initial Evaluation "
            + f" train_acc: {train_acc:.3f},"
            + f" train_loss: {train_obj:.3f} |"
            + f" valid_acc: {valid_acc:.3f},"
            + f" valid_loss: {valid_obj:.3f}"
        )

        # get total tune epochs
        if self.searchspace_type == SearchSpaceType.NB201:
            num_projections = self.model.get_num_edges() - 1
            tune_epochs = self.projection_interval * num_projections
        else:
            num_projections = (
                self.model.get_num_edges() + self.model.get_num_nodes() - 1
            )
            tune_epochs = self.projection_interval * num_projections + 1

        # reset optimizer with lr/10
        if self.trainer.start_epoch == 0:
            self._reset_optimizer_and_scheduler(tune_epochs)

        # make a dummy profile
        search_space_handler = SearchSpaceHandler(
            sampler=DARTSSampler(
                arch_parameters=self.model.arch_parameters,
                sample_frequency="step",
            )
        )

        for epoch in range(self.trainer.start_epoch, tune_epochs):
            self.trainer.logger.reset_wandb_logs()
            # project
            if epoch % self.projection_interval == 0 or epoch == tune_epochs - 1:
                if self.searchspace_type == SearchSpaceType.NB201:
                    self.select_operation(self.model, proj_queue)
                elif self.searchspace_type == SearchSpaceType.DARTS:
                    if epoch < self.projection_interval * self.model.get_num_edges():
                        self.select_operation(self.model, proj_queue)
                    else:
                        self.select_topology(self.model, proj_queue)

            # TUNE
            self.trainer._train_epoch(
                search_space_handler,
                train_queue,
                valid_queue,
                network,
                self.criterion,
                self.trainer.model_optimizer,
                self.trainer.arch_optimizer,
                self.trainer.print_freq,
            )

            train_acc, train_obj = self.evaluate(network, train_queue)
            valid_acc, valid_obj = self.evaluate(network, valid_queue)
            self.trainer.logger.log(
                f"[DARTS-PT-Tuning] [Epoch {epoch}]"
                + f" train_acc: {train_acc:.3f},"
                + f" train_loss: {train_obj:.3f} |"
                + f" valid_acc: {valid_acc:.3f},"
                + f" valid_loss: {valid_obj:.3f}"
            )

            # wandb logging
            arch_values_dict = self.trainer.get_arch_values_as_dict(network)
            self.trainer.logger.update_wandb_logs(arch_values_dict)

            with torch.no_grad():
                for i, alpha in enumerate(self.model.arch_parameters):
                    self.trainer.logger.log(f"alpha {i} is {alpha}")

            if self.is_wandb_log:
                self.trainer.logger.push_wandb_logs()

            checkpointables = self.trainer._get_checkpointables(epoch=epoch)
            self.trainer.periodic_checkpointer.step(
                iteration=epoch, checkpointables=checkpointables
            )

    def evaluate(
        self,
        model: SearchSpace | torch.nn.DataParallel,
        eval_queue: torch.utils.data.DataLoader,
    ) -> tuple[float, float]:
        valid_metric = self.trainer.evaluate(eval_queue, model, self.criterion)
        return valid_metric.acc_top1, valid_metric.loss

    def _reset_optimizer_and_scheduler(self, tune_epochs: int) -> None:
        optimizer_hyperparameters = self.trainer.model_optimizer.defaults
        optimizer_hyperparameters["lr"] = optimizer_hyperparameters["lr"] / 10

        self.trainer.model_optimizer = type(self.trainer.model_optimizer)(
            self.model.model_weight_parameters(),  # type: ignore
            **optimizer_hyperparameters,
        )

        scheduler_config = {}
        if isinstance(
            self.trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
        ):
            scheduler_config = {
                "T_max": tune_epochs,
                "eta_min": self.trainer.scheduler.eta_min,
            }

        if isinstance(
            self.trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ):
            scheduler_config = {
                "T_0": tune_epochs,
                "T_mult": self.trainer.scheduler.T_mult,
                "eta_min": self.trainer.scheduler.eta_min,
            }

        self.trainer.scheduler = type(self.trainer.scheduler)(
            self.trainer.model_optimizer,
            **scheduler_config,
        )
