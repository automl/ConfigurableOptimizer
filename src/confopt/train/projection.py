from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.utils

from confopt.oneshot.archsampler.darts.sampler import DARTSSampler
from confopt.searchspace import SearchSpace
from confopt.searchspace.common.base_search import (
    ArchSelectionSupport,
    OperationStatisticsSupport,
)
from confopt.train import ConfigurableTrainer, SearchSpaceHandler
from confopt.utils import unwrap_model


class PerturbationArchSelection:
    def __init__(
        self,
        trainer: ConfigurableTrainer,
        projection_criteria: str | dict,
        projection_interval: int,
        is_wandb_log: bool = False,
    ) -> None:
        self.trainer = trainer
        self.projection_criteria = projection_criteria
        self.projection_interval = projection_interval
        self.is_wandb_log = is_wandb_log

        if self.projection_criteria == "loss":
            self.crit_idx = 1
            self.compare = lambda x, y: x > y
        if self.projection_criteria == "acc":
            self.crit_idx = 0
            self.compare = lambda x, y: x < y

        self.model = self.trainer.model
        self.criterion = self.trainer.criterion

    def project_edge(
        self,
        model: SearchSpace,
        valid_queue: torch.utils.data.DataLoader,
        cell_type: str | None = None,
    ) -> tuple[int, list[int]]:
        assert (
            model.is_topology_supported()
        ), "project_edge can only be called on a searchspace that supports topology"

        candidate_flags = model.get_candidate_flags(cell_type)

        #### select an edge randomly
        remain_nids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
        selected_nid = np.random.choice(remain_nids, size=1)[0]
        self.trainer.logger.log(f"selected node: {selected_nid} {cell_type}")

        eids = deepcopy(model.get_edges_at_node(selected_node=selected_nid))
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
        cell_type: str,
        selected_eid: int | None = None,
    ) -> tuple[int, int]:
        candidate_flags = model.get_candidate_flags(cell_type)
        num_ops = model.get_num_ops()

        if selected_eid is None:
            remain_eids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
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

        self.trainer.logger.log(f"Found best op id: {best_opid}")
        return selected_eid, best_opid  # type: ignore

    def select_operation(
        self,
        model: SearchSpace | torch.nn.DataParallel,
        proj_queue: torch.utils.DataLoader,
    ) -> None:
        # Select Operation
        unwrapped_model = unwrap_model(model)
        unwrapped_model.set_topology(False)

        self.trainer.logger.log("Selecting Operation")

        cell_types = unwrapped_model.get_cell_types()

        for cell_type in cell_types:
            selected_eid, best_opid = self.project_op(
                unwrapped_model, proj_queue, cell_type
            )
            unwrapped_model.mark_projected_operation(
                selected_eid,
                best_opid,
                cell_type=cell_type,
            )

    def select_topology(
        self,
        model: SearchSpace | torch.nn.DataParallel,
        proj_queue: torch.utils.DataLoader,
    ) -> None:
        # Select Topology
        unwrapped_model = unwrap_model(model)
        unwrapped_model.set_topology(True)
        self.trainer.logger.log("Selecting Topology")

        cell_types = unwrapped_model.get_cell_types()

        for cell_type in cell_types:
            selected_nid, eids = self.project_edge(
                unwrapped_model,
                proj_queue,
                cell_type,
            )
            unwrapped_model.mark_projected_edge(selected_nid, eids, cell_type=cell_type)

    def select_architecture(self) -> None:
        network = (
            self.trainer._load_onto_data_parallel(self.model)
            if self.trainer.use_data_parallel
            else self.model
        )
        unwrapped_network = self.model

        assert isinstance(
            unwrapped_network, (ArchSelectionSupport, OperationStatisticsSupport)
        )

        train_queue, valid_queue, _ = self.trainer.data.get_dataloaders(
            batch_size=self.trainer.batch_size,
            n_workers=0,
        )
        proj_queue = valid_queue

        network.train()
        unwrapped_network.set_projection_mode(True)

        # get total tune epochs
        if unwrapped_network.is_topology_supported():
            num_projections = (
                self.model.get_num_edges() + self.model.get_num_nodes() - 1
            )
            tune_epochs = self.projection_interval * num_projections + 1
        else:
            num_projections = self.model.get_num_edges() - 1
            tune_epochs = self.projection_interval * num_projections

        if self.trainer.start_epoch == 0:
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

            # reset optimizer with lr/10
            self._reset_optimizer_and_scheduler(tune_epochs)

        # make a dummy profile
        search_space_handler = SearchSpaceHandler(
            sampler=DARTSSampler(
                arch_parameters=self.model.arch_parameters,
                sample_frequency="step",
            )
        )

        for epoch in range(self.trainer.start_epoch + 1, tune_epochs + 1):
            epoch_str = f"{epoch:03d}-{tune_epochs:03d}"
            self.trainer.logger.reset_wandb_logs()
            # project
            if (epoch - 1) % self.projection_interval == 0 or epoch == tune_epochs:
                if unwrapped_network.is_topology_supported():
                    if (
                        epoch
                        < self.projection_interval * unwrapped_network.get_num_edges()
                    ):
                        self.select_operation(network, proj_queue)
                    else:
                        self.select_topology(network, proj_queue)
                else:
                    self.select_operation(network, proj_queue)

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

            train_acc, train_loss = self.evaluate(network, train_queue)
            valid_acc, valid_loss = self.evaluate(network, valid_queue)
            self.trainer.logger.log(
                f"[DARTS-PT-Tuning] [{epoch_str}]"
                + f" train_acc: {train_acc:.3f},"
                + f" train_loss: {train_loss:.3f} |"
                + f" valid_acc: {valid_acc:.3f},"
                + f" valid_loss: {valid_loss:.3f}"
            )

            # wandb logging
            log_dict = {
                "tune/train_acc": train_acc,
                "tune/train_loss": train_loss,
                "tune/valid_acc": valid_acc,
                "tune/valid_loss": valid_loss,
            }

            self.trainer.logger.update_wandb_logs(log_dict)

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
            genotype = self.model.get_genotype().tostr()  # type: ignore
            self.trainer.logger.save_genotype(
                genotype, epoch, self.trainer.checkpointing_freq
            )

        unwrapped_network.set_projection_mode(False)

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
