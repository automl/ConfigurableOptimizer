from __future__ import annotations

import time

from fvcore.common.checkpoint import Checkpointer
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.modules.loss import _Loss as Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from confopt.dataset import AbstractData
from confopt.searchspace import SearchSpace
from confopt.train import DEBUG_STEPS, ConfigurableTrainer, TrainingMetrics
from confopt.utils import AverageMeter, Logger, calc_accuracy, unwrap_model
import confopt.utils.distributed as dist_utils


class DiscreteTrainer(ConfigurableTrainer):
    def __init__(
        self,
        model: nn.Module,
        data: AbstractData,
        model_optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: Loss,
        logger: Logger,
        batch_size: int,
        use_ddp: bool = False,
        print_freq: int = 2,
        drop_path_prob: float = 0.1,
        aux_weight: float = 0.0,
        model_to_load: str | int | None = None,
        checkpointing_freq: int = 20,
        epochs: int = 100,
        debug_mode: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            data=data,
            model_optimizer=model_optimizer,
            arch_optimizer=None,
            scheduler=scheduler,
            criterion=criterion,
            logger=logger,
            batch_size=batch_size,
            print_freq=print_freq,
            drop_path_prob=drop_path_prob,
            model_to_load=model_to_load,
            checkpointing_freq=checkpointing_freq,
            epochs=epochs,
            debug_mode=debug_mode,
        )
        self.use_ddp = use_ddp
        self.aux_weight = aux_weight
        # self.use_supernet_checkpoint = use_supernet_checkpoint

    def average_metrics_across_workers(
        self, metrics: TrainingMetrics
    ) -> TrainingMetrics | None:
        if not dist.is_initialized():
            return metrics

        rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
        metrics_tensor = torch.tensor(
            [metrics.loss, metrics.acc_top1, metrics.acc_top5]
        ).cuda()

        dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            metrics_tensor /= world_size
            metrics = TrainingMetrics(*metrics_tensor.cpu().tolist())
            return metrics

        return None

    def _train_epoch(  # type: ignore
        self,
        network: SearchSpace | DistributedDataParallel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Loss,
        rank: int,
        epoch: int,
        total_epochs: int,
        log_with_wandb: bool,
    ) -> None:
        start_time = time.time()
        train_time, epoch_time = AverageMeter(), AverageMeter()

        if (
            self.logger.search_space == "darts"
        ):  # FIXME: This is hacky. Should be fixed.
            unwrap_model(network).drop_path_prob = (
                self.drop_path_prob * epoch / total_epochs
            )

        base_metrics = self._train(
            train_loader,
            network,
            criterion,
            self.print_freq,
        )
        base_metrics = self.average_metrics_across_workers(base_metrics)  # type:ignore

        # Logging
        if rank == 0:
            epoch_str = f"{epoch:03d}-{total_epochs:03d}"
            self.logger.reset_wandb_logs()
            train_time.update(time.time() - start_time)
            self.logger.log_metrics(
                "[Discrete] Train: Model/Network metrics ",
                base_metrics,
                epoch_str,
                train_time.sum,
            )

        if val_loader is not None:
            valid_metrics = self.evaluate(val_loader, network, criterion)
            valid_metrics = self.average_metrics_across_workers(
                valid_metrics
            )  # type:ignore

        # Logging
        if rank == 0:
            if val_loader is not None:
                self.logger.log_metrics(
                    "[Discrete] Evaluation: ", valid_metrics, epoch_str
                )
                self.logger.add_wandb_log_metrics("discrete/eval", valid_metrics, epoch)

                (
                    self.valid_losses[epoch],
                    self.valid_accs_top1[epoch],
                    self.valid_accs_top5[epoch],
                ) = valid_metrics
            self.logger.add_wandb_log_metrics(
                "discrete/train/model", base_metrics, epoch, train_time.sum
            )

            (
                self.search_losses[epoch],
                self.search_accs_top1[epoch],
                self.search_accs_top5[epoch],
            ) = base_metrics

            checkpointables = self._get_checkpointables(epoch=epoch)
            self.periodic_checkpointer.step(
                iteration=epoch, checkpointables=checkpointables
            )

            if log_with_wandb:
                self.logger.push_wandb_logs()

            if base_metrics.acc_top1 > self.search_accs_top1["best"]:
                self.search_accs_top1["best"] = base_metrics.acc_top1
                self.logger.log(
                    f"<<<--->>> The {epoch_str}-th epoch : found the highest "
                    + f"validation accuracy : {base_metrics.acc_top1:.2f}%."
                )

                self.best_model_checkpointer.save(
                    name="best_model", checkpointables=checkpointables
                )

            if epoch == total_epochs - 1:
                self.checkpointer.save(
                    name="model_final", checkpointables=checkpointables
                )

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    def train(self, epochs: int, log_with_wandb: bool = True) -> None:  # type: ignore
        self.epochs = epochs

        self._init_experiment_state()

        if hasattr(self.model, "arch_parametes"):
            assert self.model.arch_parametes == [None]

        if self.use_ddp:
            local_rank, rank, world_size = (
                dist_utils.get_local_rank(),
                dist_utils.get_rank(),
                dist_utils.get_world_size(),
            )
            dist_utils.print_on_master_only(rank == 0)
            network, criterion = self._load_onto_distributed_data_parallel(
                self.model, self.criterion
            )
        else:
            local_rank, rank, world_size = 0, 0, 1  # noqa: F841
            network: nn.Module = self.model  # type: ignore
            criterion = self.criterion

        train_loader, val_loader, _ = self.data.get_dataloaders(
            batch_size=self.batch_size,
            n_workers=0,  # FIXME: This looks suboptimal
            use_distributed_sampler=self.use_ddp,
        )

        for epoch in range(self.start_epoch + 1, epochs + 1):
            self._train_epoch(
                network=network,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                rank=rank,
                epoch=epoch,
                total_epochs=epochs,
                log_with_wandb=log_with_wandb,
            )

    def _train(
        self,
        train_loader: DataLoader,
        network: SearchSpace | DistributedDataParallel,
        criterion: Loss,
        print_freq: int,
    ) -> TrainingMetrics:
        data_time, batch_time = AverageMeter(), AverageMeter()
        base_losses, base_top1, base_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        network.train()
        end = time.time()

        for _step, (base_inputs, base_targets) in enumerate(train_loader):
            # FIXME: What was the point of this? and is it safe to remove?
            # scheduler.update(None, 1.0 * step / len(xloader))

            base_inputs = base_inputs.to(self.device)
            base_targets = base_targets.to(self.device, non_blocking=True)

            # measure data loading time
            data_time.update(time.time() - end)

            self.model_optimizer.zero_grad()
            logits_aux, logits = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            if (
                hasattr(unwrap_model(network), "_auxiliary")
                and unwrap_model(network)._auxiliary
                and self.aux_weight > 0.0
            ):
                loss_aux = criterion(logits_aux, base_targets)
                base_loss += self.aux_weight * loss_aux
            base_loss.backward()

            torch.nn.utils.clip_grad_norm_(unwrap_model(network).parameters(), 5)

            self.model_optimizer.step()

            self._update_meters(
                inputs=base_inputs,
                logits=logits,
                targets=base_targets,
                loss=base_loss,
                loss_meter=base_losses,
                top1_meter=base_top1,
                top5_meter=base_top5,
            )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.debug_mode and _step > DEBUG_STEPS:
                self.logger.log(f"DEBUG MODE: Breaking after {DEBUG_STEPS}")
                break

            if _step % print_freq == 0 or _step + 1 == len(train_loader):
                # TODO: what is this doing ?
                ...

        base_metrics = TrainingMetrics(base_losses.avg, base_top1.avg, base_top5.avg)
        return base_metrics

    def test(self, log_with_wandb: bool = True) -> TrainingMetrics:
        test_losses, test_top1, test_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        self.logger.reset_wandb_logs()
        if self.use_ddp is True:
            local_rank, rank, world_size = (  # noqa: F841
                dist_utils.get_local_rank(),
                dist_utils.get_rank(),
                dist_utils.get_world_size(),
            )
            dist_utils.print_on_master_only(rank == 0)
            network, criterion = self._load_onto_distributed_data_parallel(
                self.model, self.criterion
            )
        else:
            network: nn.Module = self.model  # type:ignore
            criterion = self.criterion
        network.eval()

        *_, test_loader = self.data.get_dataloaders(
            batch_size=self.batch_size,
            n_workers=0,  # FIXME: This looks suboptimal
            use_distributed_sampler=self.use_ddp,
        )

        with torch.no_grad():
            for _step, (test_inputs, test_targets) in enumerate(test_loader):
                test_inputs = test_inputs.to(self.device)
                test_targets = test_targets.to(self.device, non_blocking=True)

                _, logits = network(test_inputs)
                test_loss = criterion(logits, test_targets)

                test_prec1, test_prec5 = calc_accuracy(
                    logits.data, test_targets.data, topk=(1, 5)
                )

                test_losses.update(test_loss.item(), test_inputs.size(0))
                test_top1.update(test_prec1.item(), test_inputs.size(0))
                test_top5.update(test_prec5.item(), test_inputs.size(0))

                if self.debug_mode and _step > DEBUG_STEPS:
                    self.logger.log(f"DEBUG MODE: Breaking after {DEBUG_STEPS}")
                    break

        test_metrics = TrainingMetrics(test_losses.avg, test_top1.avg, test_top5.avg)
        test_metrics = self.average_metrics_across_workers(test_metrics)  # type: ignore

        if dist_utils.get_rank() == 0:
            self.logger.add_wandb_log_metrics("discrete/test", test_metrics)
            if log_with_wandb:
                self.logger.push_wandb_logs()

            self.logger.log_metrics("[Discrete] Test", test_metrics, epoch_str="---")

        return test_metrics

    def _set_up_checkpointer(self, mode: str | None) -> Checkpointer:
        checkpoint_dir = self.logger.path(mode=mode)  # todo: check this
        # checkpointables = self._get_checkpointables(self.start_epoch)

        checkpointables = {
            "w_scheduler": self.scheduler,
            "w_optimizer": self.model_optimizer,
        }
        checkpointer = Checkpointer(
            model=self.model,
            save_dir=str(checkpoint_dir),
            save_to_disk=True,
            **checkpointables,
        )
        return checkpointer

    # def _load_model_state_if_exists(self) -> None:
    #     self.best_model_checkpointer = self._set_up_checkpointer(mode=None)
    #     self._init_periodic_checkpointer()

    #     if self.load_best_model:
    #         last_info = self.logger.path("best_model_discrete")
    #         info = self.best_model_checkpointer._load_file(f=last_info)
    #         self.logger.log(
    #             f"=> loading checkpoint of the best-model '{last_info}' start"
    #         )
    #     elif self.start_epoch != 0:
    #         last_info = self.logger.path("checkpoints")
    #         last_info ="{}/{}_{:07d}.pth".format(last_info, "model", self.start_epoch)
    #         info = self.checkpointer._load_file(f=last_info)
    #         self.logger.log(
    #             f"resume from discrete network trained from {self.start_epoch} epochs"
    #         )
    #     elif self.load_saved_model:
    #         last_info = self.logger.path("last_checkpoint")
    #         info = self.checkpointer._load_file(f=last_info)
    #         self.logger.log(f"=> loading checkpoint of the last-info {last_info}")
    #     else:
    #         self.logger.log("=> did not find the any file")
    #         return

    #     # if self.use_supernet_checkpoint:
    #     #     self.logger.use_supernet_checkpoint = False
    #     #     self._init_empty_model_state_info()
    #     # else:

    #     self.logger.set_up_new_run()
    #     self.best_model_checkpointer.save_dir = self.logger.path(mode=None)
    #     self.checkpointer.save_dir = self.logger.path(mode="checkpoints")
    #     self._set_checkpointer_info(info)
