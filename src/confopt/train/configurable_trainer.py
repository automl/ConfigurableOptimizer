from __future__ import annotations

from collections import namedtuple
import time

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import torch
from torch import nn
from typing_extensions import TypeAlias

from confopt.dataset import AbstractData
from confopt.searchspace import SearchSpace
from confopt.utils import AverageMeter, Logger, calc_accuracy

from .searchprofile import Profile

TrainingMetrics = namedtuple("TrainingMetrics", ["loss", "acc_top1", "acc_top5"])

DataLoaderType: TypeAlias = torch.utils.data.DataLoader
OptimizerType: TypeAlias = torch.optim.Optimizer
LRSchedulerType: TypeAlias = torch.optim.lr_scheduler.LRScheduler
CriterionType: TypeAlias = torch.nn.modules.loss._Loss

DEBUG_STEPS = 5


class ConfigurableTrainer:
    def __init__(
        self,
        model: SearchSpace,
        data: AbstractData,
        model_optimizer: OptimizerType,
        arch_optimizer: OptimizerType,
        scheduler: LRSchedulerType,
        criterion: CriterionType,
        logger: Logger,
        batch_size: int,
        use_data_parallel: bool = False,
        print_freq: int = 20,
        drop_path_prob: float = 0.1,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        start_epoch: int = 0,
        checkpointing_freq: int = 1,
        epochs: int = 100,
        debug_mode: bool = False,
    ) -> None:
        self.model = model
        self.model_optimizer = model_optimizer
        self.arch_optimizer = arch_optimizer
        self.scheduler = scheduler
        self.data = data
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.logger = logger
        self.criterion = criterion
        self.use_data_parallel = use_data_parallel
        self.print_freq = print_freq
        self.batch_size = batch_size
        self.drop_path_prob = drop_path_prob
        self.load_saved_model = load_saved_model
        self.load_best_model = load_best_model
        self.start_epoch = start_epoch
        self.checkpointing_freq = checkpointing_freq
        self.epochs = epochs
        self.debug_mode = debug_mode

    def train(self, profile: Profile, epochs: int, is_wandb_log: bool = True) -> None:
        self.epochs = epochs
        profile.adapt_search_space(self.model)

        if self.load_saved_model or self.load_best_model or self.start_epoch != 0:
            self._load_model_state_if_exists()
        else:
            self._init_empty_model_state_info()

        if self.use_data_parallel:
            network, criterion = self._load_onto_data_parallel(
                self.model, self.criterion
            )
        else:
            network: nn.Module = self.model  # type: ignore
            criterion = self.criterion

        start_time = time.time()
        search_time, epoch_time = AverageMeter(), AverageMeter()

        train_loader, val_loader, _ = self.data.get_dataloaders(
            batch_size=self.batch_size,
            n_workers=0,
        )

        for epoch in range(self.start_epoch, epochs):
            epoch_str = f"{epoch:03d}-{epochs:03d}"

            self._component_new_step_or_epoch(network, calling_frequency="epoch")
            self.update_sample_function(profile, network, calling_frequency="epoch")

            base_metrics, arch_metrics = self.train_func(
                profile,
                train_loader,
                val_loader,
                network,
                criterion,
                self.scheduler,
                self.model_optimizer,
                self.arch_optimizer,
                epoch_str,
                self.print_freq,
                self.logger,
            )

            # Logging
            search_time.update(time.time() - start_time)
            self.logger.log_metrics(
                "Search: Model metrics ",
                base_metrics,
                epoch_str,
                search_time.sum,
            )

            self.logger.log_metrics(
                "Search: Architecture metrics ", arch_metrics, epoch_str
            )

            valid_metrics = self.valid_func(val_loader, self.model, self.criterion)
            self.logger.log_metrics("Evaluation: ", valid_metrics, epoch_str)

            if is_wandb_log:
                self.logger.wandb_log_metrics(
                    "search/model", base_metrics, epoch, search_time.sum
                )
                self.logger.wandb_log_metrics("search/arch", arch_metrics, epoch)
                self.logger.wandb_log_metrics("eval", valid_metrics, epoch)

            (
                self.valid_losses[epoch],
                self.valid_accs_top1[epoch],
                self.valid_accs_top5[epoch],
            ) = valid_metrics

            (
                self.search_losses[epoch],
                self.search_accs_top1[epoch],
                self.search_accs_top5[epoch],
            ) = base_metrics

            checkpointables = self._get_checkpointables(epoch=epoch)
            self.periodic_checkpointer.step(
                iteration=epoch, checkpointables=checkpointables
            )
            if valid_metrics.acc_top1 > self.valid_accs_top1["best"]:
                self.valid_accs_top1["best"] = valid_metrics.acc_top1
                self.logger.log(
                    f"<<<--->>> The {epoch_str}-th epoch : found the highest "
                    + f"validation accuracy : {valid_metrics.acc_top1:.2f}%."
                )

                self.best_model_checkpointer.save(
                    name="best_model", checkpointables=checkpointables
                )

            with torch.no_grad():
                for i, alpha in enumerate(self.model.arch_parameters):
                    self.logger.log(f"alpha {i} is {alpha}")

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

            if self.scheduler is not None:
                self.scheduler.step()

    def train_func(
        self,
        profile: Profile,
        train_loader: DataLoaderType,
        valid_loader: DataLoaderType,
        network: SearchSpace | torch.nn.DataParallel,
        criterion: CriterionType,
        w_scheduler: LRSchedulerType,  # noqa: ARG002  TODO:Fix
        w_optimizer: OptimizerType,
        arch_optimizer: OptimizerType,
        epoch_str: str,  # noqa: ARG002  TODO:Fix
        print_freq: int,
        logger: Logger,  # noqa: ARG002  TODO:Fix
    ) -> tuple[TrainingMetrics, TrainingMetrics]:
        data_time, batch_time = AverageMeter(), AverageMeter()
        base_losses, base_top1, base_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        arch_losses, arch_top1, arch_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        network.train()
        end = time.time()

        for step, (base_inputs, base_targets) in enumerate(train_loader):
            # FIXME: What was the point of this? and is it safe to remove?
            # scheduler.update(None, 1.0 * step / len(xloader))
            self._component_new_step_or_epoch(network, calling_frequency="step")
            if step == 1:
                self.update_sample_function(profile, network, calling_frequency="step")

            arch_inputs, arch_targets = next(iter(valid_loader))

            base_inputs, arch_inputs = (
                base_inputs.to(self.device),
                arch_inputs.to(self.device),
            )
            base_targets = base_targets.to(self.device, non_blocking=True)
            arch_targets = arch_targets.to(self.device, non_blocking=True)

            # measure data loading time
            data_time.update(time.time() - end)

            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            arch_loss.backward()
            arch_optimizer.step()

            profile.perturb_parameter(network)

            self._update_meters(
                inputs=arch_inputs,
                logits=logits,
                targets=arch_targets,
                loss=arch_loss,
                loss_meter=arch_losses,
                top1_meter=arch_top1,
                top5_meter=arch_top5,
            )

            # update the model weights
            w_optimizer.zero_grad()
            arch_optimizer.zero_grad()

            _, logits = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            base_loss.backward()
            # TODO: Does this vary with the one-shot optimizers?
            if isinstance(network, torch.nn.DataParallel):
                torch.nn.utils.clip_grad_norm_(
                    network.module.model_weight_parameters(), 5
                )
            else:
                torch.nn.utils.clip_grad_norm_(network.model_weight_parameters(), 5)

            w_optimizer.step()

            w_optimizer.zero_grad()
            arch_optimizer.zero_grad()

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

            if step % print_freq == 0 or step + 1 == len(train_loader):
                # Tstr = f"Time {batch_time.val:.2f} ({batch_time.avg:.2f})" \
                #     +   f"Data {data_time.val:.2f} ({data_time.avg:.2f})"

                # Wstr = f"Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1" \
                #     +   f"{top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f}" \
                #     +   f"({top5.avg:.2f})]"

                # Astr = f"Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1" \
                #     +   f"{top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f}" \
                #     +   f"({top5.avg:.2f})]"

                # logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr)
                ...

            if self.debug_mode and step > DEBUG_STEPS:
                break

        base_metrics = TrainingMetrics(base_losses.avg, base_top1.avg, base_top5.avg)
        arch_metrics = TrainingMetrics(arch_losses.avg, arch_top1.avg, arch_top5.avg)

        return base_metrics, arch_metrics

    def valid_func(
        self,
        valid_loader: DataLoaderType,
        network: SearchSpace | torch.nn.DataParallel,
        criterion: CriterionType,
    ) -> TrainingMetrics:
        arch_losses, arch_top1, arch_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        network.eval()

        with torch.no_grad():
            for _step, (arch_inputs, arch_targets) in enumerate(valid_loader):
                # if torch.cuda.is_available():
                #     arch_targets = arch_targets.cuda(non_blocking=True)
                #     arch_inputs = arch_inputs.cuda(non_blocking=True)

                # prediction
                arch_inputs = arch_inputs.to(self.device)
                arch_targets = arch_targets.to(self.device, non_blocking=True)

                _, logits = network(arch_inputs)
                arch_loss = criterion(logits, arch_targets)

                # record
                arch_prec1, arch_prec5 = calc_accuracy(
                    logits.data, arch_targets.data, topk=(1, 5)
                )

                arch_losses.update(arch_loss.item(), arch_inputs.size(0))
                arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
                arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

                if self.debug_mode and _step > DEBUG_STEPS:
                    break

        return TrainingMetrics(arch_losses.avg, arch_top1.avg, arch_top5.avg)

    def _load_onto_data_parallel(
        self, network: nn.Module, criterion: CriterionType
    ) -> tuple[nn.Module, CriterionType]:
        if torch.cuda.is_available():
            network, criterion = (
                torch.nn.DataParallel(self.model).cuda(),
                criterion.cuda(),
            )

        return network, criterion

    def _init_empty_model_state_info(self) -> None:
        self.start_epoch = 0
        self.search_losses: dict[int, float] = {}
        self.search_accs_top1: dict[int, float] = {}
        self.search_accs_top5: dict[int, float] = {}
        self.valid_losses: dict[int, float] = {}
        self.valid_accs_top1: dict[int | str, float | int] = {"best": -1}
        self.valid_accs_top5: dict[int, float] = {}

        self._init_periodic_checkpointer()
        self.best_model_checkpointer = self._set_up_checkpointer(mode=None)
        self.logger.set_up_run()

    def _set_up_checkpointer(self, mode: str | None) -> Checkpointer:
        checkpoint_dir = self.logger.path(mode=mode)  # todo: check this
        # checkpointables = self._get_checkpointables(self.start_epoch)
        # todo: return scheduler and optimizers that do have state_dict()
        checkpointer = Checkpointer(
            model=self.model,
            save_dir=checkpoint_dir,
            save_to_disk=True,
            # checkpointables=checkpointables,
        )
        checkpointer.add_checkpointable("w_scheduler", self.scheduler)
        checkpointer.add_checkpointable("w_optimizer", self.model_optimizer)
        checkpointer.add_checkpointable("arch_optimizer", self.arch_optimizer)
        return checkpointer

    def _init_periodic_checkpointer(self) -> None:
        self.checkpointer = self._set_up_checkpointer(mode="checkpoints")
        self.periodic_checkpointer = PeriodicCheckpointer(
            checkpointer=self.checkpointer,
            period=self.checkpointing_freq,
            max_iter=self.epochs,
        )

    def _get_checkpointables(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "search_losses": self.search_losses,
            "search_accs_top1": self.search_accs_top1,
            "search_accs_top5": self.search_accs_top5,
            "valid_losses": self.valid_losses,
            "valid_accs_top1": self.valid_accs_top1,
            "valid_accs_top5": self.valid_accs_top5,
        }

    def _set_checkpointer_info(self, last_checkpoint: dict) -> None:
        self.model.load_state_dict(last_checkpoint["model"])
        self.scheduler.load_state_dict(last_checkpoint["w_scheduler"])
        self.model_optimizer.load_state_dict(last_checkpoint["w_optimizer"])
        self.arch_optimizer.load_state_dict(last_checkpoint["arch_optimizer"])

        last_checkpoint = last_checkpoint["checkpointables"]
        self.start_epoch = last_checkpoint["epoch"]
        self.search_losses = last_checkpoint["search_losses"]
        self.search_accs_top1 = last_checkpoint["search_accs_top1"]
        self.search_accs_top5 = last_checkpoint["search_accs_top5"]
        self.valid_losses = last_checkpoint["valid_losses"]
        self.valid_accs_top1 = last_checkpoint["valid_accs_top1"]
        self.valid_accs_top5 = last_checkpoint["valid_accs_top5"]
        self.logger.log(f"start with {self.start_epoch}-th epoch.")

    def _load_model_state_if_exists(self) -> None:
        self.best_model_checkpointer = self._set_up_checkpointer(mode=None)
        self._init_periodic_checkpointer()

        if self.load_best_model:
            last_info = self.logger.path("best_model")
            self.logger.log(
                f"=> loading checkpoint of the best-model '{last_info}' start"
            )
            info = self.best_model_checkpointer._load_file(f=last_info)
        elif self.start_epoch != 0:
            last_info = self.logger.path("checkpoints")
            last_info = "{}/{}_{:07d}.pth".format(last_info, "model", self.start_epoch)
            info = self.checkpointer._load_file(f=last_info)
        elif self.load_saved_model:
            last_info = self.logger.path("last_checkpoint")
            info = self.checkpointer._load_file(f=last_info)
            self.logger.log(f"=> loading checkpoint of the last-info {last_info}")
        else:
            self.logger.log("=> did not find the any file")
            return

        self.logger.set_up_new_run()
        self.best_model_checkpointer.save_dir = self.logger.path(mode=None)
        self.checkpointer.save_dir = self.logger.path(mode="checkpoints")
        self._set_checkpointer_info(info)

        self.logger.log(
            "=> loading checkpoint " + f"start with {self.start_epoch}-th epoch."
        )

        # Then put checkpoint data into the self and model

    def _update_meters(
        self,
        inputs: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        loss_meter: AverageMeter,
        top1_meter: AverageMeter,
        top5_meter: AverageMeter,
    ) -> None:
        base_prec1, base_prec5 = calc_accuracy(logits.data, targets.data, topk=(1, 5))
        loss_meter.update(loss.item(), inputs.size(0))
        top1_meter.update(base_prec1.item(), inputs.size(0))
        top5_meter.update(base_prec5.item(), inputs.size(0))

    def _component_new_step_or_epoch(
        self, model: SearchSpace | torch.nn.DataParallel, calling_frequency: str
    ) -> None:
        assert calling_frequency in [
            "epoch",
            "step",
        ], "Called Frequency should be either epoch or step"
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        assert (
            len(model.components) > 0
        ), "There are no oneshot components inside the search space"
        if calling_frequency == "epoch":
            model.new_epoch()
        elif calling_frequency == "step":
            model.new_step()

    def update_sample_function(
        self,
        profile: Profile,
        model: SearchSpace | torch.nn.DataParallel,
        calling_frequency: str,
    ) -> None:
        assert calling_frequency in [
            "epoch",
            "step",
        ], "Called Frequency should be either epoch or step"
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        assert (
            len(model.components) > 0
        ), "There are no oneshot components inside the search space"
        if calling_frequency == "epoch":
            profile.update_sample_function_from_sampler(model)
        elif (
            calling_frequency == "step" and profile.sampler.sample_frequency == "epoch"
        ):
            profile.reset_sample_function(model)
