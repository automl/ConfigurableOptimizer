from __future__ import annotations

from collections import namedtuple
import time
from typing import Any

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.modules.loss import _Loss as Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from confopt.dataset import AbstractData
from confopt.searchspace import SearchSpace
from confopt.utils import (
    AverageMeter,
    Logger,
    calc_accuracy,
    clear_grad_cosine,
    get_device,
)

from .search_space_handler import SearchSpaceHandler

TrainingMetrics = namedtuple("TrainingMetrics", ["loss", "acc_top1", "acc_top5"])


DEBUG_STEPS = 5


class ConfigurableTrainer:
    def __init__(
        self,
        model: SearchSpace,
        data: AbstractData,
        model_optimizer: Optimizer,
        arch_optimizer: Optimizer | None,
        scheduler: LRScheduler,
        criterion: Loss,
        logger: Logger,
        batch_size: int,
        use_data_parallel: bool = False,
        print_freq: int = 20,
        drop_path_prob: float = 0.1,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        start_epoch: int = 0,
        checkpointing_freq: int = 2,
        epochs: int = 100,
        debug_mode: bool = False,
        query_dataset: str = "cifar10",
        benchmark_api: Any | None = None,
    ) -> None:
        self.model = model
        self.model_optimizer = model_optimizer
        self.arch_optimizer = arch_optimizer
        self.scheduler = scheduler
        self.data = data
        self.device = get_device()
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
        self.query_dataset = query_dataset
        self.benchmark_api = benchmark_api

    def train(  # noqa: C901, PLR0915, PLR0912
        self,
        search_space_handler: SearchSpaceHandler,
        is_wandb_log: bool = True,
        lora_warm_epochs: int = 0,
        oles: bool = False,
        calc_gm_score: bool = False,
    ) -> None:
        search_space_handler.adapt_search_space(self.model)

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
        is_warm_epoch = False
        if lora_warm_epochs > 0:
            assert (
                search_space_handler.lora_configs is not None
            ), "The SearchSpaceHandler's LoRA configs are missing"
            assert (
                search_space_handler.lora_configs.get("r", 0) > 0
            ), "Value of r should be greater than 0"
            is_warm_epoch = True
        warm_epochs = lora_warm_epochs
        if search_space_handler.partial_connector:
            warm_epochs = max(
                search_space_handler.partial_connector.num_warm_epoch, lora_warm_epochs
            )
            is_warm_epoch = True

        layer_alignment_scores = (AverageMeter(), AverageMeter())

        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            epoch_str = f"{epoch:03d}-{self.epochs:03d}"
            if epoch == warm_epochs + 1:
                if lora_warm_epochs > 0:
                    self._initialize_lora_modules(
                        lora_warm_epochs, search_space_handler, network, calc_gm_score
                    )
                is_warm_epoch = False

            self._component_new_step_or_epoch(network, calling_frequency="epoch")
            self.update_sample_function(
                search_space_handler, network, calling_frequency="epoch"
            )

            # Reset WandB Log dictionary
            self.logger.reset_wandb_logs()

            base_metrics, arch_metrics = self._train_epoch(
                search_space_handler,
                train_loader,
                val_loader,
                network,
                criterion,
                self.model_optimizer,
                self.arch_optimizer,
                self.print_freq,
                is_warm_epoch=is_warm_epoch,
                oles=oles,
                calc_gm_score=calc_gm_score,
                layer_alignment_scores=layer_alignment_scores,
            )

            # Logging
            # Log Search Metrics
            search_time.update(time.time() - start_time)
            self.logger.log_metrics(
                "Search: Model metrics ",
                base_metrics,
                epoch_str,
                search_time.sum,
            )

            if not is_warm_epoch:
                self.logger.log_metrics(
                    "Search: Architecture metrics ", arch_metrics, epoch_str
                )

            # Log Valid Metrics
            valid_metrics = self.evaluate(val_loader, self.model, self.criterion)
            self.logger.log_metrics("Evaluation: ", valid_metrics, epoch_str)

            self.logger.add_wandb_log_metrics(
                "search/model", base_metrics, epoch, search_time.sum
            )
            self.logger.add_wandb_log_metrics("search/arch", arch_metrics, epoch)
            self.logger.add_wandb_log_metrics("eval", valid_metrics, epoch)

            # Log architectural parameter values
            arch_values_dict = self.get_arch_values_as_dict(network)
            self.logger.update_wandb_logs(arch_values_dict)

            # Log GM scores
            if calc_gm_score:
                if self.use_data_parallel:
                    gm_score = network.module.calc_avg_gm_score()
                else:
                    gm_score = network.calc_avg_gm_score()
                gm_metrics = {
                    "gm_scores/mean_gm": gm_score,
                    "gm_scores/epochs": epoch,
                }
                gm_metrics.update(self.get_all_running_mean_scores(network))

                # Add for all modules
                self.logger.update_wandb_logs(gm_metrics)

            # Count skip connections in this epoch
            normal_cell_n_skip, reduce_cell_n_skip = (
                network.module.get_num_skip_ops()
                if self.use_data_parallel
                else network.get_num_skip_ops()
            )

            n_skip_connections = {
                "skip_connections/normal": normal_cell_n_skip,
                "skip_connections/reduce": reduce_cell_n_skip,
            }
            self.logger.update_wandb_logs(n_skip_connections)

            # Log Layer Alignment scores
            self.logger.log(
                f"[{epoch_str}] Layer Alignment score: "
                + f" normal: {layer_alignment_scores[0].avg:.4f},"
                + f" reduce: {layer_alignment_scores[1].avg:.4f}"
            )
            layer_alignment_metric = {
                "layer_alignment/normal": layer_alignment_scores[0].avg,
                "layer_alignment/reduce": layer_alignment_scores[1].avg,
            }
            self.logger.update_wandb_logs(layer_alignment_metric)

            # Create checkpoints
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

            if self.scheduler is not None:
                self.scheduler.step()

            checkpointables = self._get_checkpointables(epoch=epoch)
            self.periodic_checkpointer.step(
                iteration=epoch, checkpointables=checkpointables
            )

            # Save Genotype and log best model
            # genotype = str(self.model.get_genotype())
            genotype = self.model.get_genotype().tostr()  # type: ignore
            self.logger.save_genotype(genotype, epoch, self.checkpointing_freq)
            if valid_metrics.acc_top1 > self.valid_accs_top1["best"]:
                self.valid_accs_top1["best"] = valid_metrics.acc_top1
                self.logger.log(
                    f"<<<--->>> The {epoch_str}-th epoch : found the highest "
                    + f"validation accuracy : {valid_metrics.acc_top1:.2f}%."
                )

                self.best_model_checkpointer.save(
                    name="best_model", checkpointables=checkpointables
                )
                self.logger.save_genotype(
                    genotype, epoch, self.checkpointing_freq, save_best_model=True
                )

            # Log Benchmark Results
            self.log_benchmark_result(network)

            # Push WandB Logs
            if is_wandb_log:
                self.logger.push_wandb_logs()

            # Log alpha values
            if not is_warm_epoch:
                with torch.no_grad():
                    for i, alpha in enumerate(self.model.arch_parameters):
                        self.logger.log(f"alpha {i} is {alpha}")

            # Reset GM Scores
            if calc_gm_score:
                if self.use_data_parallel:
                    network.module.reset_gm_scores()
                else:
                    network.reset_gm_scores()

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

    def _train_epoch(  # noqa: PLR0912, PLR0915, C901
        self,
        search_space_handler: SearchSpaceHandler,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        network: SearchSpace | DataParallel,
        criterion: Loss,
        w_optimizer: Optimizer,
        arch_optimizer: Optimizer,
        print_freq: int,
        is_warm_epoch: bool = False,
        oles: bool = False,
        calc_gm_score: bool = False,
        layer_alignment_scores: tuple[AverageMeter, AverageMeter] | None = None,
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
        layer_alignment_scores[0].reset()  # type: ignore
        layer_alignment_scores[1].reset()  # type: ignore

        for step, (base_inputs, base_targets) in enumerate(train_loader):
            # FIXME: What was the point of this? and is it safe to remove?
            # self.scheduler.update(None, 1.0 * step / len(xloader))
            self._component_new_step_or_epoch(network, calling_frequency="step")
            if step == 1:
                self.update_sample_function(
                    search_space_handler, network, calling_frequency="step"
                )

            arch_inputs, arch_targets = next(iter(valid_loader))

            base_inputs, arch_inputs = (
                base_inputs.to(self.device),
                arch_inputs.to(self.device),
            )
            base_targets = base_targets.to(self.device, non_blocking=True)
            arch_targets = arch_targets.to(self.device, non_blocking=True)

            # measure data loading time
            data_time.update(time.time() - end)

            if not is_warm_epoch:
                _, logits = network(arch_inputs)
                arch_loss = criterion(logits, arch_targets)
                arch_loss.backward()
                arch_optimizer.step()

                if self.use_data_parallel:
                    search_space_handler.perturb_parameter(network.module)
                else:
                    search_space_handler.perturb_parameter(network)

                self._update_meters(
                    inputs=arch_inputs,
                    logits=logits,
                    targets=arch_targets,
                    loss=arch_loss,
                    loss_meter=arch_losses,
                    top1_meter=arch_top1,
                    top5_meter=arch_top5,
                )

            # calculate gm_score
            if calc_gm_score:
                if self.use_data_parallel:
                    network.module.check_grads_cosine(oles)  # type: ignore
                else:
                    network.check_grads_cosine(oles)  # type: ignore

            # update the model weights
            w_optimizer.zero_grad()
            arch_optimizer.zero_grad()

            _, logits = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            base_loss.backward()

            network_module = network.module if self.use_data_parallel else network
            if hasattr(network_module, "get_mean_layer_alignment_score"):
                (
                    score_normal,
                    score_reduce,
                ) = network_module.get_mean_layer_alignment_score()
                layer_alignment_scores[0].update(val=score_normal)  # type: ignore
                layer_alignment_scores[1].update(val=score_reduce)  # type: ignore

            if self.use_data_parallel:
                torch.nn.utils.clip_grad_norm_(
                    network.module.model_weight_parameters(), 5
                )
            else:
                torch.nn.utils.clip_grad_norm_(network.model_weight_parameters(), 5)

            w_optimizer.step()

            # save grads of operations
            if calc_gm_score:
                if self.use_data_parallel:
                    network.module.preserve_grads()  # type: ignore
                else:
                    network.preserve_grads()  # type: ignore

            w_optimizer.zero_grad()
            if not is_warm_epoch:
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

    def evaluate(
        self,
        valid_loader: DataLoader,
        network: SearchSpace | DataParallel | DistributedDataParallel,
        criterion: Loss,
    ) -> TrainingMetrics:
        arch_losses, arch_top1, arch_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        network.eval()

        with torch.no_grad():
            for _step, (arch_inputs, arch_targets) in enumerate(valid_loader):
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

    def _load_onto_distributed_data_parallel(
        self, network: nn.Module, criterion: Loss
    ) -> tuple[nn.Module, Loss]:
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            network = DistributedDataParallel(self.model.cuda())
            criterion = criterion.cuda()

        return network, criterion

    def _load_onto_data_parallel(
        self, network: nn.Module, criterion: Loss
    ) -> tuple[nn.Module, Loss]:
        if torch.cuda.is_available():
            network, criterion = (
                DataParallel(self.model).cuda(),
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
        # TODO: this is needed?
        # self.logger.set_up_run()

    def _set_up_checkpointer(self, mode: str | None) -> Checkpointer:
        checkpoint_dir = self.logger.path(mode=mode)  # todo: check this
        # checkpointables = self._get_checkpointables(self.start_epoch)
        # todo: return scheduler and optimizers that do have state_dict()
        # checkpointables = {
        #     "w_scheduler": self.scheduler,
        #     "w_optimizer": self.model_optimizer,
        #     "arch_optimizer": self.arch_optimizer,
        # }
        checkpointer = Checkpointer(
            model=self.model,
            save_dir=checkpoint_dir,
            save_to_disk=True,
            # **checkpointables,
        )
        checkpointer.add_checkpointable("w_scheduler", self.scheduler)
        checkpointer.add_checkpointable("w_optimizer", self.model_optimizer)
        if self.arch_optimizer is not None:
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
        if self.arch_optimizer:
            self.arch_optimizer.load_state_dict(last_checkpoint["arch_optimizer"])
        self.model_optimizer.load_state_dict(last_checkpoint["w_optimizer"])
        self.scheduler.load_state_dict(last_checkpoint["w_scheduler"])
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
        self, model: SearchSpace | DataParallel, calling_frequency: str
    ) -> None:
        assert calling_frequency in [
            "epoch",
            "step",
        ], "Called Frequency should be either epoch or step"
        if self.use_data_parallel:
            model = model.module
        assert (
            len(model.components) > 0
        ), "There are no oneshot components inside the search space"
        if calling_frequency == "epoch":
            model.new_epoch()
        elif calling_frequency == "step":
            model.new_step()

    def log_benchmark_result(
        self,
        network: torch.nn.Module,
    ) -> None:
        if self.benchmark_api is None:
            return
        genotype = (
            network.module.model.genotype()
            if self.use_data_parallel
            else network.model.genotype()
        )
        result_train, rusult_valid, result_test = self.benchmark_api.query(
            genotype, dataset=self.query_dataset
        )
        self.logger.log(
            f"Benchmark Results for {self.query_dataset} -> train: {result_train}, "
            + f"valid: {rusult_valid}, test: {result_test}"
        )

        log_dict = {
            "benchmark/train": result_train,
            "benchmark/valid": rusult_valid,
            "benchmark/test": result_test,
        }
        self.logger.update_wandb_logs(log_dict)

    def _initialize_lora_modules(
        self,
        lora_warm_epochs: int,
        search_space_handler: SearchSpaceHandler,
        network: torch.nn.Module,
        is_gm_score_enabled: bool = False,
    ) -> None:
        self.logger.log(
            f"The searchspace has been warm started with {lora_warm_epochs} epochs"
        )
        search_space_handler.activate_lora(
            network, **search_space_handler.lora_configs
        )  # type: ignore
        self.logger.log(
            "LoRA layers have been initialized for all operations with"
            + " Conv2DLoRA module"
        )
        # clear OLES_OPS from pre_grads, avg and count
        if self.use_data_parallel:
            network.module.apply(clear_grad_cosine)
        else:
            network.apply(clear_grad_cosine)
        # reinitialize optimizer
        optimizer_hyperparameters = self.model_optimizer.defaults
        old_param_lrs = []
        old_initial_lrs = []
        for param_group in self.model_optimizer.param_groups:
            old_param_lrs.append(param_group["lr"])
            old_initial_lrs.append(param_group["initial_lr"])

        self.model_optimizer = type(self.model_optimizer)(
            (
                network.module.model_weight_parameters()  # type: ignore
                if self.use_data_parallel
                else network.model_weight_parameters()  # type: ignore
            ),
            **optimizer_hyperparameters,
        )
        # change the lr for optimizer
        # Update optimizer learning rate manually
        for param_id, param_group in enumerate(self.model_optimizer.param_groups):
            param_group["lr"] = old_param_lrs[param_id]
            param_group["initial_lr"] = old_initial_lrs[param_id]

        # reinitialize scheduler
        scheduler_config = {}
        if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler_config = {
                "T_max": self.scheduler.T_max,
                "eta_min": self.scheduler.eta_min,
            }

        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ):
            scheduler_config = {
                "T_0": self.scheduler.T_0,
                "T_mult": self.scheduler.T_mult,
                "eta_min": self.scheduler.eta_min,
            }

        self.scheduler = type(self.scheduler)(
            self.model_optimizer,
            **scheduler_config,
        )

        if is_gm_score_enabled:
            if self.use_data_parallel:
                network.module.reset_gm_score_attributes()
            else:
                network.reset_gm_score_attributes()

    def get_all_running_mean_scores(self, network: torch.nn.Module) -> dict:
        running_sim_dict = {}
        model = network.module if self.use_data_parallel else network
        for name, module in model.named_modules():
            if hasattr(module, "running_sim"):
                running_sim_dict[f"gm_scores/{name}"] = module.running_sim.avg
        return running_sim_dict

    def update_sample_function(
        self,
        search_space_handler: SearchSpaceHandler,
        model: SearchSpace | DataParallel,
        calling_frequency: str,
    ) -> None:
        assert calling_frequency in [
            "epoch",
            "step",
        ], "Called Frequency should be either epoch or step"
        if self.use_data_parallel:
            model = model.module
        assert (
            len(model.components) > 0
        ), "There are no oneshot components inside the search space"
        if calling_frequency == "epoch":
            search_space_handler.update_sample_function_from_sampler(model)
        elif (
            calling_frequency == "step"
            and search_space_handler.sampler.sample_frequency == "epoch"
        ):
            search_space_handler.reset_sample_function(model)

    def get_arch_values_as_dict(self, model: SearchSpace) -> dict:
        if isinstance(model, DataParallel):
            model = model.module
        arch_values = model.arch_parameters
        arch_values_dict = {}

        for i, alpha in enumerate(arch_values):
            data = {}
            alpha = torch.nn.functional.softmax(alpha, dim=-1).detach().cpu().numpy()
            for edge_idx in range(alpha.shape[0]):
                for op_idx in range(alpha.shape[1]):
                    data[f"edge_{edge_idx}_op_{op_idx}"] = alpha[edge_idx][op_idx]
            arch_values_dict[f"arch_values/alpha_{i}"] = data

        return arch_values_dict
