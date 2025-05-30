##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import IO, Any, NamedTuple

import torch
import wandb

from .time import get_runtime


def prepare_logger(
    save_dir: str,
    seed: int,
    exp_name: str,
    xargs: argparse.Namespace | None = None,
) -> Logger:
    logger = Logger(save_dir, seed=str(seed), exp_name=exp_name)
    logger.log(f"Main Function with logger : {logger}")
    logger.log("Arguments : -------------------------------")

    if xargs is not None:
        for name, value in xargs._get_kwargs():
            logger.log(f"{name:16} : {value}")

    logger.log("Python  Version  : {:}".format(sys.version.replace("\n", " ")))
    logger.log(f"PyTorch Version  : {torch.__version__}")
    logger.log(f"cuDNN   Version  : {torch.backends.cudnn.version()}")
    logger.log(f"CUDA available   : {torch.cuda.is_available()}")
    logger.log(f"CUDA GPU numbers : {torch.cuda.device_count()}")
    logger.log(
        "CUDA_VISIBLE_DEVICES : {:}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else "None"
        )
    )
    return logger


class Logger:
    def __init__(
        self,
        log_dir: str,
        exp_name: str = "",
        search_space: str = "",
        dataset: str = "cifar10",
        seed: str | int = 2,
        runtime: str | None = None,
        use_supernet_checkpoint: bool = False,
        last_run: bool = False,
        arch_selection: bool = False,
        custom_log_path: str | None = None,
    ) -> None:
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.search_space = search_space
        self.dataset = dataset
        self.seed = str(seed)
        self.use_supernet_checkpoint = use_supernet_checkpoint
        self.arch_selection = arch_selection
        self.supernet_str = "supernet"
        if self.arch_selection:
            self.supernet_str = "arch_selection"
        self.last_run = last_run

        # Optionally load from the path here, this still requires runtime
        self.custom_log_path = custom_log_path
        if self.custom_log_path is not None:
            assert os.path.exists(
                self.custom_log_path
            ), f"{self.custom_log_path} is not a valid path"

        assert sum([last_run, runtime is not None]) <= 1
        """Create a summary writer logging to log_dir."""
        if last_run:
            assert os.path.exists(self.expr_log_path() / "last_run")
            last_runtime = self.load_last_run()
            self.set_up_run(last_runtime)
        elif runtime and runtime != "":
            assert os.path.exists(self.expr_log_path() / runtime)
            self.set_up_run(runtime)
        else:
            self.set_up_new_run()

        self._wandb_logs = {}  # type: ignore

    def set_up_new_run(self) -> None:
        runtime = get_runtime()
        # runtime = time.strftime("%Y-%d-%h-%H:%M:%S", time.gmtime(time.time()))
        if hasattr(self, "logger_file"):
            self.close()
        self.set_up_run(runtime=runtime)

    def set_up_run(self, runtime: str) -> None:
        self.runtime = runtime
        (self.expr_log_path() / self.runtime).mkdir(parents=True, exist_ok=True)
        (self.expr_log_path() / self.runtime / "checkpoints").mkdir(
            parents=True, exist_ok=True
        )
        if self.use_supernet_checkpoint:
            (self.expr_log_path() / self.runtime / "genotypes").mkdir(
                parents=True, exist_ok=True
            )
        self.tensorboard_dir = (self.expr_log_path() / self.runtime) / (
            "tensorboard-{:}".format(time.strftime("%d-%h", time.gmtime(time.time())))
        )
        self.save_last_run()
        self.logger_path = self.expr_log_path() / self.runtime / "log"
        self.logger_file = open(self.logger_path, "w")  # noqa: SIM115
        self.writer = None

    def expr_log_path(self) -> Path:
        if self.custom_log_path is None:
            path_componenets = [
                self.log_dir,
                self.exp_name,
                self.search_space,
                self.dataset,
                self.seed,
                self.supernet_str if self.use_supernet_checkpoint else "discrete",
            ]
            expr_log_path_str = "/".join(path_componenets)
        else:
            expr_log_path_str = self.custom_log_path
        return Path(expr_log_path_str)

    def load_last_run(self) -> str:
        file_path = self.expr_log_path() / "last_run"

        with open(file_path) as f:
            runtime = f.read().strip()
        return runtime

    def save_last_run(self) -> None:
        file_path = self.expr_log_path() / "last_run"

        with open(file_path, "w") as f:
            f.write(self.runtime)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dir={self.log_dir}, writer={self.writer})"

    def path(self, mode: str | None) -> str:
        valids = (
            "best_model",  # checkpoint containing the best model
            "checkpoints",  # checkpoint of all the checkpoints (periodic)
            "log",  # path to the logger file
            "last_checkpoint",  # return the last checkpoint in the checkpoints folder
            "genotypes",  # path to the folder containing the genotype of the model
            "best_genotype",  # path to the file containing the best genotype
            "last_genotype",
            None,
        )
        path = None
        if mode not in valids:
            raise TypeError(f"Unknow mode = {mode}, valid modes = {valids}")
        if mode == "best_model":
            path = str((self.expr_log_path() / self.runtime) / (mode + ".pth"))
        if mode == "last_checkpoint":
            last_checkpoint_path = (
                (self.expr_log_path() / self.runtime)
                / "checkpoints"
                / "last_checkpoint"
            )
            with open(last_checkpoint_path) as f:
                path = str(
                    (self.expr_log_path() / self.runtime)
                    / "checkpoints"
                    / f.read().strip()
                )
        if mode == "genotypes":
            if self.use_supernet_checkpoint:
                path = str((self.expr_log_path() / self.runtime) / "genotypes")
            else:
                path = str((self.expr_log_path() / self.runtime) / "genotype.txt")
        if mode == "best_genotype":
            path = str((self.expr_log_path() / self.runtime) / "best_genotype.txt")
        if mode is None:
            path = str(self.expr_log_path() / self.runtime)
        if path is None:
            return str((self.expr_log_path() / self.runtime) / mode)  # type: ignore
        return path

    def extract_log(self) -> IO[Any]:
        return self.logger_file

    def close(self) -> None:
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string: str, save: bool = True, stdout: bool = False) -> None:
        if stdout:
            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write(f"{string}\n")
            self.logger_file.flush()

    def load_genotype(
        self,
        model_to_load: str | int | None = None,
        use_supernet_checkpoint: bool = False,
    ) -> str:
        assert not (
            not use_supernet_checkpoint and model_to_load is None
        ), "model_to_load must be provided when using discretized network."
        if not use_supernet_checkpoint:
            file_path = self.path(mode="genotypes")
        elif model_to_load == "best":
            file_path = self.path(mode="best_genotype")
        elif type(model_to_load) == int:
            file_path = self.path("genotypes")
            file_path = "{}/{}_{:04d}.txt".format(file_path, "genotype", model_to_load)
        elif model_to_load == "last":
            file_path = self.path("genotypes")
            last_file_path = "{}/{}".format(file_path, "last_genotype.txt")
            with open(last_file_path) as f:
                file_path = f"{file_path}/{f.read().strip()}"
        else:
            raise ValueError("Should specify model_to_load value")

        with open(file_path) as f:
            genotype = f.read().strip()
        return genotype

    def save_genotype(
        self,
        genotype: str,
        epoch: int = 0,
        checkpointing_freq: int = 1,
        is_best_model: bool = False,
    ) -> None:
        if epoch % checkpointing_freq != 0:
            return

        genotype_filename = f"genotype_{epoch:04d}.txt"

        # Log the last genotype if using supernet checkpoint
        if self.use_supernet_checkpoint:
            genotypes_dir = Path(self.path(mode="genotypes"))
            genotype_filename = f"genotype_{epoch:04d}.txt"
            genotype_filepath = genotypes_dir / genotype_filename

            last_genotype_filepath = genotypes_dir / "last_genotype.txt"

            with open(last_genotype_filepath, "w") as f:
                f.write(genotype_filename)
        else:
            # When training the discrete model, the genotype is fixed
            genotype_filepath = Path(self.path(mode="genotypes"))

        # Log the (current/fixed) genotype
        with open(genotype_filepath, "w") as f:
            f.write(genotype)

        if is_best_model:
            best_genotype_filepath = Path(self.path(mode="best_genotype"))
            with open(best_genotype_filepath, "w") as f:
                f.write(genotype_filename)

    def log_metrics(
        self,
        title: str,
        metrics: NamedTuple,
        epoch_str: str = "",
        totaltime: float | None = None,
    ) -> None:
        msg = "[{:}] {} : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
            epoch_str,
            title,
            metrics.loss,  # type: ignore
            metrics.acc_top1,  # type: ignore
            metrics.acc_top5,  # type: ignore
        )

        if totaltime is not None:
            msg += f", time-cost={totaltime:.1f} s"

        self.log(msg)

    def add_wandb_log_metrics(
        self,
        title: str,
        metrics: NamedTuple,
        epoch: int | None = None,
        totaltime: float | None = None,
    ) -> None:
        log_metrics = {
            f"{title}/epochs": epoch,
            f"{title}/loss": metrics.loss,  # type: ignore
            f"{title}/acc_top1": metrics.acc_top1,  # type: ignore
            f"{title}/acc_top5": metrics.acc_top5,  # type: ignore
        }
        if totaltime is not None:
            log_metrics.update({f"{title}/time": totaltime})

        self._wandb_logs.update(log_metrics)

    def push_wandb_logs(self) -> None:
        assert self._wandb_logs is not None, "Cannot log empty metric"
        wandb.log(self._wandb_logs)  # type: ignore

    def update_wandb_logs(self, logs: dict) -> None:
        self._wandb_logs.update(logs)

    def get_wandb_logs(self) -> dict:
        return self._wandb_logs

    def reset_wandb_logs(self) -> None:
        self._wandb_logs = {}
