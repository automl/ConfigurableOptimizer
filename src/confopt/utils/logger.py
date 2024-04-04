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


def prepare_logger(
    save_dir: str,
    seed: int,
    exp_name: str,
    xargs: argparse.Namespace | None = None,
) -> Logger:
    logger = Logger(save_dir, seed, exp_name=exp_name)
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
        seed: str | int,
        # create_model_dir: bool = True,
        exp_name: str = "",
        search_space: str = "",
        run_time: str | None = None,
        last_run: bool = False,
    ) -> None:
        """Create a summary writer logging to log_dir."""
        if last_run:
            run_time = self.load_last_run(log_dir, exp_name, search_space, str(seed))
        elif run_time is None:
            run_time = time.strftime("%Y-%d-%h-%H:%M:%S", time.gmtime(time.time()))
        else:
            print(
                "loading from the provided run time."
                "Please make sure that the run folder is avialable"
            )

        self.log_dir = Path(log_dir) / exp_name / search_space / str(seed) / run_time
        self.seed = int(seed)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        (Path(self.log_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.log_dir / (
            "tensorboard-{:}".format(time.strftime("%d-%h", time.gmtime(time.time())))
        )

        self.logger_path = self.log_dir / "log"
        self.logger_file = open(self.logger_path, "w")  # noqa: SIM115
        self.writer = None

    def set_up_new_run(self) -> None:
        run_time = time.strftime("%Y-%d-%h-%H:%M:%S", time.gmtime(time.time()))
        self.set_up_run(run_time)

    def set_up_run(self, new_run_time: str | None = None) -> None:
        parts = self.log_dir.parts
        log_dir = "/".join(parts[:-4])
        exp_name = parts[-4]
        search_space = parts[-3]
        seed = parts[-2]
        run_time = parts[-1]

        if new_run_time:
            run_time = new_run_time

        self.save_last_run(
            run_time=run_time,
            log_dir=log_dir,
            exp_name=exp_name,
            search_space=search_space,
            seed=seed,
        )
        self.log_dir = Path(log_dir) / exp_name / search_space / seed / run_time

        self.log_dir.mkdir(parents=True, exist_ok=True)
        (Path(self.log_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.log_dir / (
            "tensorboard-{:}".format(time.strftime("%d-%h", time.gmtime(time.time())))
        )

        self.logger_path = self.log_dir / "log"
        self.logger_file = open(self.logger_path, "w")  # noqa: SIM115

    def load_last_run(
        self, log_dir: str, exp_name: str, search_space: str, seed: str
    ) -> str:
        file_path = Path(log_dir) / exp_name / search_space / seed / "last_run"

        with open(file_path) as f:
            run_time = f.read().strip()
        return run_time

    def save_last_run(
        self, run_time: str, log_dir: str, exp_name: str, search_space: str, seed: str
    ) -> str:
        file_path = Path(log_dir) / exp_name / search_space / seed / "last_run"

        with open(file_path, "w") as f:
            f.write(run_time)
        return run_time

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dir={self.log_dir}, writer={self.writer})"

    def path(self, mode: str | None) -> str:
        valids = (
            "best_model",  # checkpoint containing the best model
            "checkpoints",  # checkpoint of all the checkpoints (periodic)
            "log",  # path to the logger file
            "last_checkpoint",  # return the last checkpoint in the checkpoints folder
            None,
        )

        if mode not in valids:
            raise TypeError(f"Unknow mode = {mode}, valid modes = {valids}")
        if mode == "best_model":
            return str(self.log_dir / (mode + ".pth"))
        if mode == "last_checkpoint":
            last_checkpoint_path = self.log_dir / "checkpoints" / "last_checkpoint"
            with open(last_checkpoint_path) as f:
                return str(self.log_dir / "checkpoints" / f.read().strip())
        if mode is None:
            return str(self.log_dir)
        return str(self.log_dir / mode)

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

    def wandb_log_metrics(
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

        wandb.log(log_metrics)  # type: ignore
