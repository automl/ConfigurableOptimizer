from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
import wandb

from confopt.dataset import CIFAR10Data
from confopt.oneshot.archsampler import SNASSampler
from confopt.searchspace import DARTSSearchSpace
from confopt.train import ConfigurableTrainer
from confopt.utils import BaseProfile, prepare_logger


def get_hyperparameters() -> dict:
    # This is just for test
    # TODO build a yaml config file for each of the config
    # Change the parameters here for now
    hyperparameters = {
        "model_lr": 0.025,
        "arch_lr": 3e-4,
        "model_momentum": 0.9,
        "model_weight_decay": 3e-4,
        "arch_weight_decay": 1e-3,
        "arch_betas": (0.5, 0.999),
        "epochs": 1,
        "batchsize": 64,
        "dataset": "CIFAR10",
        "exp_name": "SNAS",
    }
    return hyperparameters


# Tie logger and wandb together
logger = prepare_logger(save_dir="logs", seed=0, exp_name="SNAS")


def run_experiment() -> None:
    wandb.init(  # type: ignore
        project="Configurable_Optimizers", name="SNAS", config=get_hyperparameters()
    )

    config = wandb.config  # type: ignore
    data = CIFAR10Data("datasets", 0, 0.5)
    search_space = DARTSSearchSpace()
    sampler = SNASSampler(arch_parameters=search_space.arch_parameters)

    model_optimizer = optim.SGD(
        search_space.arch_parameters,
        lr=config["model_lr"],
        momentum=config["model_momentum"],
        weight_decay=config["model_weight_decay"],
    )
    arch_optimizer = optim.Adam(
        search_space.arch_parameters,
        lr=config["arch_lr"],
        betas=config["arch_betas"],
        weight_decay=config["arch_weight_decay"],
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, T_max=config["epochs"]
    )
    criterion = nn.CrossEntropyLoss()
    profile = BaseProfile(sampler)

    trainer = ConfigurableTrainer(
        model=search_space,
        data=data,
        model_optimizer=model_optimizer,
        arch_optimizer=arch_optimizer,
        scheduler=lr_scheduler,
        criterion=criterion,
        batchsize=config["batchsize"],
        logger=logger,
    )
    trainer.train(profile, epochs=config["epochs"], is_wandb_log=True)
    logger.close()
    wandb.finish()  # type: ignore


if __name__ == "__main__":
    run_experiment()
