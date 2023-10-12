from __future__ import annotations

from torch import nn, optim
import wandb

from confopt.dataset import CIFAR10Data
from confopt.oneshot.archsampler import DARTSSampler
from confopt.searchspace import NASBench201SearchSpace
from confopt.train import ConfigurableTrainer, Profile
from confopt.utils import prepare_logger


def get_hyperparameters() -> dict:
    # This is just for test
    # TODO build a yaml config file for each of the config
    # Change the parameters here for now
    hyperparameters = {
        "model_lr": 1e-3,
        "arch_lr": 1e-3,
        "model_momentum": 0.0,
        "arch_betas": (0.9, 0.99),
        "weight_decay": 0.1,
        "epochs": 1,
        "dataset": "CIFAR10",
        "exp_name": "DARTS",
    }
    return hyperparameters


# Tie logger and wandb together
logger = prepare_logger(save_dir="logs", seed=0, exp_name="DARTS")


def run_experiment() -> None:
    wandb.init(  # type: ignore
        project="Configurable_Optimizers", config=get_hyperparameters()
    )

    config = wandb.config  # type: ignore
    data = CIFAR10Data("datasets", 0, 0.5)
    search_space = NASBench201SearchSpace()
    sampler = DARTSSampler(search_space.arch_parameters)
    # TODO Add pertubration and partial connections into this
    model_optimizer = optim.SGD(
        search_space.arch_parameters,
        lr=config["model_lr"],
        momentum=config["model_momentum"],
        weight_decay=config["weight_decay"],
    )
    arch_optimizer = optim.Adam(
        search_space.arch_parameters,
        lr=config["arch_lr"],
        betas=config["arch_betas"],
        weight_decay=0.1,
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, T_max=config["epochs"]
    )
    criterion = nn.CrossEntropyLoss()
    profile = Profile(sampler)

    trainer = ConfigurableTrainer(
        model=search_space,
        data=data,
        model_optimizer=model_optimizer,
        arch_optimizer=arch_optimizer,
        scheduler=lr_scheduler,
        criterion=criterion,
        batchsize=16,
        logger=logger,
    )
    trainer.train(profile, epochs=config["epochs"], is_wandb_log=True)
    logger.close()
    wandb.finish()  # type: ignore


if __name__ == "__main__":
    run_experiment()
