from __future__ import annotations

import argparse
import json

import wandb

from confopt.profiles import SNASProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType

dataset_size = {
    "cifar10": 10,
    "imgnet16": 1000,
    "imgnet16_120": 120,
}


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SNAS Baseline run", add_help=False)

    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="dataset to be used (cifar10, imgnet16)",
        type=str,
    )
    parser.add_argument(
        "--search_epochs",
        default=100,
        help="number of epochs to train the supernet",
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=100,
        help="random seed",
        type=int,
    )

    parser.add_argument(
        "--search_space",
        default="darts",
        help="search space to be used (darts, nb201, tnb101, nb1shot1)",
        type=str,
    )

    parser.add_argument(
        "--batch_size", default=64, help="batch size used to train", type=int
    )

    args = parser.parse_args()
    return args


def get_snas_profile(args: argparse.Namespace) -> SNASProfile:
    # This will help to have consistent profile for discretizing
    profile = SNASProfile(
        epochs=args.search_epochs,
        sampler_sample_frequency="step",
        dropout=1e-3,
        temp_init=1,
        temp_min=0.03,
    )
    searchspace_config = {
        "num_classes": dataset_size[args.dataset],  # type: ignore
    }

    profile.set_searchspace_config(searchspace_config)

    train_config = {
        "train_portion": 0.5,
        "batch_size": args.batch_size,
        "optim_config": {
            "weight_decay": 3e-4,
            "momentum": 0.9,
            "nesterov": False,
        },
        "arch_optim_config": {
            "betas": (0.5, 0.999),
            "weight_decay": 1e-3,
        },
        "learning_rate_min": 0.001,
    }
    profile.configure_trainer(**train_config)
    profile.configure_extra(
        {
            "project_name": "BASELINES",
            "run_type": "SNAS",
        }
    )
    return profile


if __name__ == "__main__":
    args = read_args()
    searchspace = SearchSpaceType(args.search_space)  # type: ignore
    dataset = DatasetType(args.dataset)  # type: ignore
    seed = args.seed

    # Sampler and Perturbator have different sample_frequency
    profile = get_snas_profile(args)
    config = profile.get_config()

    print(json.dumps(config, indent=2, default=str))

    IS_DEBUG_MODE = False
    IS_WANDB_LOG = True
    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
        is_wandb_log=IS_WANDB_LOG,
        exp_name="SNAS_BASELINE",
    )

    trainer = experiment.train_supernet(profile)

    if IS_WANDB_LOG:
        wandb.finish()  # type: ignore
