from __future__ import annotations

import argparse
import json

import wandb

from confopt.profile import DiscreteProfile, DRNASProfile
from confopt.train import Experiment
from confopt.enums import DatasetType, SearchSpaceType

dataset_size = {
    "cifar10": 10,
    "cifar100": 100,
    "imgnet16": 1000,
    "imgnet16_120": 120,
}


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DRNAS Baseline run", add_help=False)
    parser.add_argument(
        "--searchspace",
        default="nb201",
        help="search space in (darts, nb201)",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="dataset to be used (cifar10, cifar100, imagenet)",
        type=str,
    )
    parser.add_argument(
        "--search_epochs",
        default=100,
        help="number of epochs to train the supernet",
        type=int,
    )

    parser.add_argument(
        "--eval_epochs",
        default=100,
        help="number of epochs to train the discrete network",
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=100,
        help="random seed",
        type=int,
    )

    args = parser.parse_args()
    return args


def get_drnas_profile(args: argparse.Namespace) -> DRNASProfile:
    profile = DRNASProfile(
        searchspace=args.search_space,
        epochs=args.search_epochs,
        sampler_sample_frequency="step",
    )
    # nb201 take in default configs, but for darts, we require different config
    searchspace_config = {
        "num_classes": dataset_size[args.dataset],  # type: ignore
    }
    if args.searchspace == "darts":
        profile._set_partial_connector(is_partial_connection=True)
        profile.configure_partial_connector(k=4)
        searchspace_config.update({"C": 36, "layers": 20})
    profile.configure_searchspace(**searchspace_config)

    train_config = {
        "train_portion": 0.5,
        "batch_size": 128,
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

    return profile


if __name__ == "__main__":
    args = read_args()
    assert args.searchspace in ["darts", "nb201"], "Unsupported searchspace"
    searchspace = SearchSpaceType(args.searchspace)  # type: ignore
    dataset = DatasetType(args.dataset)  # type: ignore
    seed = args.seed

    profile = get_drnas_profile(args)

    discrete_profile = DiscreteProfile(searchspace=args.search_space, epochs=args.eval_epochs, train_portion=0.9)
    discrete_profile.configure_trainer(batch_size=128)

    discrete_config = discrete_profile.get_trainer_config()
    profile.configure_extra(
        **{
            "discrete_trainer": discrete_config,
            "project_name": "BASELINES",
            "run_type": "DRNAS",
        }
    )
    config = profile.get_config()

    print(json.dumps(config, indent=2, default=str))

    IS_DEBUG_MODE = False
    IS_WANDB_LOG = False
    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
        log_with_wandb=IS_WANDB_LOG,
        exp_name="DRNAS_BASELINE",
    )

    trainer = experiment.train_supernet(profile)

    discret_trainer = experiment.train_discrete_model(
        discrete_profile,
    )
    if IS_WANDB_LOG:
        wandb.finish()  # type: ignore
