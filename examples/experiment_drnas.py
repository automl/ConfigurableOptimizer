from __future__ import annotations

import argparse
import json

import wandb

from confopt.profiles import DiscreteProfile, DRNASProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType

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


if __name__ == "__main__":
    args = read_args()
    assert args.searchspace in ["darts", "nb201"], "Unsupported searchspace"
    searchspace = SearchSpaceType(args.searchspace)  # type: ignore
    dataset = DatasetType(args.dataset)  # type: ignore
    seed = args.seed

    # Sampler and Perturbator have different sample_frequency
    profile = DRNASProfile(
        is_partial_connection=args.searchspace == "darts",
        epochs=args.search_epochs,
        sampler_sample_frequency="step",
    )
    # nb201 take in default configs, but for darts, we require different config
    searchspace_config = {
        "num_classes": dataset_size[args.dataset],  # type: ignore
    }
    if args.searchspace == "darts":
        profile.configure_partial_connector(k=4)
        searchspace_config.update({"C": 36, "layers": 20})
    profile.set_searchspace_config(searchspace_config)

    train_config = {
        "train_portion": 0.5,
        "batch_size": 64,
        "optim_config": {
            "weight_decay": 3e-4,
            "momentum": 0.9,
            "nesterov": 0,
        },
        "arch_optim_config": {
            "betas": (0.5, 0.999),
            "weight_decay": 1e-3,
        },
        "learning_rate_min": 0.001,
    }
    profile.configure_trainer(**train_config)
    discrete_profile = DiscreteProfile(epochs=args.eval_epochs, train_portion=0.9)
    discrete_profile.configure_trainer(batch_size=64)

    discrete_config = discrete_profile.get_trainer_config()
    profile.configure_extra(
        {
            "discrete_trainer": discrete_config,
            "project_name": "BASELINES",
            "run_type": "DRNAS",
        }
    )
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
        exp_name="DRNAS_BASELINE",
    )

    trainer = experiment.run_with_profile(profile)

    discret_trainer = experiment.run_discrete_model_with_profile(
        discrete_profile,
        # start_epoch=args.eval_epochs,
        # load_saved_model=args.load_saved_model,
        # load_best_model=args.load_best_model,
    )
    if IS_WANDB_LOG:
        wandb.finish()  # type: ignore
