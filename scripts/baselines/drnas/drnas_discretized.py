from __future__ import annotations

import argparse

from drnas_search_and_discrete import get_drnas_profile
import wandb

from confopt.profile import DiscreteProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType
from confopt.utils import validate_model_to_load_value

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

    parser.add_argument(
        "--model_to_load",
        type=validate_model_to_load_value,
        help="if str, could be best, last. If int, then load the checkpoint of that epoch",
        default=None,
    )

    parser.add_argument(
        "--last_search_runtime",
        default="NOT_VALID",
        help="The search run time to take in for the discretization step",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()
    assert args.searchspace in ["darts", "nb201"], "Unsupported searchspace"
    searchspace = SearchSpaceType(args.searchspace)  # type: ignore
    dataset = DatasetType(args.dataset)  # type: ignore
    seed = args.seed

    discrete_profile = DiscreteProfile(epochs=args.eval_epochs, train_portion=0.9)
    discrete_profile.configure_trainer(batch_size=128)

    discrete_config = discrete_profile.get_trainer_config()
    discrete_config.update({"seed": seed})  # for identifying runs in wandb

    IS_WANDB_LOG = False

    if IS_WANDB_LOG:
        wandb.init(  # type: ignore
            project="BASELINES",
            group="DRNAS",
            config=discrete_config,
        )

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        log_with_wandb=IS_WANDB_LOG,
        exp_name="DRNAS_BASELINE",
    )

    # bare minimum for initalizing experiment
    profile_arg_dict = {
        "search_epochs": 1,
        "dataset": args.dataset,
        "searchspace": args.searchspace,
    }
    profile_args = argparse.Namespace(**profile_arg_dict)
    drnas_profile = get_drnas_profile(profile_args)
    # Control from last_run
    experiment.initialize_from_last_run(
        profile_config=drnas_profile, last_search_runtime=args.last_search_runtime
    )

    discret_trainer = experiment.train_discrete_model(
        discrete_profile,
        model_to_load=args.model_to_load,
    )
    if IS_WANDB_LOG:
        wandb.finish()  # type: ignore
