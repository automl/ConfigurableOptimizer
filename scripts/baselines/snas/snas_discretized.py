from __future__ import annotations

import argparse

from snas_search import get_snas_profile
import wandb

from confopt.profiles import DiscreteProfile
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
        "--search_space",
        default="darts",
        help="search space to be used - darts, nb201, tnb101",
        type=str,
    )

    parser.add_argument(
        "--eval_epochs",
        default=600,
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
        "--start_epoch", default=0, help="which epoch to start the training from"
    )

    parser.add_argument(
        "--load_saved_model",
        default=False,
        action="store_true",
        help="load from last saved checkpoint (based on seed)",
    )

    parser.add_argument(
        "--last_search_runtime",
        default="NOT_VALID",
        help="The search run time to take in for the discretization step",
        type=str,
    )

    parser.add_argument(
        "--batch_size", default=64, help="batch size used to train", type=int
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()
    searchspace = SearchSpaceType(args.search_space)  # type: ignore
    dataset = DatasetType(args.dataset)  # type: ignore
    seed = args.seed

    discrete_profile = DiscreteProfile(epochs=args.eval_epochs, train_portion=0.9)
    discrete_profile.configure_trainer(batch_size=args.batch_size)

    if args.search_space == "darts":
        discretize_search_space_config = {
            "C": 36,
            "layers": 20,
            "num_classes": dataset_size[args.dataset],
        }
        discrete_profile.train_config.update(
            {
                "search_space": discretize_search_space_config,
            }
        )

    discrete_config = discrete_profile.get_trainer_config()
    discrete_config.update({"run_nature": "discrete"})
    discrete_config.update({"seed": seed})  # for identifying runs in wandb

    IS_WANDB_LOG = True

    if IS_WANDB_LOG:
        wandb.init(  # type: ignore
            project="BASELINES",
            group="SNAS_" + str(args.search_space),
            config=discrete_config,
        )

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        is_wandb_log=IS_WANDB_LOG,
        exp_name="SNAS_BASELINE",
    )

    # get a bare minimum profile
    profile_arg_dict = {
        "search_epochs": 1,
        "dataset": args.dataset,
    }
    profile_args = argparse.Namespace(**profile_arg_dict)
    profile = get_snas_profile(profile_args)

    experiment.initialize_from_last_run(
        profile_config=profile,
        last_search_runtime=args.last_search_runtime,
    )

    discret_trainer = experiment.train_discrete_model(
        discrete_profile,
        start_epoch=int(args.start_epoch),
        load_saved_model=args.load_saved_model,
        # load_best_model=args.load_best_model,
    )

    if IS_WANDB_LOG:
        wandb.finish()  # type: ignore
