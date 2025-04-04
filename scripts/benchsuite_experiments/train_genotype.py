from __future__ import annotations
import os
import argparse

from confopt.profile.profiles import DiscreteProfile
from confopt.train import Experiment
from benchsuite import (  # type: ignore
    configure_discrete_profile_with_search_space,  # type: ignore
    configure_discrete_profile_with_hp_set,  # type: ignore
    BenchSuiteOpSet,
    BenchSuiteSpace,
)
from confopt.enums import SearchSpaceType, DatasetType

WANDB_LOG = True
DEBUG_MODE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "cifar10",
            "cifar10_model",
            "cifar100",
            "imagenet16",
            "imgnet16_120",
            "taskonomy",
            "aircraft",
        ],
        default="cifar10_model",
        type=str,
        help="dataset to use",
    )
    parser.add_argument(
        "--optimizer",
        choices=["darts", "gdas", "drnas"],
        required=True,
        type=str,
        help="Optimizer which was used to obtain this model",
    )
    parser.add_argument(
        "--subspace",
        choices=["wide", "deep", "single_cell"],
        required=True,
        type=str,
        help="benchsuite type to use",
    )
    parser.add_argument(
        "--opset",
        choices=["regular", "no_skip", "all_skip"],
        required=True,
        type=str,
        help="benchsuite operation set to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--searchspace",
        type=str,
        default="darts",
    )
    parser.add_argument("--tag", type=str, required=True, help="tag for the experiment")
    parser.add_argument(
        "--other",
        type=str,
        default="baseline",
        choices=["baseline", "fairdarts", "oles", "pcdarts", "sdarts"],
        help="other optimizer (FairDARTS, PC-DARTS etc)",
    )
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    parser.add_argument(
        "--hpset",
        type=int,
        required=True,
        help="hyperparameter set to use to train this model",
    )

    parser.add_argument(
        "--genotypes_folder",
        type=str,
        default="genotypes",
        help="path to the file of the genotype you want to run.",
    )

    return parser.parse_args()


def set_profile_genotype(discrete_profile: DiscreteProfile, path: str) -> None:
    with open(path, "r") as file:
        genotype = file.read()
    discrete_profile.genotype = str(genotype)


def main(args: argparse.Namespace, hpset: int) -> None:

    discrete_profile = DiscreteProfile(
        searchspace_type="darts",
        epochs=args.epochs,
        seed=args.seed,
    )
    configure_discrete_profile_with_search_space(
        profile=discrete_profile,
        space=BenchSuiteSpace(args.subspace),
        opset=BenchSuiteOpSet(args.opset),
    )

    configure_discrete_profile_with_hp_set(
        profile=discrete_profile,
        hyperparameter_set=hpset,
    )

    genotype_filename = (
        f"{args.optimizer}-{args.other}-{args.subspace}-{args.opset}.txt"
    )
    genotype_filepath = os.path.join(args.genotypes_folder, genotype_filename)

    discrete_profile.configure_extra(
        project_name="ConfoptAutoML25-Models",
        tag=args.tag,
        hyperparameter_set=hpset,
        optimizer=args.optimizer,
        optimizer_other=args.other,
        dataset=args.dataset,
    )

    print("Path to genotype file: ", genotype_filepath)

    experiment = Experiment(
        search_space=SearchSpaceType(args.searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        log_with_wandb=WANDB_LOG,
        debug_mode=DEBUG_MODE,
        exp_name=f"{genotype_filepath.split('/')[-1].split('.')[0]}",
    )

    set_profile_genotype(discrete_profile, genotype_filepath)

    print("Training model with genotype: ", discrete_profile.genotype)
    print("Training model with search space: ", discrete_profile.searchspace_config)

    experiment.train_discrete_model(discrete_profile)


if __name__ == "__main__":
    args = parse_args()
    main(args, args.hpset)
