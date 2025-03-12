
from __future__ import annotations
import argparse

from confopt.profile.profiles import DiscreteProfile, DARTSProfile
from confopt.train import Experiment
from benchsuite import ( # type: ignore
    configure_discrete_profile_with_search_space,  # type: ignore
    BenchSuiteOpSet, 
    BenchSuiteSpace,
) 
from confopt.enums import SearchSpaceType, DatasetType

WANDB_LOG = False
DEBUG_MODE = True

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cirfar100", "imagenet16", "imgnet16_120", "taskonomy", "aircraft"],
        default="cifar10",
        type=str,
        help="dataset to use",
    )
    parser.add_argument(
        "--benchsuit_space",
        choices=["wide", "deep", "single_cell"],
        default="wide",
        type=str,
        help="benchsuit type to use",
    )
    parser.add_argument(
        "--benchsuit_op_set",
        choices=["regular", "no_skip", "all_skip"],
        default="regular",
        type=str,
        help="benchsuit operation set to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--searchspace",
        type=str,
        default="darts",
    )
    parser.add_argument(
        "--taskonomy_dataset_domain",
        choices=["class_object", "class_scene"],
        default="class_object",
        type=str,
        help="taskonomy dataset domain to use",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--genotype_path",
        type=str,
        default="scripts/benchsuite_experiments/genotype.txt",
        help="path to the file of the genotype you want to run."
    )
    return parser

def set_profile_genotype(discrete_profile: DARTSProfile, path: str) -> None:
    with open(path, 'r') as file:
        genotype = file.read()
    discrete_profile.genotype = str(genotype)

if __name__ == "__main__":
    args = parse_args().parse_args()
    discrete_profile = DiscreteProfile(
        searchspace="darts",
        epochs=args.epochs,
        seed=args.seed,
    )
    configure_discrete_profile_with_search_space(
        profile=discrete_profile,
        space=BenchSuiteSpace(args.benchsuit_space),
        opset=BenchSuiteOpSet(args.benchsuit_op_set),
    )
    experiment = Experiment(
        search_space=SearchSpaceType(args.searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        log_with_wandb=WANDB_LOG,
        debug_mode=DEBUG_MODE,
        exp_name=f"darts-benchsuit-{args.benchsuit_space}-{args.benchsuit_op_set}-debug-run",

    )
    
    set_profile_genotype(discrete_profile, args.genotype_path)
    experiment.train_discrete_model(discrete_profile)