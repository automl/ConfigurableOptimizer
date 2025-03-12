from __future__ import annotations
import argparse

from confopt.profile import DARTSProfile, DRNASProfile, GDASProfile
from confopt.profile.base import BaseProfile
from confopt.profile.profiles import ReinMaxProfile
from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType
from confopt.utils import get_num_classes

def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Synthetic Experiment", add_help=False)

    parser.add_argument(
        "--sampler",
        default="darts",
        help="arch sampler to use (darts, drnas, gdas, reinmax)",
        type=str,
    )

    parser.add_argument(
        "--search_epochs",
        default=100,
        help="number of epochs to train the supernet",
        type=int,
    )
    
    parser.add_argument(
        "--signal-receptive-field",
        default=5,
        help="receptive field of the signal",
        type=int,
    )

    parser.add_argument(
        "--shortcut-receptive-field",
        default=3,
        help="receptive field of the signal",
        type=int,
    )

    parser.add_argument(
        "--shortcut_strength",
        default=0.1,
        help="strength of shortcut",
        type=float,
    )

    parser.add_argument(
        "--seed",
        default=9001,
        help="Seed for the experiment",
        type=int,
    )

    args = parser.parse_args()
    return args

def get_profile(args: argparse.Namespace) -> BaseProfile:
    if args.sampler == "darts":
        return DARTSProfile

    if args.sampler == "gdas":
        return GDASProfile

    if args.sampler == "drnas":
        return DRNASProfile

    if args.sampler == "reinmax":
        return ReinMaxProfile


if __name__ == "__main__":
    args = read_args()

    searchspace = SearchSpaceType.BABYDARTS
    dataset = DatasetType.SYNTHETIC

    profile = get_profile(args)(searchspace=searchspace, epochs=args.search_epochs)

    profile.configure_synthetic_dataset(
        signal_receptive_field=args.signal_receptive_field,
        shortcut_receptive_field=args.shortcut_receptive_field,
        shortcut_strength=args.shortcut_strength,
    )

    profile.configure_searchspace(
        num_classes=get_num_classes(dataset.value),
        primitives=[
            "skip_connect",
            "sep_conv_3x3",
        ],
    )
    project_name = "Synthetic-Benchsuite"
    exp_name=(f"synthetic-test-{args.sampler}-"
            f"sig{args.signal_receptive_field}x{args.signal_receptive_field}-"
            f"short{args.shortcut_receptive_field}x{args.shortcut_receptive_field}-"
            f"strength{args.shortcut_strength:.3f}"
    )
    profile.configure_extra(project_name=project_name, meta_info=exp_name)
    # Configure experiment parameters
    experiment = Experiment(
        search_space=SearchSpaceType.BABYDARTS,
        dataset=DatasetType.SYNTHETIC,
        seed=args.seed,
        debug_mode=False,
        exp_name=exp_name,
        log_with_wandb=False,
    )

    # Execute the training process
    experiment.train_supernet(profile)
