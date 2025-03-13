from __future__ import annotations

import argparse
from confopt.profile import (
    DARTSProfile,
    # LambdaDARTSProfile,
    GDASProfile,
    DRNASProfile,
    ReinMaxProfile,
)

from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType
from benchsuite import (
    BenchSuiteSpace,
    BenchSuiteOpSet,
    configure_profile_with_search_space,
)

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", type=str)
parser.add_argument("--subspace", type=str)
parser.add_argument("--ops", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--tag", default="", type=str)
parser.add_argument("--oles", action="store_true", default=False)
parser.add_argument("--pcdarts", action="store_true", default=False)
parser.add_argument("--fairdarts", action="store_true", default=False)
parser.add_argument("--sdarts", choices=["none", "adverserial", "random"], default="none", type=str)
parser.add_argument("--perturbator_sample_frequency", choices=["epoch", "step"], default="epoch", type=str)

args = parser.parse_args()

if __name__ == "__main__":
    DEBUG_MODE = False
    WANDB_LOG = True
    SEARCHSPACE = SearchSpaceType.DARTS
    DATASET = DatasetType(args.dataset)
    DATASET_DIR = "/path/to/datasets"  # UPDATE WITH YOUR DATASET DIR!!!

    subspace = BenchSuiteSpace(args.subspace)
    opset = BenchSuiteOpSet(args.ops)

    profile_classes = {
        "darts": DARTSProfile,
        "gdas": GDASProfile,
        "drnas": DRNASProfile,
    }

    epochs_profiles = {
        "darts": 50,
        "gdas": 250,
        "drnas": 50,
    }

    num_classes = {
        DatasetType.CIFAR10: 10,
        DatasetType.AIRCRAFT: 30,
    }

    partial_connector_config = None
    if args.pcdarts:
        partial_connector_config = {
            "k": 4,
            "num_warm_epoch": 15,
        }

    regularization_config = None
    if args.fairdarts:
        fairdarts_weight = 10
        regularization_config = {
            "reg_weights": [fairdarts_weight],
            "loss_weight": 1,
            "active_reg_terms": ["fairdarts"],
            "drnas_config": {"reg_scale": 1e-3},
            "flops_config": {},
            "fairdarts_config": {},
        }

    epochs = epochs_profiles[args.optimizer]
    profile = profile_classes[args.optimizer](
        searchspace=SEARCHSPACE,
        epochs=epochs,
        is_partial_connection=args.pcdarts,
        perturbation=args.sdarts,
        perturbator_sample_frequency=args.perturbator_sample_frequency,
        partial_connector_config=partial_connector_config,
        seed=args.seed,
        oles=args.oles,
        calc_gm_score=args.oles,
        is_regularization_enabled=args.fairdarts,
        regularization_config=regularization_config,
    )

    configure_profile_with_search_space(
        profile,
        space=subspace,
        opset=opset,
    )

    exp_name = (
        f"{SEARCHSPACE}-{subspace}-{opset}-{profile.SAMPLER_TYPE}-{DATASET}-e{epochs}"
    )

    profile.configure_trainer(
        checkpointing_freq=10,
    )

    profile.configure_searchspace(
        num_classes=num_classes[DATASET],
    )

    profile.configure_extra(
        space=subspace,
        opset=opset,
        benchmark=f"{subspace}-{opset}",
        tag=args.tag,
    )

    experiment = Experiment(
        search_space=SEARCHSPACE,
        dataset=DATASET,
        seed=args.seed,
        debug_mode=DEBUG_MODE,
        exp_name=exp_name,
        log_with_wandb=WANDB_LOG,
        dataset_dir=DATASET_DIR,
    )
    experiment.train_supernet(profile)
