from __future__ import annotations

import argparse
from confopt.profile import (
    DARTSProfile,
    GDASProfile,
    DRNASProfile,
)

from confopt.train import Experiment
from confopt.enums import SamplerType, SearchSpaceType, DatasetType
from benchsuite import (
    BenchSuiteSpace,
    BenchSuiteOpSet,
    configure_profile_with_search_space,
)


def bool_type(x: str) -> bool:
    return x.lower() in ["true", "1", "yes"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
    choices=["darts", "drnas", "gdas"],
    help="Choose the type of sampler",
)
parser.add_argument(
    "--subspace",
    type=str,
    required=True,
    choices=["deep", "wide", "single_cell"],
    help="Choose the type of subspace",
)
parser.add_argument(
    "--ops",
    type=str,
    required=True,
    choices=["regular", "no_skip", "all_skip"],
    help="Choose the opset for the run",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "cifar10",
        "cifar10_supernet",
        "cifar100",
        "imgnet16",
        "imgnet16_120",
        "taskonomy",
        "aircraft",
    ],
    help="Choose the opset for the run",
)
parser.add_argument("--seed", type=int, required=True, help="seed for the experiments")
parser.add_argument("--tag", type=str, required=True, help="tag for the experiment")
parser.add_argument(
    "--oles", type=bool_type, default="false", help="turn oles on for the experiment"
)
parser.add_argument(
    "--pcdarts",
    type=bool_type,
    default="false",
    help="use partial channels for the experiment",
)
parser.add_argument(
    "--fairdarts",
    type=bool_type,
    default="false",
    help="use fairdarts sampling scheme and regualrization",
)

parser.add_argument(
    "--sdarts",
    choices=["none", "adversarial", "random"],
    type=str,
    default="none",
    help="use perturbation for the experiments",
)

parser.add_argument(
    "--dryrun",
    type=bool_type,
    default="false",
    help="test things out with dryrun argument",
)

parser.add_argument(
    "--log_with_wandb",
    type=bool_type,
    default="false",
    help="test things out with dryrun argument",
)

args = parser.parse_args()


def read_config(file_path: str) -> dict[str, str]:
    config = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):  # Ignore empty lines and comments
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip().strip(
                    '"'
                )  # Remove surrounding quotes if any
    return config


batch_sizes = {
    SamplerType.DARTS: {
        BenchSuiteSpace.DEEP: 64,
        BenchSuiteSpace.WIDE: 96,
        BenchSuiteSpace.SINGLE_CELL: 96,
    },
    SamplerType.DRNAS: {
        BenchSuiteSpace.DEEP: 64,
        BenchSuiteSpace.WIDE: 96,
        BenchSuiteSpace.SINGLE_CELL: 96,
    },
    SamplerType.GDAS: {
        BenchSuiteSpace.DEEP: 320,
        BenchSuiteSpace.WIDE: 480,
        BenchSuiteSpace.SINGLE_CELL: 480,
    },
}

model_learning_rates = {
    SamplerType.DARTS: 0.025,
    SamplerType.DRNAS: 0.1,
    SamplerType.GDAS: 0.025,
}

arch_learning_rates = {
    SamplerType.DARTS: 3e-4,
    SamplerType.DRNAS: 6e-4,
    SamplerType.GDAS: 3e-4,
}

epochs_profiles = {
    "darts": 100,
    "gdas": 300,
    "drnas": 100,
}

num_classes = {
    DatasetType.CIFAR10: 10,
    DatasetType.CIFAR10_MODEL: 10,
    DatasetType.CIFAR10_SUPERNET: 10,
    DatasetType.AIRCRAFT: 30,
}

if __name__ == "__main__":
    WANDB_LOG = args.log_with_wandb
    SEARCHSPACE = SearchSpaceType.DARTS
    DATASET = DatasetType(args.dataset)

    subspace = BenchSuiteSpace(args.subspace)
    opset = BenchSuiteOpSet(args.ops)

    profile_classes = {
        "darts": DARTSProfile,
        "gdas": GDASProfile,
        "drnas": DRNASProfile,
    }

    partial_connector_config = None
    if args.pcdarts:
        partial_connector_config = {
            "k": 4,
            "num_warm_epoch": 15,
        }

    reg_weights = []
    active_reg_terms = []
    arch_combine_fn = "default"

    sampler_type = SamplerType(args.optimizer)

    if sampler_type == SamplerType.DRNAS:
        drnas_weight = 1
        reg_weights.append(drnas_weight)
        active_reg_terms.append("drnas")

    if args.fairdarts:
        fairdarts_weight = 10
        reg_weights.append(fairdarts_weight)
        active_reg_terms.append("fairdarts")
        arch_combine_fn = "sigmoid"

    regularization_config = None
    if active_reg_terms:
        regularization_config = {
            "reg_weights": reg_weights,
            "loss_weight": 1,
            "active_reg_terms": active_reg_terms,
            "drnas_config": {"reg_scale": 1e-3, "reg_type": "l2"},
            "flops_config": {},
            "fairdarts_config": {},
        }

    epochs = 3 if args.dryrun is True else epochs_profiles[args.optimizer]
    profile = profile_classes[args.optimizer](
        trainer_preset=SEARCHSPACE.value,
        epochs=epochs,
        is_partial_connection=args.pcdarts,
        perturbation=args.sdarts,
        perturbator_sample_frequency="epoch",
        partial_connector_config=partial_connector_config,
        seed=args.seed,
        oles=args.oles,
        calc_gm_score=True,
        regularization_config=regularization_config,
        sampler_arch_combine_fn=arch_combine_fn,
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
        batch_size=batch_sizes[sampler_type][subspace],
        lr=model_learning_rates[sampler_type],
        arch_lr=arch_learning_rates[sampler_type],
    )

    profile.configure_searchspace(
        num_classes=num_classes[DATASET],
    )

    profile.configure_extra(
        space=subspace,
        opset=opset,
        benchmark=f"{subspace}-{opset}",
        tag=args.tag,
        is_debug_run=args.dryrun,
        project_name="ConfoptAutoML25",
    )

    experiment = Experiment(
        search_space=SEARCHSPACE,
        dataset=DATASET,
        seed=args.seed,
        debug_mode=args.dryrun,
        exp_name=exp_name,
        log_with_wandb=WANDB_LOG,
    )
    experiment.train_supernet(profile)
