from __future__ import annotations

from enum import Enum
from confopt.enums import DatasetType, SearchSpaceType
from confopt.profile import BaseProfile, DARTSProfile, DiscreteProfile
from confopt.searchspace.darts.core.genotypes import PRIMITIVES
from confopt.train import Experiment


class BenchSuiteSpace(Enum):
    WIDE = "wide"
    DEEP = "deep"
    SINGLE_CELL = "single_cell"

    def __str__(self) -> str:
        return self.value


class BenchSuiteOpSet(Enum):
    REGULAR = "regular"
    NO_SKIP = "no_skip"
    ALL_SKIP = "all_skip"

    def __str__(self) -> str:
        return self.value


SEARCH_SPACE_CONFIGS = {
    BenchSuiteSpace.WIDE: {
        "C": 18,
        "layers": 4,
    },
    BenchSuiteSpace.DEEP: {
        "C": 7,
        "layers": 18,
    },
    BenchSuiteSpace.SINGLE_CELL: {
        "C": 26,
        "layers": 1,
        "steps": 8,
    },
}

OPSET_CONFIGS = {
    BenchSuiteOpSet.REGULAR: {"primitives": PRIMITIVES},
    BenchSuiteOpSet.NO_SKIP: {
        "primitives": [prim for prim in PRIMITIVES if prim != "skip_connect"]
    },
    BenchSuiteOpSet.ALL_SKIP: {
        "primitives": PRIMITIVES,
    },
}

MODEL_TRAIN_HYPERPARAMETERS = {
    1: {
        "lr": 0.025,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 1e-4,
        },
    },
    2: {
        "lr": 0.025,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 3e-4,
        },
    },
    3: {
        "lr": 0.025,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 1e-3,
        },
    },
    4: {
        "lr": 0.1,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 1e-4,
        },
    },
    5: {
        "lr": 0.1,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 3e-4,
        },
    },
    6: {
        "lr": 0.1,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 1e-3,
        },
    },
    7: {
        "lr": 0.01,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 1e-4,
        },
    },
    8: {
        "lr": 0.01,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 3e-4,
        },
    },
    9: {
        "lr": 0.01,
        "batch_size": 512,
        "optim_config": {
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 1e-3,
        },
    },
}


def configure_profile_with_search_space(
    profile: BaseProfile,
    space: BenchSuiteSpace,
    opset: BenchSuiteOpSet,
) -> None:
    search_space = SEARCH_SPACE_CONFIGS[space]
    operations = OPSET_CONFIGS[opset]
    profile.configure_searchspace(**search_space, **operations)

    if opset == BenchSuiteOpSet.ALL_SKIP:
        profile.use_auxiliary_skip_connection = True


def configure_discrete_profile_with_search_space(
    profile: DiscreteProfile,
    space: BenchSuiteSpace,
    opset: BenchSuiteOpSet,
) -> None:
    search_space = SEARCH_SPACE_CONFIGS[space]

    if space == BenchSuiteSpace.SINGLE_CELL:
        search_space.pop("steps", None)

    profile.configure_searchspace(**search_space)
    profile.configure_extra(
        subspace=space,
        opset=opset,
    )

    if opset == BenchSuiteOpSet.ALL_SKIP:
        searchspace_config = {"use_auxiliary_skip_connection": True}
        profile.configure_searchspace(**searchspace_config)


def configure_discrete_profile_with_hp_set(
    profile: DiscreteProfile,
    hyperparameter_set: int,
) -> None:
    hyperparameters = MODEL_TRAIN_HYPERPARAMETERS[hyperparameter_set]
    profile.configure_trainer(**hyperparameters)


if __name__ == "__main__":
    profile = DARTSProfile(
        searchspace_type=SearchSpaceType.DARTS,
        epochs=10,
    )

    configure_profile_with_search_space(
        profile,
        space=BenchSuiteSpace.WIDE,
        opset=BenchSuiteOpSet.NO_SKIP,
    )

    profile.configure_searchspace(
        C=4,
    )

    experiment = Experiment(
        search_space=SearchSpaceType.DARTS,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        debug_mode=True,
        exp_name="darts-shallow-wide-no-skip-debug-run",
    )
    experiment.train_supernet(profile)
