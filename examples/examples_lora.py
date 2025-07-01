from __future__ import annotations

import json

from confopt.profile import DARTSProfile
from confopt.train import Experiment
from confopt.enums import DatasetType, SearchSpaceType

if __name__ == "__main__":
    searchspace = SearchSpaceType("darts")
    dataset = DatasetType("cifar10")
    seed = 100

    profile = DARTSProfile(
        trainer_preset="darts",
        is_partial_connection=True,
        perturbation="random",
        sampler_sample_frequency="step",
        perturbator_sample_frequency="epoch",
        epochs=20,
        lora_rank=4,  # comment out this line and the next to run regular DARTS
        lora_warm_epochs=10,  # comment out to run DARTS
    )

    config = profile.get_config()
    print(json.dumps(config, indent=2, default=str))
    IS_DEBUG_MODE = True  # Set to False for a full run

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
    )

    search_trainer = experiment.train_supernet(profile)
