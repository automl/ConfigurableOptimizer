from __future__ import annotations

import json

from confopt.profiles import DartsProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType

if __name__ == "__main__":
    searchspace = SearchSpaceType("darts")
    dataset = DatasetType("cifar10")
    seed = 100

    # Sampler and Perturbator have different sample_frequency
    profile = DartsProfile(
        is_partial_connection=True,
        perturbation="random",
        sampler_sample_frequency="step",
        perturbator_sample_frequency="epoch",
        epochs=20,
        lora_rank=4,
        lora_warm_epochs=10,
    )

    config = profile.get_config()
    print(json.dumps(config, indent=2, default=str))
    IS_DEBUG_MODE = True

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
    )

    search_trainer = experiment.run_with_profile(profile)
