from __future__ import annotations

from confopt.profiles import GDASProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType

if __name__ == "__main__":
    searchspace = SearchSpaceType("nb201")
    dataset = DatasetType("cifar10")
    seed = 100

    # Sampler and Perturbator have different sample_frequency
    profile = GDASProfile(
        # is_partial_connection=True,
        # perturbation="random",
        sampler_sample_frequency="step",
        perturbator_sample_frequency="epoch",
        tau_max=20,
        tau_min=0.2,
        epochs=1,
    )

    config = profile.get_config()
    print(config)
    IS_DEBUG_MODE = True

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
    )

    experiment.run_with_profile(profile)
