from __future__ import annotations

from confopt.profile import DARTSProfile, LambdaDARTSProfile
from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType

if __name__ == "__main__":
    profile = LambdaDARTSProfile(
        searchspace=SearchSpaceType.DARTS, epochs=3,
    )
    profile.configure_searchspace(C=1)

    experiment = Experiment(
        search_space=SearchSpaceType.DARTS,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        debug_mode=True,
        exp_name="demo-simple",
    )
    experiment.train_supernet(profile)
