from __future__ import annotations

from confopt.profile import GDASProfile
from confopt.train import Experiment
from confopt.enums import DatasetType, SearchSpaceType
from confopt.utils import get_num_classes

if __name__ == "__main__":
    domain = "class_object"
    profile = GDASProfile(
        trainer_preset="tnb101", epochs=3, searchspace_domain=domain
    )
    profile.configure_searchspace(num_classes=get_num_classes("taskonomy", domain))
    experiment = Experiment(
        search_space=SearchSpaceType.TNB101,
        dataset=DatasetType.TASKONOMY,
        seed=9001,
        debug_mode=True,
        exp_name="demo-simple",
        dataset_domain=domain,
    )
    experiment.train_supernet(profile)
