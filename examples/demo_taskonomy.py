from __future__ import annotations

from confopt.profiles import GDASProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType
from confopt.utils import get_num_classes

if __name__ == "__main__":
    profile = GDASProfile(searchspace_str="tnb101", epochs=3)
    domain = "class_object"
    profile.set_searchspace_config({"num_classes": get_num_classes("taskonomy", domain)})
    experiment = Experiment(
        search_space=SearchSpaceType.TNB101,
        dataset=DatasetType.TASKONOMY,
        seed=9001,
        debug_mode=True,
        exp_name="demo-simple",
        domain=domain,
    )
    experiment.train_supernet(profile)
