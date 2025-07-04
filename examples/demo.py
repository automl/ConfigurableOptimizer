from __future__ import annotations

from confopt.profile import DARTSProfile, DRNASProfile, GDASProfile, ReinMaxProfile
from confopt.train import Experiment
from confopt.enums import DatasetType, TrainerPresetType, SearchSpaceType

if __name__ == "__main__":
    searchspace = SearchSpaceType.DARTS
    dataset = DatasetType.CIFAR10
    seed = 100

    reg_config = {
        "active_reg_terms": ["drnas", "flops"],
        "reg_weights": [0.3, 0.2],
        "loss_weight": 0.5,
        "drnas_config": {
            "reg_scale": 2e-3,
        },
    }

    profile = DRNASProfile(
        trainer_preset=TrainerPresetType.DARTS,
        epochs=10,
        oles=True,
        calc_gm_score=True,
        is_arch_attention_enabled=True,
        regularization_config=reg_config,
    )
    profile.configure_searchspace(layers=8, C=2)
    config = profile.get_config()

    print(config)
    IS_DEBUG_MODE = True  # Set to False for a full run

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
    )

    experiment.train_supernet(profile)
