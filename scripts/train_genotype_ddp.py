from confopt.enums import TrainerPresetType
from confopt.profile.profiles import DiscreteProfile
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType

if __name__ == "__main__":
    profile = DiscreteProfile(trainer_preset=TrainerPresetType.DARTS)
    config = profile.get_trainer_config()
    profile.configure_trainer(use_ddp=True)
    config.update({"genotype": profile.get_genotype()})

    experiment = Experiment(
        search_space=SearchSpaceType.DARTS,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        log_with_wandb=False,
        exp_name="Debug Experiment",
    )

    profile.train_config["epochs"] = 1

    experiment.init_ddp()
    trainer = experiment.train_discrete_model(
        profile,
        model_to_load=0,
    )
    experiment.cleanup_ddp()
