from confopt.profiles.profiles import DiscreteProfile
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType

if __name__ == "__main__":
    profile = DiscreteProfile()
    config = profile.get_trainer_config()
    profile.configure_trainer(use_ddp=True)
    config.update({"genotype": profile.get_genotype()})

    experiment = Experiment(
        search_space=SearchSpaceType.DARTS,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        is_wandb_log=False,
        exp_name="Debug Experiment",
        runtime="now",
    )

    profile.train_config["epochs"] = 1


    experiment.init_ddp()
    trainer = experiment.train_discrete_model(
        profile,
        start_epoch=0,
        load_saved_model=False,
    )
    experiment.cleanup_ddp()
