from __future__ import annotations

from confopt.profile import GDASProfile
from confopt.train import Experiment
from confopt.enums import DatasetType, SearchSpaceType

if __name__ == "__main__":
    search_space = SearchSpaceType.DARTS

    profile = GDASProfile(
        searchspace_type=search_space,
        epochs=10,
        perturbation="random",
        entangle_op_weights=True,
        lora_rank=1,
        lora_warm_epochs=5,
        prune_epochs=[3, 6],
        prune_fractions=[0.2, 0.2],
    )

    profile.configure_searchspace(layers=4, C=76)
    profile.configure_lora(
        lora_alpha=2.0,  # used in scaling in LoRA
    )
    profile.configure_perturbator(
        epsilon=0.1,  # epsilon pertubation to add to arch parameters
    )

    # Configure the Trainer
    profile.configure_trainer(
        lr=0.03,  # lr of model optimizer
        arch_lr=3e-4,  # lr of arch optimizer
        optim="sgd",  # model optimizer
        arch_optim="adam",  # arch optimizer
        optim_config={  # configuration of the model optimizer
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 3e-4,
        },
        arch_optim_config={  # configuration of the arch optimizer
            "weight_decay": 1e-3,
        },
        scheduler="cosine_annealing_warm_restart",
        batch_size=4,
        train_portion=0.7,  # portion of data to use for training the model
        use_data_parallel=False,  # Use UseDataParallel is enabled
        checkpointing_freq=5,  # How frequently to save the supernet
    )

    # Add any additional configurations to this run
    # Used to tell runs apart in WandB, if required
    profile.configure_extra(
        **{
            "project_name": "my-wandb-projectname",  # Name of the Wandb Project
            "run_purpose": "my-run-purpose",  # Purpose of the run
        }
    )

    experiment = Experiment(
        search_space=search_space,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        debug_mode=True,
        exp_name="demo-advanced",
        log_with_wandb=True,  # enable logging with Weights and Biases
    )

    experiment.train_supernet(
        profile,
        use_benchmark=False,  # query the benchmark at the end of every epoch
    )
