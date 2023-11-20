from __future__ import annotations

from abc import abstractmethod

import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO Change this to real data
ADVERSERIAL_DATA = torch.randn(2, 3, 32, 32).to(DEVICE), torch.randint(0, 9, (2,)).to(
    DEVICE
)


class ProfileConfig:
    def __init__(self, config_type: str) -> None:
        self.config_type = config_type

    def set_perturb(
        self,
        perturb_type: str | None = None,
        perturbator_sample_frequency: str = "epoch",
    ) -> None:
        assert perturb_type in ["adverserial", "random", "none", None]
        assert perturbator_sample_frequency in ["epoch", "step"]
        self.perturb_type = perturb_type
        self.perturbator_sample_frequency = perturbator_sample_frequency

    def set_partial_connector(self, is_partial_connection: bool = False) -> None:
        self.is_partial_connection = is_partial_connection

    def get_config(self) -> dict:
        sampler_config = self.get_sampler_config()
        perturb_config = self.get_perturb_config()
        partial_connector_config = self.get_partial_conenctor()
        trainer_config = self.get_trainer_config()
        config = {
            "sampler": sampler_config,
            "perturbator": perturb_config,
            "partial_connector": partial_connector_config,
            "trainer": trainer_config,
        }
        searchspace_config = self.get_searchspace_config()
        if searchspace_config is not None:
            config.update({"search_space": searchspace_config})
        return config

    @abstractmethod
    def get_sampler_config(self) -> dict:
        pass

    # User can overide this function
    @abstractmethod
    def get_perturb_config(self) -> dict | None:
        if self.perturb_type == "adverserial":
            perturb_config = {
                "epsilon": 0.03,
                "data": ADVERSERIAL_DATA,
                "loss_criterion": torch.nn.CrossEntropyLoss(),
                "steps": 20,
                "random_start": True,
                "sample_frequency": self.perturbator_sample_frequency,
            }
        elif self.perturb_type == "random":
            perturb_config = {
                "epsilon": 0.03,
                "sample_frequency": self.perturbator_sample_frequency,
            }
        else:
            return None
        return perturb_config

    # User can overide this function
    @abstractmethod
    def get_partial_conenctor(self) -> dict | None:
        partial_connector_config = {"k": 4}
        return partial_connector_config

    # User can overide this function
    @abstractmethod
    def get_trainer_config(self) -> dict:
        default_train_config = {
            "lr": 0.025,
            "epochs": 100,
            "optim": "sgd",
            "arch_optim": "adam",
            "momentum": 0.9,
            "nesterov": 0,
            "criterion": "cross_entropy",
            "batch_size": 4,
            "learning_rate_min": 0.0,
            "weight_decay": 3e-4,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.7,
            "use_data_parallel": 0,
            "checkpointing_freq": 1,
        }
        return default_train_config

    @abstractmethod
    def get_searchspace_config(self) -> dict:
        pass
