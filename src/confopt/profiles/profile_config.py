from __future__ import annotations

from abc import abstractmethod

import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO Change this to real data
ADVERSERIAL_DATA = (
    torch.randn(2, 3, 32, 32).to(DEVICE),
    torch.randint(0, 9, (2,)).to(DEVICE),
)


class ProfileConfig:
    def __init__(
        self,
        config_type: str,
        epochs: int = 100,
        is_partial_connection: bool = False,
        dropout: float | None = None,
        perturbation: str | None = None,
        perturbator_sample_frequency: str = "epoch",
        lora_rank: int = 0,
        lora_warm_epochs: int = 0,
    ) -> None:
        self.config_type = config_type
        self.epochs = epochs
        self.lora_warm_epochs = lora_warm_epochs
        self._initialize_trainer_config()
        self._initialize_sampler_config()
        self._set_partial_connector(is_partial_connection)
        self._set_lora_configs(lora_rank)
        self._set_dropout(dropout)
        self._set_perturb(perturbation, perturbator_sample_frequency)

    def _set_lora_configs(
        self,
        lora_rank: int = 0,
        lora_dropout: float = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
    ) -> None:
        self.lora_config = {
            "r": lora_rank,
            "lora_dropout": lora_dropout,
            "lora_alpha": lora_alpha,
            "merge_weights": merge_weights,
        }

    def _set_perturb(
        self,
        perturb_type: str | None = None,
        perturbator_sample_frequency: str = "epoch",
    ) -> None:
        assert perturbator_sample_frequency in ["epoch", "step"]
        assert perturb_type in ["adverserial", "random", "none", None]
        if perturb_type is None:
            self.perturb_type = "none"
        else:
            self.perturb_type = perturb_type
        self.perturbator_sample_frequency = perturbator_sample_frequency
        self._initialize_perturbation_config()

    def _set_partial_connector(self, is_partial_connection: bool = False) -> None:
        self.is_partial_connection = is_partial_connection
        self._initialize_partial_connector_config()

    def _set_dropout(self, dropout: float | None = None) -> None:
        self.dropout = dropout
        self._initialize_dropout_config()

    def get_config(self) -> dict:
        assert (
            self.sampler_config is not None
        ), "atleast a sampler is needed to initialize the search space"
        config = {
            "sampler": self.sampler_config,
            "perturbator": self.perturb_config,
            "partial_connector": self.partial_connector_config,
            "dropout": self.dropout_config,
            "trainer": self.trainer_config,
            "lora": self.lora_config,
        }
        if hasattr(self, "searchspace_config") and self.searchspace_config is not None:
            config.update({"search_space": self.searchspace_config})

        if hasattr(self, "extra_config") and self.extra_config is not None:
            config.update(self.extra_config)
        return config

    @abstractmethod
    def _initialize_sampler_config(self) -> None:
        self.sampler_config = None

    @abstractmethod
    def _initialize_perturbation_config(self) -> None:
        if self.perturb_type == "adverserial":
            perturb_config = {
                "epsilon": 0.3,
                "data": ADVERSERIAL_DATA,
                "loss_criterion": torch.nn.CrossEntropyLoss(),
                "steps": 20,
                "random_start": True,
                "sample_frequency": self.perturbator_sample_frequency,
            }
        elif self.perturb_type == "random":
            perturb_config = {
                "epsilon": 0.3,
                "sample_frequency": self.perturbator_sample_frequency,
            }
        else:
            perturb_config = None

        self.perturb_config = perturb_config

    @abstractmethod
    def _initialize_partial_connector_config(self) -> None:
        partial_connector_config = {"k": 4} if self.is_partial_connection else None
        self.partial_connector_config = partial_connector_config

    @abstractmethod
    def _initialize_trainer_config(self) -> None:
        trainer_config = {
            "lr": 0.025,
            "arch_lr": 3e-4,
            "epochs": self.epochs,
            "lora_warm_epochs": self.lora_warm_epochs,
            "optim": "sgd",
            "arch_optim": "adam",
            "optim_config": {
                "momentum": 0.9,
                "nesterov": 0,
                "weight_decay": 3e-4,
            },
            "arch_optim_config": {
                "weight_decay": 1e-3,
            },
            "scheduler": "cosine_annealing_warm_restart",
            "criterion": "cross_entropy",
            "batch_size": 64,
            "learning_rate_min": 0.0,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.7,
            "use_data_parallel": True,
            "checkpointing_freq": 1,
        }

        self.trainer_config = trainer_config

    @abstractmethod
    def _initialize_dropout_config(self) -> None:
        dropout_config = {
            "p": self.dropout if self.dropout is not None else 0.0,
            "p_min": 0.0,
            "anneal_frequency": "epoch",
            "anneal_type": "linear",
            "max_iter": self.epochs,
        }
        self.dropout_config = dropout_config

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        assert self.sampler_config is not None
        for config_key in kwargs:
            assert (
                config_key in self.sampler_config  # type: ignore
            ), f"{config_key} not a valid configuration for the sampler of type \
                {self.config_type}"
            self.sampler_config[config_key] = kwargs[config_key]  # type: ignore

    def configure_perturbator(self, **kwargs) -> None:  # type: ignore
        assert (
            self.perturb_type != "none"
        ), "Perturbator is initialized with None, \
            re-initialize with random or adverserial"

        for config_key in kwargs:
            assert (
                config_key in self.perturb_config  # type: ignore
            ), f"{config_key} not a valid configuration for the perturbator of \
                type {self.perturb_type}"
            self.perturb_config[config_key] = kwargs[config_key]  # type: ignore

    def configure_partial_connector(self, **kwargs) -> None:  # type: ignore
        assert self.is_partial_connection is True
        for config_key in kwargs:
            assert (
                config_key in self.partial_connector_config  # type: ignore
            ), f"{config_key} not a valid configuration for the partial connector"
            self.partial_connector_config[config_key] = kwargs[  # type: ignore
                config_key
            ]

    def configure_trainer(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.trainer_config
            ), f"{config_key} not a valid configuration for the trainer"
            self.trainer_config[config_key] = kwargs[config_key]

    def configure_dropout(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.dropout_config
            ), f"{config_key} not a valid configuration for the dropout module"
            self.dropout_config[config_key] = kwargs[config_key]

    def configure_lora_config(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.lora_config
            ), f"{config_key} not a valid configuration for the lora layers"
            self.lora_config[config_key] = kwargs[config_key]

    @abstractmethod
    def set_searchspace_config(self, config: dict) -> None:
        self.searchspace_config = config

    @abstractmethod
    def configure_extra_config(self, config: dict) -> None:
        self.extra_config = config
