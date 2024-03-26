from __future__ import annotations

from abc import ABC

from .profile_config import ProfileConfig


class DartsProfile(ProfileConfig, ABC):
    def __init__(
        self,
        epochs: int,
        is_partial_connection: bool = False,
        dropout: float | None = None,
        perturbation: str | None = None,
        sampler_sample_frequency: str = "step",
        perturbator_sample_frequency: str = "epoch",
        partial_connector_config: dict | None = None,
        perturbator_config: dict | None = None,
        entangle_op_weights: bool = False,
    ) -> None:
        PROFILE_TYPE = "DARTS"
        self.sampler_type = str.lower(PROFILE_TYPE)
        self.sampler_sample_frequency = sampler_sample_frequency
        super().__init__(
            PROFILE_TYPE,
            epochs,
            is_partial_connection,
            dropout,
            perturbation,
            perturbator_sample_frequency,
            entangle_op_weights,
        )

        if partial_connector_config is not None:
            self.configure_partial_connector(**partial_connector_config)

        if perturbator_config is not None:
            self.configure_perturbator(**perturbator_config)

    def _initialize_sampler_config(self) -> None:
        darts_config = {"sample_frequency": self.sampler_sample_frequency}
        self.sampler_config = darts_config  # type: ignore


class GDASProfile(ProfileConfig, ABC):
    def __init__(
        self,
        epochs: int,
        is_partial_connection: bool = False,
        dropout: float | None = None,
        perturbation: str | None = None,
        sampler_sample_frequency: str = "step",
        perturbator_sample_frequency: str = "epoch",
        tau_min: float = 0.1,
        tau_max: float = 10,
        partial_connector_config: dict | None = None,
        perturbator_config: dict | None = None,
        entangle_op_weights: bool = False,
    ) -> None:
        PROFILE_TYPE = "GDAS"
        self.sampler_type = str.lower(PROFILE_TYPE)
        self.sampler_sample_frequency = sampler_sample_frequency
        self.tau_min = tau_min
        self.tau_max = tau_max
        super().__init__(
            PROFILE_TYPE,
            epochs,
            is_partial_connection,
            dropout,
            perturbation,
            perturbator_sample_frequency,
            entangle_op_weights,
        )

        if partial_connector_config is not None:
            self.configure_partial_connector(**partial_connector_config)

        if perturbator_config is not None:
            self.configure_perturbator(**perturbator_config)

    def _initialize_sampler_config(self) -> None:
        gdas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
        }
        self.sampler_config = gdas_config  # type: ignore


class SNASProfile(ProfileConfig, ABC):
    def __init__(
        self,
        epochs: int,
        is_partial_connection: bool = False,
        dropout: float | None = None,
        perturbation: str | None = None,
        sampler_sample_frequency: str = "step",
        perturbator_sample_frequency: str = "epoch",
        temp_init: float = 1.0,
        temp_min: float = 0.33,
        temp_annealing: bool = True,
        total_epochs: int = 250,
        partial_connector_config: dict | None = None,
        perturbator_config: dict | None = None,
        entangle_op_weights: bool = False,
    ) -> None:
        PROFILE_TYPE = "SNAS"
        self.sampler_type = str.lower(PROFILE_TYPE)
        self.sampler_sample_frequency = sampler_sample_frequency
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_annealing = temp_annealing
        self.total_epochs = total_epochs
        super().__init__(  # type: ignore
            PROFILE_TYPE,
            epochs,
            is_partial_connection,
            dropout,
            perturbation,
            perturbator_sample_frequency,
            entangle_op_weights,
        )

        if partial_connector_config is not None:
            self.configure_partial_connector(**partial_connector_config)

        if perturbator_config is not None:
            self.configure_perturbator(**perturbator_config)

    def _initialize_sampler_config(self) -> None:
        snas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "temp_init": self.temp_init,
            "temp_min": self.temp_min,
            "temp_annealing": self.temp_annealing,
            "total_epochs": self.total_epochs,
        }
        self.sampler_config = snas_config  # type: ignore


class DRNASProfile(ProfileConfig, ABC):
    def __init__(
        self,
        epochs: int,
        is_partial_connection: bool = False,
        dropout: float | None = None,
        perturbation: str | None = None,
        sampler_sample_frequency: str = "step",
        perturbator_sample_frequency: str = "epoch",
        partial_connector_config: dict | None = None,
        perturbator_config: dict | None = None,
        entangle_op_weights: bool = False,
    ) -> None:
        PROFILE_TYPE = "DRNAS"
        self.sampler_type = str.lower(PROFILE_TYPE)
        self.sampler_sample_frequency = sampler_sample_frequency
        super().__init__(  # type: ignore
            PROFILE_TYPE,
            epochs,
            is_partial_connection,
            dropout,
            perturbation,
            perturbator_sample_frequency,
            entangle_op_weights,
        )

        if partial_connector_config is not None:
            self.configure_partial_connector(**partial_connector_config)

        if perturbator_config is not None:
            self.configure_perturbator(**perturbator_config)

    def _initialize_sampler_config(self) -> None:
        drnas_config = {
            "sample_frequency": self.sampler_sample_frequency,
        }
        self.sampler_config = drnas_config  # type: ignore


class DiscreteProfile:
    def __init__(self, **kwargs) -> None:  # type: ignore
        self._initialize_trainer_config()
        self.configure_trainer(**kwargs)

    def get_trainer_config(self) -> dict:
        return self.train_config

    def _initialize_trainer_config(self) -> None:
        default_train_config = {
            "lr": 0.025,
            "epochs": 100,
            "optim": "sgd",
            "optim_config": {
                "momentum": 0.9,
                "nesterov": 0,
                "weight_decay": 3e-4,
            },
            "criterion": "cross_entropy",
            "batch_size": 96,
            "learning_rate_min": 0.0,
            "channel": 36,
            "drop_path_prob": 0.2,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.7,
            "use_data_parallel": True,
            "checkpointing_freq": 1,
        }
        self.train_config = default_train_config

    def configure_trainer(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.train_config
            ), f"{config_key} not a valid configuration for training a \
            discrete architecture"
            self.train_config[config_key] = kwargs[config_key]
