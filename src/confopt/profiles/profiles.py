from __future__ import annotations

from abc import ABC
from collections import namedtuple
from typing import Any

from confopt.utils import get_num_classes

from .profile_config import BaseProfile

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")


class DARTSProfile(BaseProfile, ABC):
    def __init__(
        self,
        epochs: int,
        **kwargs: Any,
    ) -> None:
        PROFILE_TYPE = "DARTS"

        super().__init__(
            PROFILE_TYPE,
            epochs,
            **kwargs,
        )
        self.sampler_type = str.lower(PROFILE_TYPE)

    def _initialize_sampler_config(self) -> None:
        darts_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "arch_combine_fn": self.sampler_arch_combine_fn,
        }
        self.sampler_config = darts_config  # type: ignore


class GDASProfile(BaseProfile, ABC):
    PROFILE_TYPE = "GDAS"

    def __init__(
        self,
        epochs: int,
        tau_min: float = 0.1,
        tau_max: float = 10,
        **kwargs: Any,
    ) -> None:
        self.tau_min = tau_min
        self.tau_max = tau_max

        super().__init__(
            self.PROFILE_TYPE,
            epochs,
            **kwargs,
        )
        self.sampler_type = str.lower(self.PROFILE_TYPE)

    def _initialize_sampler_config(self) -> None:
        gdas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "arch_combine_fn": self.sampler_arch_combine_fn,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
        }
        self.sampler_config = gdas_config  # type: ignore


class ReinMaxProfile(GDASProfile):
    PROFILE_TYPE = "REINMAX"


class SNASProfile(BaseProfile, ABC):
    def __init__(
        self,
        epochs: int,
        temp_init: float = 1.0,
        temp_min: float = 0.33,
        temp_annealing: bool = True,
        **kwargs: Any,
    ) -> None:
        PROFILE_TYPE = "SNAS"
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_annealing = temp_annealing
        self.total_epochs = epochs

        super().__init__(  # type: ignore
            PROFILE_TYPE,
            epochs,
            **kwargs,
        )
        self.sampler_type = str.lower(PROFILE_TYPE)

    def _initialize_sampler_config(self) -> None:
        snas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "arch_combine_fn": self.sampler_arch_combine_fn,
            "temp_init": self.temp_init,
            "temp_min": self.temp_min,
            "temp_annealing": self.temp_annealing,
            "total_epochs": self.total_epochs,
        }
        self.sampler_config = snas_config  # type: ignore


class DRNASProfile(BaseProfile, ABC):
    def __init__(
        self,
        epochs: int,
        **kwargs: Any,
    ) -> None:
        PROFILE_TYPE = "DRNAS"

        super().__init__(  # type: ignore
            PROFILE_TYPE,
            epochs,
            **kwargs,
        )
        self.sampler_type = str.lower(PROFILE_TYPE)

    def _initialize_sampler_config(self) -> None:
        drnas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "arch_combine_fn": self.sampler_arch_combine_fn,
        }
        self.sampler_config = drnas_config  # type: ignore


class DiscreteProfile:
    def __init__(self, **kwargs) -> None:  # type: ignore
        self._initialize_trainer_config()
        self._initializa_genotype()
        self.configure_trainer(**kwargs)

    def get_trainer_config(self) -> dict:
        return self.train_config

    def get_genotype(self) -> str:
        return self.genotype

    def _initialize_trainer_config(self) -> None:
        default_train_config = {
            "lr": 0.025,
            "epochs": 100,
            "optim": "sgd",
            "scheduler": "cosine_annealing_lr",
            "optim_config": {
                "momentum": 0.9,
                "nesterov": 0,
                "weight_decay": 3e-4,
            },
            "criterion": "cross_entropy",
            "batch_size": 96,
            "learning_rate_min": 0.0,
            "channel": 36,
            "print_freq": 2,
            "drop_path_prob": 0.2,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.7,
            "use_ddp": True,
            "checkpointing_freq": 2,
        }
        self.train_config = default_train_config

    def _initializa_genotype(self) -> None:
        self.genotype = str(
            Genotype(
                normal=[
                    ("sep_conv_3x3", 1),
                    ("sep_conv_3x3", 0),
                    ("skip_connect", 0),
                    ("sep_conv_3x3", 1),
                    ("skip_connect", 0),
                    ("sep_conv_3x3", 1),
                    ("sep_conv_3x3", 0),
                    ("skip_connect", 2),
                ],
                normal_concat=[2, 3, 4, 5],
                reduce=[
                    ("max_pool_3x3", 0),
                    ("max_pool_3x3", 1),
                    ("skip_connect", 2),
                    ("max_pool_3x3", 0),
                    ("max_pool_3x3", 0),
                    ("skip_connect", 2),
                    ("skip_connect", 2),
                    ("avg_pool_3x3", 0),
                ],
                reduce_concat=[2, 3, 4, 5],
            )
        )

    def configure_trainer(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.train_config
            ), f"{config_key} not a valid configuration for training a \
            discrete architecture"
            self.train_config[config_key] = kwargs[config_key]

    def set_search_space_config(self, config: dict) -> None:
        self.searchspace_config = config

    def get_searchspace_config(self, searchspace_str: str, dataset_str: str) -> dict:
        if hasattr(self, "searchspace_config"):
            return self.searchspace_config
        if searchspace_str == "nb201":
            searchspace_config = {
                "N": 5,  # num_cells
                "C": 16,  # channels
            }
        elif searchspace_str == "darts":
            searchspace_config = {
                "C": 36,  # init channels
                "layers": 20,  # number of layers
                "auxiliary": False,
            }
        else:
            raise ValueError("search space is not correct")
        searchspace_config["num_classes"] = get_num_classes(dataset_str)
        return searchspace_config
