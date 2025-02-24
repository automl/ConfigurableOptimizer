from __future__ import annotations

from typing import Any, Literal
import warnings

import torch

from confopt.enums import SamplerType, SearchSpaceType

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO Change this to real data
ADVERSARIAL_DATA = (
    torch.randn(2, 3, 32, 32).to(DEVICE),
    torch.randint(0, 9, (2,)).to(DEVICE),
)


class BaseProfile:
    def __init__(
        self,
        sampler_type: str | SamplerType,
        searchspace: str | SearchSpaceType,
        epochs: int = 50,
        *,
        sampler_sample_frequency: str = "step",
        is_partial_connection: bool = False,
        dropout: float | None = None,
        perturbation: str | None = None,
        perturbator_sample_frequency: str = "epoch",
        sampler_arch_combine_fn: str = "default",
        partial_connector_config: dict | None = None,
        perturbator_config: dict | None = None,
        entangle_op_weights: bool = False,
        lora_rank: int = 0,
        lora_warm_epochs: int = 0,
        lora_toggle_epochs: list[int] | None = None,
        lora_toggle_probability: float | None = None,
        seed: int = 100,
        oles: bool = False,
        calc_gm_score: bool = False,
        prune_epochs: list[int] | None = None,
        prune_fractions: list[float] | None = None,
        is_arch_attention_enabled: bool = False,
        is_regularization_enabled: bool = False,
        regularization_config: dict | None = None,
        pt_select_architecture: bool = False,
        searchspace_domain: str | None = None,
        use_auxiliary_skip_connection: bool = False,
        searchspace_subspace: str | None = None,
    ) -> None:
        self.searchspace_type = (
            SearchSpaceType(searchspace)
            if isinstance(searchspace, str)
            else searchspace
        )
        self.sampler_type = (
            SamplerType(sampler_type) if isinstance(sampler_type, str) else sampler_type
        )

        assert isinstance(
            self.searchspace_type, SearchSpaceType
        ), f"Illegal value {self.searchspace_type} for searchspace_type"
        assert isinstance(
            self.sampler_type, SamplerType
        ), f"Illegal value {self.sampler_type} for sampler_type"

        if self.searchspace_type == SearchSpaceType.TNB101:
            assert searchspace_domain in [
                "class_object",
                "class_scene",
            ], "searchspace_domain must be either class_object or class_scene"
        else:
            assert (
                searchspace_domain is None
            ), "searchspace_domain is not required for this searchspace"
        if searchspace == "nb1shot1" or searchspace == SearchSpaceType.NB1SHOT1:
            assert searchspace_subspace in [
                "S1",
                "S2",
                "S3",
            ], "searchspace subspace must be S1, S2 or S3"
        else:
            assert (
                searchspace_subspace is None
            ), "searchspace_subspace is not required for this searchspace"

        self.searchspace_domain = searchspace_domain
        self.epochs = epochs
        self.sampler_sample_frequency = (
            sampler_sample_frequency  # TODO-ICLR: Remove this
        )
        self.lora_warm_epochs = lora_warm_epochs
        self.seed = seed
        self.sampler_arch_combine_fn = sampler_arch_combine_fn
        self._initialize_trainer_config()
        self._initialize_sampler_config()
        self._set_partial_connector(is_partial_connection)
        self._set_lora_configs(
            lora_rank,
            lora_warm_epochs,
            toggle_epochs=lora_toggle_epochs,
            lora_toggle_probability=lora_toggle_probability,
        )
        self._set_dropout(dropout)
        self._set_perturb(perturbation, perturbator_sample_frequency)
        self.entangle_op_weights = entangle_op_weights
        self._set_oles_configs(oles, calc_gm_score)
        self._set_pruner_configs(prune_epochs, prune_fractions)
        self._set_pt_select_configs(pt_select_architecture)
        self.is_arch_attention_enabled = is_arch_attention_enabled
        self._set_regularization(is_regularization_enabled)

        if regularization_config is not None:
            self.configure_regularization(**regularization_config)

        if partial_connector_config is not None:
            self.configure_partial_connector(**partial_connector_config)

        if perturbator_config is not None:
            self.configure_perturbator(**perturbator_config)

        self.use_auxiliary_skip_connection = use_auxiliary_skip_connection
        if searchspace_subspace is not None:
            # TODO: why can't I directly have the dict as an argument?
            searchspace_subspace_dict = {"search_space": searchspace_subspace}
            self.configure_searchspace(**searchspace_subspace_dict)

    def _set_pt_select_configs(
        self,
        pt_select_architecture: bool = False,
        pt_projection_criteria: Literal["acc", "loss"] = "acc",
        pt_projection_interval: int = 10,
    ) -> None:
        if pt_select_architecture:
            self.pt_select_configs = {
                "projection_interval": pt_projection_interval,
                "projection_criteria": pt_projection_criteria,
            }
        else:
            self.pt_select_configs = None  # type: ignore

    def _set_pruner_configs(
        self,
        prune_epochs: list[int] | None = None,
        prune_fractions: list[float] | None = None,
    ) -> None:
        if prune_epochs is not None:
            assert (
                prune_fractions is not None
            ), "Please provide epochs prune-fractions to prune with"
            assert len(prune_fractions) == len(
                prune_epochs
            ), "Length of both prune_epochs and prune_fractions must be same"
            self.pruner_config = {
                "prune_epochs": prune_epochs,
                "prune_fractions": prune_fractions,
            }

    def _set_lora_configs(
        self,
        lora_rank: int = 0,
        lora_warm_epochs: int = 0,
        lora_dropout: float = 0,
        lora_alpha: int = 1,
        lora_toggle_probability: float | None = None,
        merge_weights: bool = True,
        toggle_epochs: list[int] | None = None,
    ) -> None:
        self.lora_config = {
            "r": lora_rank,
            "lora_dropout": lora_dropout,
            "lora_alpha": lora_alpha,
            "merge_weights": merge_weights,
        }
        self.lora_toggle_epochs = toggle_epochs
        self.lora_warm_epochs = lora_warm_epochs
        self.lora_toggle_probability = lora_toggle_probability

    def _set_oles_configs(
        self,
        oles: bool = False,
        calc_gm_score: bool = False,
    ) -> None:
        if oles and not calc_gm_score:
            calc_gm_score = True
            warnings.warn(
                "OLES needs gm_score, please pass calc_gm_score as True.", stacklevel=2
            )
        self.oles_config = {
            "oles": oles,
            "calc_gm_score": calc_gm_score,
            "frequency": 20,
            "threshold": 0.4,
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

    def _set_regularization(self, is_regularization_enabled: bool = False) -> None:
        self.is_regularization_enabled = is_regularization_enabled
        self._initialize_regularization_config()

    def get_config(self) -> dict:
        assert (
            self.sampler_config is not None
        ), "atleast a sampler is needed to initialize the search space"
        weight_type = (
            "weight_entanglement" if self.entangle_op_weights else "weight_sharing"
        )
        config = {
            "sampler": self.sampler_config,
            "perturbator": self.perturb_config,
            "partial_connector": self.partial_connector_config,
            "dropout": self.dropout_config,
            "trainer": self.trainer_config,
            "lora": self.lora_config,
            "lora_extra": {
                "toggle_epochs": self.lora_toggle_epochs,
                "warm_epochs": self.lora_warm_epochs,
                "toggle_probability": self.lora_toggle_probability,
            },
            "sampler_type": self.sampler_type,
            "searchspace": self.searchspace_type.value,
            "searchspace_domain": self.searchspace_domain,
            "weight_type": weight_type,
            "oles": self.oles_config,
            "pt_selection": self.pt_select_configs,
            "is_arch_attention_enabled": self.is_arch_attention_enabled,
            "regularization": self.regularization_config,
            "use_auxiliary_skip_connection": self.use_auxiliary_skip_connection,
        }

        if hasattr(self, "pruner_config"):
            config.update({"pruner": self.pruner_config})

        if hasattr(self, "searchspace_config") and self.searchspace_config is not None:
            config.update({"search_space": self.searchspace_config})

        if hasattr(self, "extra_config") and self.extra_config is not None:
            config.update(self.extra_config)
        return config

    def _initialize_sampler_config(self) -> None:
        self.sampler_config = None

    def _initialize_perturbation_config(self) -> None:
        if self.perturb_type == "adverserial":
            perturb_config = {
                "epsilon": 0.3,
                "data": ADVERSARIAL_DATA,
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

    def _initialize_partial_connector_config(self) -> None:
        if self.is_partial_connection:
            partial_connector_config = {"k": 4, "num_warm_epoch": 15}
            self.configure_searchspace(**partial_connector_config)
        else:
            partial_connector_config = None
        self.partial_connector_config = partial_connector_config

    def _initialize_trainer_config(self) -> None:
        if self.searchspace_type == SearchSpaceType.NB201:
            self._initialize_trainer_config_nb201()
        elif self.searchspace_type == SearchSpaceType.DARTS:
            self._initialize_trainer_config_darts()
        elif self.searchspace_type == SearchSpaceType.NB1SHOT1:
            self._initialize_trainer_config_1shot1()
        elif self.searchspace_type == SearchSpaceType.TNB101:
            self._initialize_trainer_config_tnb101()

    def _initialize_dropout_config(self) -> None:
        dropout_config = {
            "p": self.dropout if self.dropout is not None else 0.0,
            "p_min": 0.0,
            "anneal_frequency": "epoch",
            "anneal_type": "linear",
            "max_iter": self.epochs,
        }
        self.dropout_config = dropout_config

    def _initialize_regularization_config(self) -> None:
        regularization_config = {
            "reg_weights": [0.0],
            "loss_weight": 1.0,
            "active_reg_terms": [],
            "drnas_config": {"reg_scale": 1e-3},
            "flops_config": {},
            "fairdarts_config": {},
        }
        self.regularization_config = regularization_config

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        assert self.sampler_config is not None
        for config_key in kwargs:
            assert (
                config_key in self.sampler_config  # type: ignore
            ), f"{config_key} not a valid configuration for the sampler of type \
                {self.sampler_type}"
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

        if kwargs.get("k"):
            self.configure_searchspace(k=kwargs["k"])

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

    def configure_lora(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.lora_config
            ), f"{config_key} not a valid configuration for the lora layers"
            self.lora_config[config_key] = kwargs[config_key]

    def configure_oles(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.oles_config
            ), f"{config_key} not a valid configuration for the oles config"
            self.oles_config[config_key] = kwargs[config_key]

    def configure_regularization(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.regularization_config
            ), f"{config_key} not a valid configuration for the regularization config"

            if isinstance(self.regularization_config[config_key], dict):
                assert set(kwargs[config_key].keys()).issubset(
                    self.regularization_config[config_key].keys()  # type: ignore
                ), f"Invalid keys for the regularization config '{config_key}'"

            self.regularization_config[config_key] = kwargs[config_key]

    def configure_pt_selection(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert config_key in self.pt_select_configs, (
                f"{config_key} not a valid configuration for the"
                + "perturbation based selection config"
            )
            self.pt_select_configs[config_key] = kwargs[config_key]

    def configure_searchspace(self, **config: Any) -> None:
        if not hasattr(self, "searchspace_config"):
            self.searchspace_config = config
        else:
            self.searchspace_config.update(config)

    def configure_extra(self, **config) -> None:  # type: ignore
        self.extra_config = config

    def get_run_description(self) -> str:
        run_configs = []
        run_configs.append(f"ss_{self.searchspace_type}")
        if self.entangle_op_weights:
            run_configs.append("type_we")
        else:
            run_configs.append("type_ws")
        run_configs.append(f"opt_{self.sampler_type}")
        if self.lora_warm_epochs > 0:
            run_configs.append(f"lorarank_{self.lora_config.get('r')}")
            run_configs.append(f"lorawarmup_{self.lora_warm_epochs}")
        run_configs.append(f"epochs_{self.trainer_config.get('epochs')}")
        run_configs.append(f"seed_{self.seed}")
        if self.oles_config.get("oles"):
            run_configs.append("with_oles")
        return "-".join(run_configs)

    def _initialize_trainer_config_nb201(self) -> None:
        trainer_config = {
            "lr": 0.025,
            "arch_lr": 3e-4,
            "epochs": self.epochs,  # 200
            "lora_warm_epochs": self.lora_warm_epochs,
            "optim": "sgd",
            "arch_optim": "adam",
            "optim_config": {
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 5e-4,
            },
            "arch_optim_config": {
                "weight_decay": 1e-3,
            },
            "scheduler": "cosine_annealing_lr",
            "scheduler_config": {},
            "criterion": "cross_entropy",
            "batch_size": 64,
            "learning_rate_min": 0.0,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.5,
            "use_data_parallel": False,
            "checkpointing_freq": 1,
            "seed": self.seed,
        }

        self.trainer_config = trainer_config

    def _initialize_trainer_config_darts(self) -> None:
        trainer_config = {
            "lr": 0.025,
            "arch_lr": 3e-4,
            "epochs": self.epochs,  # 50
            "lora_warm_epochs": self.lora_warm_epochs,
            "optim": "sgd",
            "arch_optim": "adam",
            "optim_config": {
                "momentum": 0.9,
                "nesterov": False,
                "weight_decay": 3e-4,
            },
            "arch_optim_config": {
                "weight_decay": 1e-3,
                "betas": (0.5, 0.999),
            },
            "scheduler": "cosine_annealing_lr",
            "scheduler_config": {},
            "criterion": "cross_entropy",
            "batch_size": 64,
            "learning_rate_min": 0.001,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.5,
            "use_data_parallel": False,
            "checkpointing_freq": 1,
            "seed": self.seed,
        }

        self.trainer_config = trainer_config

    def _initialize_trainer_config_1shot1(self) -> None:
        trainer_config = {
            "lr": 0.025,
            "arch_lr": 3e-4,
            "epochs": self.epochs,  # 50
            "lora_warm_epochs": self.lora_warm_epochs,
            "optim": "sgd",
            "arch_optim": "adam",
            "use_data_parallel": False,
            "seed": self.seed,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.5,
            "optim_config": {
                "momentum": 0.9,
                "nesterov": False,
                "weight_decay": 3e-4,
            },
            "arch_optim_config": {
                "weight_decay": 1e-3,
                "betas": (0.5, 0.999),
            },
            "scheduler": "cosine_annealing_lr",
            "learning_rate_min": 0.001,
            "criterion": "cross_entropy",
            "batch_size": 64,
            "checkpointing_freq": 3,
            "drop_path_prob": 0.2,
        }
        self.trainer_config = trainer_config

    def _initialize_trainer_config_tnb101(self) -> None:
        trainer_config = {
            "lr": 0.025,
            "arch_lr": 3e-4,
            "epochs": self.epochs,  # 50
            "lora_warm_epochs": self.lora_warm_epochs,
            "optim": "sgd",
            "arch_optim": "adam",
            "optim_config": {
                "momentum": 0.9,
                "nesterov": False,
                "weight_decay": 3e-4,
            },
            "arch_optim_config": {
                "weight_decay": 1e-3,
                "betas": (0.5, 0.999),
            },
            "scheduler": "cosine_annealing_lr",
            "scheduler_config": {},
            "criterion": "cross_entropy",
            "use_data_parallel": False,
            "checkpointing_freq": 1,
            "seed": self.seed,
            "cutout": -1,
            "cutout_length": 16,
            "batch_size": 32,
            "train_portion": 0.5,
            "learning_rate_min": 0.001,
        }

        self.trainer_config = trainer_config
