from __future__ import annotations

from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from confopt.profiles.profile_config import BaseProfile
from confopt.train.experiment import SamplerType, SearchSpaceType


class OneShotOptimizerSampler:
    def __init__(self, search_space: SearchSpaceType):
        self.search_space = search_space
        self.CONFIG_SUFFIX = "_config"
        self.init_config_space()

    def init_config_space(self) -> ConfigurationSpace:
        """Initializes the configuration space for the one-shot optimizer.

        Returns:
            ConfigurationSpace: Configuration space for the one-shot optimizer
        """
        ##### Sampler Space #####
        self.sampler_space = ConfigurationSpace(
            name="sampler",
            space={
                "sampler": Categorical(
                    "sampler_type", ["darts", "drnas", "gdas", "reinmax"]
                ),
            },
        )

        gdas_cs = ConfigurationSpace(
            name="gdas_space",
            space={
                "tau_min": Float("tau_min", bounds=(0.1, 2)),
                "tau_max": Float("tau_max", bounds=(5, 15)),
            },
        )

        self.sampler_space.add_configuration_space(
            prefix=f"gdas{self.CONFIG_SUFFIX}",
            configuration_space=gdas_cs,
            parent_hyperparameter={
                "parent": self.sampler_space["sampler_type"],
                "value": "gdas",
            },
        )

        self.sampler_space.add_configuration_space(
            prefix="reinmax_config",
            configuration_space=gdas_cs,
            parent_hyperparameter={
                "parent": self.sampler_space["sampler_type"],
                "value": "reinmax",
            },
        )

        ##### LoRA Space #####
        self.lora_space = ConfigurationSpace(
            name="lora",
            space={
                "use_lora": Categorical("use_lora", [True, False]),
            },
        )

        lora_cs = ConfigurationSpace(
            name="lora_space",
            space={
                "r": Categorical("r", [1, 2, 4, 8]),
                "lora_alpha": Integer("lora_alpha", bounds=(1, 8)),
                "lora_dropout": Float("lora_dropout", bounds=(0.0, 1.0)),
                "lora_warm_epochs": Integer("lora_warm_epochs", bounds=(5, 25)),
            },
        )

        self.lora_space.add_configuration_space(
            prefix=f"lora{self.CONFIG_SUFFIX}",
            configuration_space=lora_cs,
            parent_hyperparameter={
                "parent": self.lora_space["use_lora"],
                "value": True,
            },
        )

        ##### Perturbation Space #####
        self.perturbation_space = ConfigurationSpace(
            name="perturbation",
            space={
                "use_perturbation": Categorical("use_perturbation", [True, False]),
            },
        )

        perturbation_cs = ConfigurationSpace(
            name="perturbation_space",
            space={
                "perturbation": Categorical("perturbation", ["random", "adversarial"]),
                "epsilon": Float("epsilon", bounds=(0.05, 1.0)),
            },
        )

        adversarial_cs = ConfigurationSpace(
            name="adversarial_space",
            space={
                "steps": Integer("steps", bounds=(1, 50)),
                "random_start": Categorical("random_start", [True, False]),
            },
        )

        perturbation_cs.add_configuration_space(
            prefix=f"adversarial{self.CONFIG_SUFFIX}",
            configuration_space=adversarial_cs,
            parent_hyperparameter={
                "parent": perturbation_cs["perturbation"],
                "value": "adversarial",
            },
        )

        self.perturbation_space.add_configuration_space(
            prefix="perturbation_config",
            configuration_space=perturbation_cs,
            parent_hyperparameter={
                "parent": self.perturbation_space["use_perturbation"],
                "value": True,
            },
        )

        ##### Partial Connection Space #####
        self.partial_connection_space = ConfigurationSpace(
            name="partial_connection",
            space={
                "use_partial_connection": Categorical(
                    "use_partial_connection", [True, False]
                ),
            },
        )

        partial_connection_cs = ConfigurationSpace(
            name="partial_connection_space",
            space={
                "k": Categorical("k", [1, 2, 4, 8, 16]),
            },
        )

        self.partial_connection_space.add_configuration_space(
            prefix=f"partial_connection{self.CONFIG_SUFFIX}",
            configuration_space=partial_connection_cs,
            parent_hyperparameter={
                "parent": self.partial_connection_space["use_partial_connection"],
                "value": True,
            },
        )

        ##### Prune Space #####
        self.prune_space = ConfigurationSpace(
            name="prune",
            space={
                "use_prune": Categorical("use_prune", [True, False]),
            },
        )

        prune_cs = ConfigurationSpace(
            name="prune_space",
            space={
                "n_prune": Integer("n_prune", bounds=(1, 5)),
                "prune_interval": Categorical("prune_interval", [5, 10]),
            },
        )

        self.prune_space.add_configuration_space(
            prefix=f"prune{self.CONFIG_SUFFIX}",
            configuration_space=prune_cs,
            parent_hyperparameter={
                "parent": self.prune_space["use_prune"],
                "value": True,
            },
        )

        ##### Arch Attention Space #####
        self.arch_attention_space = ConfigurationSpace(
            name="arch_attention",
            space={
                "use_arch_attention": Categorical("use_arch_attention", [True, False]),
            },
        )

        #### Arch Params Combine Function Space ####
        self.sampler_arch_combine_fn = ConfigurationSpace(
            name="sampler_arch_combine_fn",
            space={
                "sampler_arch_combine_fn": Categorical(
                    "sampler_arch_combine_fn", ["softmax", "sigmoid"]
                ),
            },
        )

    def sample_sampler(
        self, sampler: SamplerType | None, sample_sampler_config: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Samples the sampler configuration.

        Args:
            sampler: Sampler to be used. Sampled if None.
            sample_sampler_config: Whether to sample the hyperparameters of the
                optimizer. If False, uses the default values.

        Returns:
            dict[str, Any]: Sampler configuration
        """
        if sampler is not None and sample_sampler_config is False:  # Nothing to sample
            return {"sampler_type": sampler.value}, {}

        def split_config(
            config: dict[str, Any]
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            base_params = {
                k: v for k, v in config.items() if self.CONFIG_SUFFIX + ":" not in k
            }
            config_params = {
                k.split(":")[-1]: v
                for k, v in config.items()
                if self.CONFIG_SUFFIX + ":" in k
            }

            return base_params, config_params

        # Sampler is given, but have to sample its configuration
        if sampler is not None:
            while True:
                config = self.sampler_space.sample_configuration(1)
                if config["sampler_type"] == sampler.value:
                    break
        else:  # Sample the sampler and its configuration
            config = self.sampler_space.sample_configuration(1)

        base_config, extra_config = split_config(config)
        extra_config = extra_config if sample_sampler_config else {}

        return base_config, extra_config

    def sample_lora(
        self, lora: bool | None, sample_lora_config: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Samples the LoRA configuration.

        Args:
            lora: Whether to use LoRA. Sampled if None.
            sample_lora_config: Whether to sample the hyperparameters of the lora.
                If False, uses the default values.

        Returns:
            dict[str, Any]: LoRA configuration
        """
        if lora is False:
            return {"lora_rank": 0}, {}

        if lora is True:
            while True:
                config = self.lora_space.sample_configuration(1)
                if config["use_lora"] is True:
                    break
        elif lora is None:
            config = self.lora_space.sample_configuration(1)

        base_config = {
            "lora_rank": config["lora_config:r"] if "lora_config:r" in config else 0
        }

        extra_keys = [
            "lora_config:lora_warm_epochs",
            "lora_config:lora_alpha",
            "lora_config:lora_dropout",
            # "lora_config:lora_toggle_probability", # TODO-AK: Add this to the config?
        ]
        extra_config = (
            {k.split(":")[-1]: config[k] for k in extra_keys if k in config}
            if sample_lora_config
            else {}
        )

        return base_config, extra_config

    def sample_perturbation(
        self, perturbation: bool | None, sample_perturbation_config: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Samples the perturbation configuration.

        Args:
            perturbation: Whether to use perturbation. Sampled if None.
            sample_perturbation_config: Whether to sample the hyperparameters of the
                perturbation. If False, uses the default values.

        Returns:
            dict[str, Any]: perturbation configuration
        """
        if perturbation is False:
            return {"perturbation": None}, {}

        if perturbation is True:
            while True:
                config = self.perturbation_space.sample_configuration(1)
                if config["use_perturbation"] is True:
                    break
        elif perturbation is None:
            config = self.perturbation_space.sample_configuration(1)

        if bool(config["use_perturbation"]) is True:
            base_config = {"perturbation": config["perturbation_config:perturbation"]}
        else:
            base_config = {"perturbation": None}

        extra_config = {
            k.split(":")[-1]: v
            for k, v in config.items()
            if self.CONFIG_SUFFIX + ":" in k
        }
        extra_config.pop("perturbation", None)
        extra_config = extra_config if sample_perturbation_config else {}

        return base_config, extra_config

    def sample_partial_connection(
        self, partial_connection: bool | None, sample_partial_connection_config: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Samples the partial connection configuration.

        Args:
            partial_connection: Whether to use partial connection. Sampled if None.
            sample_partial_connection_config: Whether to sample the hyperparameters of
                the partial connection. If False, uses the default values.

        Returns:
            dict[str, Any]: Partial connection configuration
        """
        if partial_connection is False:
            return {"is_partial_connection": False}, {}

        if partial_connection is True:
            while True:
                config = self.partial_connection_space.sample_configuration(1)
                if bool(config["use_partial_connection"]) is True:
                    break
        elif partial_connection is None:
            config = self.partial_connection_space.sample_configuration(1)

        base_config = {"is_partial_connection": config["use_partial_connection"]}
        extra_config = {"k": config["k"]} if "k" in config else {}
        extra_config = {} if sample_partial_connection_config else {}

        return base_config, extra_config

    def sample_prune(
        self, prune: bool | None, sample_prune_config: bool, epochs: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Samples the prune configuration.

        Args:
            prune: Whether to use prune. Sampled if None.
            sample_prune_config: Whether to sample the hyperparameters of the
                prune. If False, uses the default values.
            epochs: Number of epochs to train the supernet.

        Returns:
            dict[str, Any]: Prune configuration
        """
        if prune is False:
            return {}, {}

        if prune is True:
            while True:
                config = self.prune_space.sample_configuration(1)
                if config["use_prune"] is True:
                    break
        elif prune is None:
            config = self.prune_space.sample_configuration(1)

        _ = epochs
        _ = sample_prune_config

        return (
            {},
            {},
        )  # TODO-AK: Incomplete. Deal with pruning when the API is finalized.

    def sample_arch_attention(
        self, arch_attention: bool | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Samples the architecture attention configuration.

        Args:
            arch_attention: Whether to use attention between edges for arch parameters.
                Sampled if None.

        Returns:
            dict[str, Any]: Architecture attention configuration
        """
        if arch_attention is None:
            config = self.arch_attention_space.sample_configuration(1)
            return {"is_arch_attention_enabled": config["use_arch_attention"]}, {}

        return {"is_arch_attention_enabled": arch_attention}, {}

    def sample_arch_params_combine_fn(
        self, sampler_arch_combine_fn: str | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Samples the architecture parameters combine function configuration.

        Args:
            sampler_arch_combine_fn: Post processing function for the arch parameters.
                Sampled if None.

        Returns:
            dict[str, Any]: Architecture parameters combine function configuration
        """
        if sampler_arch_combine_fn is None:
            config = self.sampler_arch_combine_fn.sample_configuration(1)
            return {"sampler_arch_combine_fn": config["sampler_arch_combine_fn"]}, {}

        return {"sampler_arch_combine_fn": sampler_arch_combine_fn}, {}

    def sample_entangle_op_weights(
        self, entangle_op_weights: bool | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Samples the weight entanglement configuration.

        Args:
            entangle_op_weights: Whether to use weight entanglement. Sampled if None.

        Returns:
            dict[str, Any]: Weight entanglement configuration
        """
        if entangle_op_weights is None:
            return {"entangle_op_weights": True}, {}

        return {"entangle_op_weights": entangle_op_weights}, {}

    def sample(
        self,
        epochs: int,
        *,
        sampler: SamplerType | None = None,
        lora: bool | None = None,
        perturbation: bool | None = None,
        partial_connection: bool | None = True,
        prune: bool | None = None,
        arch_attention: bool | None = None,
        arch_params_combine_fn: str | None = None,
        entangle_op_weights: bool | None = None,
        sample_sampler_config: bool = True,
        sample_lora_config: bool = True,
        sample_perturbation_config: bool = True,
        sample_partial_connection_config: bool = True,
        sample_prune_config: bool = True,
    ) -> BaseProfile:
        """Samples a new optimizer profile.

        Key items to sample:
            - sampler
            - lora
            - perturbation
            - partial_connection
            - prune
            - attention between edges for arch parameters
            - post processing function for the arch parameters (softmax or sigmoid)
            - weight entanglement

        All parameters are sampled by default, indicated by None. Additionally,
        this method gives the user to override the default sampling behavior by
        specifying the values for the parameters. E.g., if user wants to use a
        specific optimizer, they can pass the optimizer parameter and set the
        remaining parameters to None.

        Args:
            epochs: Number of epochs to train the supernet.
            sampler: Sampler to be used. Sampled if None.
            lora: Whether to use LoRA. Sampled if None.
            perturbation: Whether to use perturbation. Sampled if None.
            partial_connection: Whether to use partial connection. Sampled if None.
            prune: Whether to use prune. Sampled if None.
            arch_attention: Whether to use attention between edges for arch parameters.
                Sampled if None.
            arch_params_combine_fn: Post processing function for the arch parameters.
                Sampled if None.
            entangle_op_weights: Whether to use weight entanglement. Sampled if None.
            sample_sampler_config: Whether to sample the hyperparameters of the
                optimizer. If False, uses the default values.
            sample_lora_config: Whether to sample the hyperparameters of the lora.
                If False, uses the default values.
            sample_perturbation_config: Whether to sample the hyperparameters of the
                perturbation. If False, uses the default values.
            sample_partial_connection_config: Whether to sample the hyperparameters of
                the partial connection. If False, uses the default values.
            sample_prune_config: Whether to sample the hyperparameters of the prune.
                If False, uses the default values.

        Returns:
            BaseProfile: Sampled optimizer profile
        """
        full_config = {}

        base_config, extra_sampler_config = self.sample_sampler(
            sampler, sample_sampler_config
        )
        full_config.update(base_config)

        base_config, extra_lora_config = self.sample_lora(lora, sample_lora_config)
        full_config.update(base_config)

        base_config, extra_perturbation_config = self.sample_perturbation(
            perturbation, sample_perturbation_config
        )
        full_config.update(base_config)

        base_config, extra_partial_connection_config = self.sample_partial_connection(
            partial_connection, sample_partial_connection_config
        )
        full_config.update(base_config)

        base_config, extra_prune_config = self.sample_prune(
            prune, sample_prune_config, epochs
        )
        full_config.update(base_config)

        base_config, extra_arch_attention_config = self.sample_arch_attention(
            arch_attention
        )
        full_config.update(base_config)

        (
            base_config,
            extra_arch_params_combine_fn_config,
        ) = self.sample_arch_params_combine_fn(arch_params_combine_fn)
        full_config.update(base_config)

        base_config, extra_entangle_op_weights_config = self.sample_entangle_op_weights(
            entangle_op_weights
        )
        full_config.update(base_config)

        profile = BaseProfile(epochs=epochs, **full_config)

        ###### TODO-AK: Fix. Super ugly. ######
        sampler_config = {
            "sample_frequency": None,
            "arch_combine_fn": None,
        }
        if full_config["sampler_type"] in ["gdas", "reinmax"]:
            sampler_config.update(
                {
                    "tau_min": None,
                    "tau_max": None,
                }
            )
        profile.sampler_config = sampler_config
        ###### END TODO ######

        profile.configure_sampler(**extra_sampler_config)
        profile.configure_lora(**extra_lora_config)

        if full_config["perturbation"] is True:
            profile.configure_perturbator(**extra_perturbation_config)

        if full_config["is_partial_connection"] is True:
            profile.configure_partial_connector(**extra_partial_connection_config)

        return profile


if __name__ == "__main__":
    optimizer_sampler = OneShotOptimizerSampler(SearchSpaceType.DARTS)

    for _ in range(100):
        try:
            print("*" * 10)
            profile = optimizer_sampler.sample(epochs=100)
            print("SUCCEEDED")
        except Exception as e:  # noqa: BLE001
            print("FAILED")
            print(e)
