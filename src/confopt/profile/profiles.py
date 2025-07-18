from __future__ import annotations

from abc import ABC
from dataclasses import asdict
from typing import Any

from typing_extensions import override

from confopt.enums import SamplerType, TrainerPresetType
from confopt.searchspace.common import LambdaReg
from confopt.searchspace.darts.core.genotypes import DARTSGenotype
from confopt.utils import get_num_classes

from .base import BaseProfile


class DARTSProfile(BaseProfile, ABC):
    """Train Profile class for DARTS (Differentiable Architecture Search).
    It inherits from the BaseProfile class and sets the sampler config for DARTS
    sampler.

    Methods:
        __init__: Initializes the DARTSProfile object.
        _initialize_sampler_config: Initializes the sampler configuration for DARTS.
    """

    SAMPLER_TYPE = SamplerType.DARTS

    def __init__(
        self,
        trainer_preset: str | TrainerPresetType,
        epochs: int,
        use_lambda_regularizer: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            self.SAMPLER_TYPE,
            trainer_preset,
            epochs,
            **kwargs,
        )
        self._set_lambda_regularizer(use_lambda_regularizer)

    def _set_lambda_regularizer(self, use_lambda_regularizer: bool = False) -> None:
        self.use_lambda_regularizer = use_lambda_regularizer
        self.lambda_regularizer_config = (
            asdict(LambdaReg()) if use_lambda_regularizer else None
        )

    def get_config(self) -> dict:
        config = super().get_config()
        if self.lambda_regularizer_config:
            config["lambda_regularizer"] = self.lambda_regularizer_config

        return config

    def configure_lambda_regularizer(self, **kwargs: Any) -> None:
        assert self.use_lambda_regularizer is True
        assert self.lambda_regularizer_config is not None

        for config_key in kwargs:
            assert config_key in self.lambda_regularizer_config, (
                f"{config_key} not a valid configuration for the"
                + "lambda regularization config"
            )
            self.lambda_regularizer_config[config_key] = kwargs[config_key]

    def _initialize_sampler_config(self) -> None:
        """Initializes the sampler configuration for DARTS.

        The sampler configuration includes the sample frequency and the architecture
        combine function.

        Args:
            None

        Returns:
            None
        """
        darts_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "arch_combine_fn": self.sampler_arch_combine_fn,
        }
        self.sampler_config = darts_config  # type: ignore

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        """Configures the sampler settings based on provided keyword arguments.

        The configuration options are passed as keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:
                sample_frequency (str): The rate at which samples should be taken
                arch_combine_fn (str): The function used to combine architectures \
                when FairDARTS is used, this should be 'sigmoid'. Default value is
                'default'.

        Example:
            >>> from confopt.profile import DARTSProfile
            >>> darts_profile = DARTSProfile(trainer_preset='darts', epochs=50)
            >>> darts_profile.configure_sampler(arch_combine_fn='sigmoid')

        The accepted keyword arguments should align with the sampler's configuration and
        the attributes can be configured dynamically.

        Raises:
            ValueError: If an unrecognized keyword is passed in kwargs.


        Returns:
            None
        """
        super().configure_sampler(**kwargs)


class GDASProfile(BaseProfile, ABC):
    """Train Profile class for GDAS
    (Gradient-based search using Differentiable Architecture Sampler)
    It inherits from the BaseProfile class and sets the sampler config for
    DARTS sampler.

    Attributes:
        tau_min (float): Minimum temperature for sampling.
        tau_max (float): Maximum temperature for sampling.
    """

    SAMPLER_TYPE = SamplerType.GDAS

    def __init__(
        self,
        trainer_preset: str | TrainerPresetType,
        epochs: int,
        tau_min: float = 0.1,
        tau_max: float = 10,
        **kwargs: Any,
    ) -> None:
        self.tau_min = tau_min
        self.tau_max = tau_max

        super().__init__(
            self.SAMPLER_TYPE,
            trainer_preset,
            epochs,
            **kwargs,
        )

    def _initialize_sampler_config(self) -> None:
        gdas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "arch_combine_fn": self.sampler_arch_combine_fn,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
        }
        self.sampler_config = gdas_config  # type: ignore

    def _initialize_trainer_config_nb201(self) -> None:
        # self.epochs = 250
        super()._initialize_trainer_config_nb201()
        self.trainer_config.update(
            {
                "batch_size": 64,
                "epochs": self.epochs,
            }
        )
        self.trainer_config.update({"learning_rate_min": 0.001})

    def _initialize_trainer_config_darts(self) -> None:
        super()._initialize_trainer_config_darts()

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        """Configures the sampler settings based on provided keyword arguments.

        The configuration options are passed as keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                sample_frequency (str): The rate at which samples should be taken

                arch_combine_fn (str): The function used to combine architectures \
                (str) when FairDARTS is used, this should be 'sigmoid'. Default value \
                is 'default'.

                tau_min (float): Minimum temperature for sampling.

                tau_max (float): Maximum temperature for sampling.


        Example:
            >>> from confopt.profile import GDASProfile
            >>> gdas_profile = GDASProfile(trainer_preset='darts', epochs=50)
            >>> gdas_profile.configure_sampler(tau_min=02, tau_max=20)

        The accepted keyword arguments should align with the sampler's configuration and
        the attributes can be configured dynamically.

        Raises:
            ValueError: If an unrecognized keyword is passed in kwargs.


        Returns:
            None
        """
        super().configure_sampler(**kwargs)


class ReinMaxProfile(GDASProfile):
    """Train Profile class for ReinMax (Multi-objective Differentiable Neural
    Architecture Search).
    It inherits from the GDASProfile class and sets the sampler config for ReinMax
    sampler.
    """

    SAMPLER_TYPE = SamplerType.REINMAX


class SNASProfile(BaseProfile, ABC):
    """Train Profile class for SNAS (Stochastic Neural Architecture Search)..
    It inherits from the BaseProfile class and sets the sampler config for SNAS sampler.

    Attributes:
        temp_init (float): Initial temperature for sampling.
        temp_min (float): Minimum temperature for sampling.
        temp_annealing (bool): Flag to  use temperature annealing.
        total_epochs (int): Total number of epochs for training.
    """

    SAMPLER_TYPE = SamplerType.SNAS

    def __init__(
        self,
        trainer_preset: str | TrainerPresetType,
        epochs: int,
        temp_init: float = 1.0,
        temp_min: float = 0.03,
        temp_annealing: bool = True,
        **kwargs: Any,
    ) -> None:
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_annealing = temp_annealing
        self.total_epochs = epochs

        super().__init__(  # type: ignore
            self.SAMPLER_TYPE,
            trainer_preset,
            epochs,
            **kwargs,
        )

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

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        """Configures the sampler settings based on provided keyword arguments.

        The configuration options are passed as keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys \
                include:

                sample_frequency (str): The rate at which samples should be taken.

                arch_combine_fn (str): The function used to combine architectures \
                (str). when FairDARTS is used, this should be 'sigmoid'. Default value \
                is 'default'.

                temp_init (float): Initial temperature for sampling.

                temp_min (float): Minimum temperature for sampling.

                temp_annealing (bool): Flag to  use temperature annealing.

                total_epochs (int): Total number of epochs for training.

        The accepted keyword arguments should align with the sampler's configuration and
        the attributes can be configured dynamically.

        Raises:
            ValueError: If an unrecognized keyword is passed in kwargs.


        Returns:
            None
        """
        super().configure_sampler(**kwargs)


class DRNASProfile(BaseProfile, ABC):
    """Train Profile class for DRNAS (Dirichlet Neural Architecture Search).
    It inherits from the BaseProfile class and sets the sampler config for DRNAS
    sampler.

    """

    SAMPLER_TYPE = SamplerType.DRNAS

    def __init__(
        self,
        trainer_preset: str | TrainerPresetType,
        epochs: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(  # type: ignore
            self.SAMPLER_TYPE,
            trainer_preset,
            epochs,
            **kwargs,
        )

    def _initialize_sampler_config(self) -> None:
        drnas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "arch_combine_fn": self.sampler_arch_combine_fn,
        }
        self.sampler_config = drnas_config  # type: ignore

    def _initialize_trainer_config_nb201(self) -> None:
        trainer_config = {
            "lr": 0.025,
            "arch_lr": 3e-4,
            "epochs": self.epochs,  # 100
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

        # self.tau_min = 1
        # self.tau_max = 10
        self.trainer_config = trainer_config
        searchspace_config = {"N": 5, "C": 16}
        if hasattr(self, "searchspace_config"):
            self.searchspace_config.update(**searchspace_config)
        else:
            self.searchspace_config = searchspace_config

    def _initialize_trainer_config_darts(self) -> None:
        default_train_config = {
            "lr": 0.1,
            "arch_lr": 6e-4,
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
            "criterion": "cross_entropy",
            "batch_size": 64,
            "learning_rate_min": 0.0,
            # "drop_path_prob": 0.3,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.5,
            "use_data_parallel": False,
            "checkpointing_freq": 2,
            "seed": self.seed,
        }
        self.trainer_config = default_train_config
        searchspace_config = {"layers": 20, "C": 36}
        if hasattr(self, "searchspace_config"):
            self.searchspace_config.update(**searchspace_config)
        else:
            self.searchspace_config = searchspace_config

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        """Configures the sampler settings based on provided keyword arguments.

        The configuration options are passed as keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                sample_frequency (str): The rate at which samples should be taken \
                (int).

                arch_combine_fn (str): The function used to combine architectures.\
                when FairDARTS is used, this should be 'sigmoid'. Default value is
                'default'.

        Example:
            >>> from confopt.profile import DRNASProfile
            >>> drnas_profile = DRNASProfile(trainer_preset='darts', epochs=50)
            >>> drnas_profile.configure_sampler(arch_combine_fn='sigmoid')

        The accepted keyword arguments should align with the sampler's configuration and
        the attributes can be configured dynamically.

        Raises:
            ValueError: If an unrecognized keyword is passed in kwargs.


        Returns:
            None
        """
        super().configure_sampler(**kwargs)


class CompositeProfile(BaseProfile, ABC):
    SAMPLER_TYPE = SamplerType.COMPOSITE

    def __init__(
        self,
        trainer_preset: str | TrainerPresetType,
        samplers: list[str | SamplerType],
        epochs: int,
        # GDAS configs
        tau_min: float = 0.1,
        tau_max: float = 10,
        # SNAS configs
        temp_init: float = 1.0,
        temp_min: float = 0.03,
        temp_annealing: bool = True,
        **kwargs: Any,
    ) -> None:
        self.samplers = []
        for sampler in samplers:
            if isinstance(sampler, str):
                sampler = SamplerType(sampler)
            self.samplers.append(sampler)

        self.tau_min = tau_min
        self.tau_max = tau_max

        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_annealing = temp_annealing

        super().__init__(  # type: ignore
            self.SAMPLER_TYPE,
            trainer_preset,
            epochs,
            **kwargs,
        )

    def _initialize_sampler_config(self) -> None:
        """Initializes the sampler configuration for Composite samplers.

        The sampler configuration includes the sample frequency and the architecture
        combine function.

        Args:
            None

        Returns:
            None
        """
        self.sampler_config: dict[int, dict] = {}  # type: ignore
        for i, sampler in enumerate(self.samplers):
            if sampler in [SamplerType.DARTS, SamplerType.DRNAS]:
                config = {
                    "sampler_type": sampler,
                    "sample_frequency": self.sampler_sample_frequency,
                    "arch_combine_fn": self.sampler_arch_combine_fn,
                }
            elif sampler in [SamplerType.GDAS, SamplerType.REINMAX]:
                config = {
                    "sampler_type": sampler,
                    "sample_frequency": self.sampler_sample_frequency,
                    "arch_combine_fn": self.sampler_arch_combine_fn,
                    "tau_min": self.tau_min,
                    "tau_max": self.tau_max,
                }
            elif sampler == SamplerType.SNAS:
                config = {
                    "sampler_type": sampler,
                    "sample_frequency": self.sampler_sample_frequency,
                    "arch_combine_fn": self.sampler_arch_combine_fn,
                    "temp_init": self.temp_init,
                    "temp_min": self.temp_min,
                    "temp_annealing": self.temp_annealing,
                    "total_epochs": self.epochs,
                }
            else:
                raise AttributeError(f"Illegal sampler type {sampler} provided!")

            self.sampler_config[i] = config

    @override
    def configure_sampler(  # type: ignore[override]
        self, sampler_config_map: dict[int, dict]
    ) -> None:
        """Configures the sampler settings based on the provided configurations.

        Args:
            sampler_config_map (dict[int, dict]): A dictionary where each key is an \
                integer representing the order of the sampler (zero-indexed), and \
                each value is a dictionary containing the configuration parameters \
                for that sampler.

            The inner configuration dictionary can contain different sets of keys
            depending on the type of sampler being configured. The available keys
            include:

                Generic Configurations:
                    - sample_frequency (str): The rate at which samples should be taken.
                    - arch_combine_fn (str): Function to combine architectures.
                      For FairDARTS, set this to 'sigmoid'. Default is 'default'.

                GDAS-specific Configurations:
                    - tau_min (float): Minimum temperature for sampling.
                    - tau_max (float): Maximum temperature for sampling.

                SNAS-specific Configurations:
                    - temp_init (float): Initial temperature for sampling.
                    - temp_min (float): Minimum temperature for sampling.
                    - temp_annealing (bool): Whether to apply temperature annealing.
                    - total_epochs (int): Total number of training epochs.

        The specific keys required in the dictionary depend on the type of
        sampler being used.
        Please make sure that the sample frequency of all the configurations are same.
        Each configuration is validated, and an error is raised if unknown keys are
        provided.

        Raises:
            ValueError: If an unrecognized configuration key is detected.

        Returns:
            None
        """
        assert self.sampler_config is not None

        for idx in sampler_config_map:
            for config_key in sampler_config_map[idx]:
                assert idx in self.sampler_config
                exists = False
                sampler_type = self.sampler_config[idx]["sampler_type"]
                if config_key in self.sampler_config[idx]:
                    exists = True
                    self.sampler_config[idx][config_key] = sampler_config_map[idx][
                        config_key
                    ]
                assert exists, (
                    f"{config_key} is not a valid configuration for {sampler_type}",
                    "sampler inside composite sampler",
                )


class DiscreteProfile:
    def __init__(
        self,
        trainer_preset: str | TrainerPresetType,
        domain: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.trainer_preset = (
            TrainerPresetType(trainer_preset)
            if isinstance(trainer_preset, str)
            else trainer_preset
        )
        assert isinstance(
            self.trainer_preset, TrainerPresetType
        ), f"Invalid trainer_preset type: {trainer_preset}"
        self.domain = domain
        self._initialize_trainer_config()
        self._initialize_genotype()
        self.configure_trainer(**kwargs)

    def get_trainer_config(self) -> dict:
        """Returns the trainer configuration for the discrete profile.

        Args:
            None

        Returns:
            dict: A dictionary containing the trainer configuration.
        """
        return self.train_config

    def get_genotype(self) -> str:
        """Returns the genotype of the discrete profile.

        Args:
            None

        Returns:
            str: The genotype of the discrete profile.
        """
        return self.genotype

    def _initialize_trainer_config(self) -> None:
        """Initializes the trainer configuration for the discrete profile.
        This method sets the default training configuration parameters for the
        discrete model.

        Args:
            None

        Returns:
            None
        """
        default_train_config = {
            "searchspace": self.trainer_preset,
            "lr": 0.025,
            "epochs": 100,
            "optim": "sgd",
            "scheduler": "cosine_annealing_lr",
            "optim_config": {
                "momentum": 0.9,
                "nesterov": False,
                "weight_decay": 3e-4,
            },
            "criterion": "cross_entropy",
            "batch_size": 96,
            "learning_rate_min": 0.0,
            # "channel": 36,
            "print_freq": 2,
            "drop_path_prob": 0.3,
            "auxiliary_weight": 0.4,
            "cutout": 1,
            "cutout_length": 16,
            "train_portion": 1,
            "use_ddp": False,
            "checkpointing_freq": 2,
            "seed": 0,
            "use_auxiliary_skip_connection": False,
        }
        self.train_config = default_train_config

    def _initialize_genotype(self) -> None:
        """Initializes the genotype for the discrete profile.
        This method sets the genotype of the best DARTS supernet found
        after 50 epochs of running the DARTS optimizer. Thus it is only
        a valid genotype if the search space is of type DARTS.

        Args:
            None

        Returns:
            None
        """
        self.genotype = str(
            DARTSGenotype(
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
        """Configure the trainer component for the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                lr (float): Learning rate for the optimizer.

                epochs (int): Number of training epochs.

                optim (str): Optimizer type. Can be 'sgd', 'adam', etc.

                optim_config (dict): Additional configuration for the optimizer.

                ...

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        for config_key in kwargs:
            assert (
                config_key in self.train_config
            ), f"{config_key} not a valid configuration for training a \
            discrete architecture"
            self.train_config[config_key] = kwargs[config_key]

    def set_search_space_config(self, config: dict) -> None:
        """Sets the search space configuration for the discrete profile.
        This method allows the user to set custom configuration parameters.

        Args:
            config (dict): A dictionary containing the search space configuration.

        Returns:
            None
        """
        self.searchspace_config = config

    def configure_searchspace(self, **config: Any) -> None:
        """Configure the search space for the supernet.

        Args:
            **config: Arbitrary keyword arguments. Possible depend on the \
                the trainer preset type. For more information please check the \
                Parameters of the supernet of each search space.


        Returns:
            None
        """
        if not hasattr(self, "searchspace_config"):
            self.searchspace_config = config
        else:
            self.searchspace_config.update(config)

    def configure_extra(self, **config: Any) -> None:
        """Configure any extra settings for the supernet.
        Could be useful for tracking Weights & Biases metadata.

        Args:
            **config: Arbitrary keyword arguments.

        Returns:
            None
        """
        if not hasattr(self, "extra_config"):
            self.extra_config = config
        else:
            self.extra_config.update(config)

    def get_searchspace_config(self, dataset_str: str) -> dict:
        """Returns the search space configuration based on the trainer preset type.

        Args:
            dataset_str (str): The dataset string.

        Raises:
            ValueError: If the search space is not any of the valid values of
            NasBench201, DARTS, NasBench1Shot1 or TransNasBench101.

        Returns:
            dict: A dictionary containing the search space configuration which
            would be passed to as the arguments for creating the new discrete
            model object.
        """
        if self.trainer_preset == TrainerPresetType.NB201:
            searchspace_config = {
                "N": 5,  # num_cells
                "C": 16,  # channels
                "num_classes": get_num_classes(dataset_str),
            }
        elif self.trainer_preset == TrainerPresetType.DARTS:
            searchspace_config = {
                "C": 36,  # init channels
                "layers": 20,  # number of layers
                "auxiliary": False,
                "num_classes": get_num_classes(dataset_str),
            }
        elif self.trainer_preset == TrainerPresetType.TNB101:
            assert self.domain is not None, "domain must be specified"
            searchspace_config = {
                "domain": self.domain,  # type: ignore
                "num_classes": get_num_classes(dataset_str, domain=self.domain),
            }
        else:
            raise ValueError("trainer preset is not correct")
        if hasattr(self, "searchspace_config"):
            searchspace_config.update(self.searchspace_config)
        return searchspace_config

    def get_extra_config(self) -> dict:
        """This method returns the extra configuration parameters for the
        discrete profile for example could be used for Weights & Biases metadata.

        Args:
            None

        Returns:
            dict: A dictionary containing extra configuration parameters.
        """
        return self.extra_config if hasattr(self, "extra_config") else {}

    def get_run_description(self) -> str:
        """This method returns a string description of the run configuration.
        The description is used for tracking purposes in Weights & Biases.

        Args:
            None

        Returns:
            str: A string describing the run configuration.
        """
        run_configs = []
        run_configs.append(f"train_model_{self.train_config.get('searchspace')}")
        run_configs.append(f"epochs_{self.train_config.get('epochs')}")
        run_configs.append(f"seed_{self.train_config.get('seed')}")
        return "-".join(run_configs)
