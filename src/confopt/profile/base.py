from __future__ import annotations

from typing import Any, Literal
import warnings

import torch

from confopt.enums import SamplerType, TrainerPresetType

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO Change this to real data
ADVERSARIAL_DATA = (
    torch.randn(2, 3, 32, 32).to(DEVICE),
    torch.randint(0, 9, (2,)).to(DEVICE),
)


class BaseProfile:
    """Base class for configuring the supernet and the experiment's Profile.

    This class serves as a foundational component for setting up various aspects of
    training the supernet, including search spaces, epochs, and more advanced
    configurations such as dropout rates, perturbation settings, regularization
    settings, pruning and LoRA configurations. It offers flexibility in modifying
    the training experiment of the supernet through multiple setup methods. For
    further details on the specific configurations, please refer to the individual
    methods. The examples provided in each of the components presents how can each
    component be used.

    Parameters:
        sampler_type (SamplerType or str): Type of sampler to use.
        searchspace (SearchSpaceType or str): Type of search space.
        epochs (int): Number of training epochs.
        sampler_sample_frequency (str): Frequency of sampling. Valid values are \
            'step' or 'epoch'. Defaults to 'step'.

        is_partial_connection (bool): Flag to enable partial connections in the \
            supernet. Defaults to False.
        partial_connector_config (dict, optional): Configuration for partial \
            connector if is_partial_connection is True.

        dropout (float, optional): Dropout operation rate for architectural \
            parameters. Defaults to None.

        perturbation (str, optional): Type of perturbation to apply. Valid values \
            are 'adversarial' and 'random'. Defaults to None.
        perturbator_sample_frequency (str): Sampling frequency for perturbator. \
            Defaults to 'epoch'.
        perturbator_config (dict, optional): Configuration for the perturbator.

        entangle_op_weights (bool): Flag to enable weight entanglement \
            between candidate operations. Defaults to False.

        lora_rank (int): Rank for LoRA configuration. Defaults to 0.
        lora_warm_epochs (int): Number of warm-up epochs for LoRA. Defaults to 0.
        lora_toggle_epochs (list[int], optional): Specific epochs to toggle LoRA \
            configuration. Defaults to None.
        lora_toggle_probability (float, optional): Probability to toggle LoRA \
            configuration. Defaults to None.

        seed (int): Seed for random number generators to ensure reproducibility. \
            Defaults to 100.

        oles (bool): Flag to enable OLES. Defaults to False.
        calc_gm_score (bool): Flag to calculate GM score for OLES. Required if \
              oles is True.

        prune_epochs (list[int], optional): List of epochs to apply pruning at. \
            Defaults to None.
        prune_fractions (list[float], optional): Fractions to prune in specified \
            epochs. Defaults to None.

        is_arch_attention_enabled (bool): Flag to enable Multi-head attention for \
            architectural parameters Defaults to False.

        is_regularization_enabled (bool): Flag to enable regularization during \
            training. Defaults to False.
        regularization_config (dict, optional): Configuration for regularization if \
            regularization is enabled.
        sampler_arch_combine_fn (str): Function to combine architecture samples. \
            Used in FairDARTS. Defaults to 'default'.

        pt_select_architecture (bool): Flag to enable supernet's projection. \
            Defaults to False.

        searchspace_domain (str, optional): Domain/Task of the search space \
            TransNasBench101. Defaults to None.

        use_auxiliary_skip_connection (bool): Flag to use auxiliary skip \
            connections in the supernet's edges(OperationBlock). Defaults to False.

        searchspace_subspace (str, optional): Subspace of the search space NB1Shot1. \
            Defaults to None.

        early_stopper (str, optional): Strategy for early stopping. Defaults to None.
        early_stopper_config (dict, optional): Configuration for early stopping if \
            early_stopper is not None.

        synthetic_dataset_config (dict, optional): Configuration for using a synthetic \
            dataset. Defaults to None.

        extra_config (dict, optional): Any additional configurations that may be \
            needed for example could be used for Weights & Biases metadata.

        use_dynamic_exploration (bool): Flag to use dynamic exploration \
            connections in the supernet's edges(OperationBlock). Defaults to False.

        dynamic_exploration_config (dict, optional): Configuration for dynamic \
            exploration if dynamic exploration is enabled.

    """

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        sampler_type: str | SamplerType,
        trainer_preset: str | TrainerPresetType,
        epochs: int = 50,
        *,
        sampler_sample_frequency: str = "step",
        is_partial_connection: bool = False,
        partial_connector_config: dict | None = None,
        dropout: float | None = None,
        perturbation: str | None = None,
        perturbator_sample_frequency: str = "epoch",
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
        regularization_config: dict | None = None,
        sampler_arch_combine_fn: str = "default",
        pt_select_architecture: bool = False,
        searchspace_domain: str | None = None,
        use_auxiliary_skip_connection: bool = False,
        searchspace_subspace: str | None = None,
        early_stopper: str | None = None,
        early_stopper_config: dict | None = None,
        synthetic_dataset_config: dict | None = None,
        extra_config: dict | None = None,
        use_dynamic_exploration: bool = False,
        dynamic_exploration_config: dict | None = None,
    ) -> None:
        """Initialize a BaseProfile instance with configurations for training the
        supernet.

        Raises:
            AssertionError: If any of the provided configurations are invalid or \
                inconsistent.
        """
        self.trainer_preset = (
            TrainerPresetType(trainer_preset)
            if isinstance(trainer_preset, str)
            else trainer_preset
        )
        self.sampler_type = (
            SamplerType(sampler_type) if isinstance(sampler_type, str) else sampler_type
        )

        assert isinstance(
            self.trainer_preset, TrainerPresetType
        ), f"Illegal value {self.trainer_preset} for trainer_preset"
        assert isinstance(
            self.sampler_type, SamplerType
        ), f"Illegal value {self.sampler_type} for sampler_type"

        if self.trainer_preset == TrainerPresetType.TNB101:
            assert searchspace_domain in [
                "class_object",
                "class_scene",
            ], "searchspace_domain must be either class_object or class_scene"
        else:
            assert (
                searchspace_domain is None
            ), "searchspace_domain is not required for this searchspace"
        if trainer_preset in ("nb1shot1", TrainerPresetType.NB1SHOT1):
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
        self.early_stopper = early_stopper
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
        self._initialize_regularization_config()

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

        if early_stopper_config is not None:
            self.configure_early_stopper(**early_stopper_config)
        else:
            self.early_stopper_config: dict | None = None

        if synthetic_dataset_config is not None:
            self.synthetic_dataset_config = synthetic_dataset_config
        else:
            self.synthetic_dataset_config = None  # type: ignore

        if extra_config is not None:
            self.extra_config = extra_config
        else:
            self.extra_config = None  # type: ignore

        self.use_dynamic_exploration = use_dynamic_exploration
        if self.use_dynamic_exploration:
            dynamic_exploration_config = (
                dynamic_exploration_config if dynamic_exploration_config else {}
            )
            self._set_dynamic_exploration_configs(**dynamic_exploration_config)

    def _set_dynamic_exploration_configs(
        self, attention_weight: float = 1, min_attention_weight: float = 1e-4
    ) -> None:
        """Set the configuration for the dynamic exploration oneshot \
            module (DynamicAttentionExplorer).

        Args:
            attention_weight (float): Initial attention weight for the DAN module.\
                Defaults to 1.
            min_attention_weight (float): Minimum attention weight possible\
                at the last schedule step. Defaults to 1e-4.

        Returns:
            None
        """
        self.dynamic_exploration_config = {
            "total_epochs": self.epochs,
            "attention_weight": attention_weight,
            "min_attention_weight": min_attention_weight,
        }

    def _set_pt_select_configs(
        self,
        pt_select_architecture: bool = False,
        pt_projection_criteria: Literal["acc", "loss"] = "acc",
        pt_projection_interval: int = 10,
    ) -> None:
        """Set the configuration for the projection-based selection of the supernet.

        Args:
            pt_select_architecture (bool): Flag to enable projection-based \
                selection of the supernet.
            pt_projection_criteria (str): Criteria for projection. Can be \
                'acc' or 'loss'.
            pt_projection_interval (int): Interval for applying the projection \
                while training.

        Returns:
            None
        """
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
        """Set the configuration for the pruning of the supernet.

        Args:
            prune_epochs (list[int] | None): List of epochs to apply pruning at.
            prune_fractions (list[float] | None): List of fractions to prune in \
                the specified epochs.

        Raises:
            AssertionError: If prune_epochs and prune_fractions are not of \
                the same length.

        Returns:
            None
        """
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
        """Set the configuration for LoRA (Low-Rank Adaptation) layers.

        Args:
            lora_rank (int): Rank for LoRA configuration. Defaults to 0.
            lora_warm_epochs (int): Number of warm-up epochs before \
                initializing LoRA _A_ and LoRA _B_. Defaults to 0.
            lora_dropout (float): Dropout rate for LoRA layers. Defaults to 0.
            lora_alpha (int): Scaling factor for LoRA layers. Defaults to 1.
            lora_toggle_probability (float | None): Probability to toggle \
                LoRA and deactivate it. Defaults to None.
            merge_weights (bool): Flag to merge LoRA weights. Defaults to True.
            toggle_epochs (list[int] | None): Specific epochs to toggle \
                LoRA configuration. Defaults to None.

        Returns:
            None
        """
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
        """Set the configuration for OLES (Operation-Level Early Stopping).

        Args:
            oles (bool): Flag to enable OLES. Defaults to False.
            calc_gm_score (bool): Flag to calculate Gradient Matching \
                score for OLES. Defaults to False.

        Raises:
            UserWarning: If OLES is enabled but calc_gm_score is not \
                set to True.

        Returns:
            None
        """
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
        """Set the configuration for perturbation of the supernet.

        Args:
            perturb_type (str | None): Type of perturbation to apply. \
                Valid values are 'adversarial' and 'random'.
            perturbator_sample_frequency (str): Sampling frequency for \
                perturbator. Defaults to 'epoch'.

        Raises:
            AssertionError: If perturbator_sample_frequency is not 'epoch' or 'step'.
            AssertionError: If perturb_type is neither the string values \
                'adversarial', 'random', 'none' or None.

        Returns:
            None
        """
        assert perturbator_sample_frequency in ["epoch", "step"]
        assert perturb_type in ["adversarial", "random", "none", None]
        if perturb_type is None:
            self.perturb_type = "none"
        else:
            self.perturb_type = perturb_type
        self.perturbator_sample_frequency = perturbator_sample_frequency
        self._initialize_perturbation_config()

    def _set_partial_connector(self, is_partial_connection: bool = False) -> None:
        """Initializes the default configuration for the partial connector.

        Args:
            is_partial_connection (bool): Flag to enable partial connections in \
                the supernet. Defaults to False.

        Returns:
            None
        """
        self.is_partial_connection = is_partial_connection
        self._initialize_partial_connector_config()

    def _set_dropout(self, dropout: float | None = None) -> None:
        """Initializes the default configurations for dropout.

        Args:
            dropout (float | None): Dropout operation rate for architectural \
                parameters. Must be in the range [0, 1). Defaults to None.

        Returns:
            None
        """
        self.dropout = dropout
        self._initialize_dropout_config()

    def get_config(self) -> dict:
        """This method returns a dictionary representation of the Profile class. \
        The configurations are used for training the supernet.

        Args:
            None

        Returns:
            dict: A dictionary representation of the Profile class.
        """
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
            "searchspace": self.trainer_preset.value,
            "searchspace_domain": self.searchspace_domain,
            "weight_type": weight_type,
            "oles": self.oles_config,
            "pt_selection": self.pt_select_configs,
            "is_arch_attention_enabled": self.is_arch_attention_enabled,
            "regularization": self.regularization_config,
            "use_auxiliary_skip_connection": self.use_auxiliary_skip_connection,
            "early_stopper": self.early_stopper,
            "early_stopper_config": self.early_stopper_config,
            "synthetic_dataset_config": self.synthetic_dataset_config,
        }

        if hasattr(self, "pruner_config"):
            config.update({"pruner": self.pruner_config})

        if hasattr(self, "searchspace_config") and self.searchspace_config is not None:
            config.update({"search_space": self.searchspace_config})

        if hasattr(self, "extra_config") and self.extra_config is not None:
            config.update(self.extra_config)

        if hasattr(self, "dynamic_exploration_config"):
            config.update({"dynamic_exploration": self.dynamic_exploration_config})

        return config

    def _initialize_sampler_config(self) -> None:
        """Initializes the configuration for the sampler.
        The inherited classes override this with their own implementations.

        Args:
            None

        Returns:
            None
        """
        self.sampler_config = None

    def _initialize_perturbation_config(self) -> None:
        """Initialize the configuration for the perturbation based on the perturb_type.

        Args:
            None

        Returns:
            None
        """
        if self.perturb_type == "adversarial":
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
        """Initialize the configuration for the partial connector.
        If the is_partial_connection flag is disabled, the configuration is set
        to None, otherwise it is set to a default configuration.

        Note:
            The default configuration for the partial connector includes:
            - `k`: 4 (1/k of the number of channels would be used during training)
            - `num_warm_epoch`: 15 (indicates the number of warm-up epochs)

        Args:
            None

        Returns:
            None
        """
        if self.is_partial_connection:
            partial_connector_config = {"k": 4, "num_warm_epoch": 15}
            self.configure_searchspace(k=partial_connector_config["k"])
        else:
            partial_connector_config = None
        self.partial_connector_config = partial_connector_config

    def _initialize_trainer_config(self) -> None:
        """Initialize the configuration for the trainer based on the
        trainer_preset.

        Args:
            None

        Returns:
            None
        """
        if self.trainer_preset == TrainerPresetType.NB201:
            self._initialize_trainer_config_nb201()
        elif self.trainer_preset in (
            TrainerPresetType.BABYDARTS,
            TrainerPresetType.DARTS,
            TrainerPresetType.RobustDARTS,
        ):
            self._initialize_trainer_config_darts()
        elif self.trainer_preset == TrainerPresetType.NB1SHOT1:
            self._initialize_trainer_config_1shot1()
        elif self.trainer_preset == TrainerPresetType.TNB101:
            self._initialize_trainer_config_tnb101()

    def _initialize_dropout_config(self) -> None:
        """Initialize the configuration for the dropout module.

        Args:
            None

        Returns:
            None
        """
        dropout_config = {
            "p": self.dropout if self.dropout is not None else 0.0,
            "p_min": 0.0,
            "anneal_frequency": "epoch",
            "anneal_type": "linear",
            "max_iter": self.epochs,
        }
        self.dropout_config = dropout_config

    def _initialize_regularization_config(self) -> None:
        """Initialize the configuration for the regularization module.

        Args:
            None

        Returns:
            None
        """
        regularization_config = {
            "reg_weights": [0.0],
            "loss_weight": 1.0,
            "active_reg_terms": [],
            "drnas_config": {"reg_scale": 1e-3, "reg_type": "l2"},
            "flops_config": {},
            "fairdarts_config": {},
        }
        self.regularization_config = regularization_config

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        """Configures the sampler used for training the supernet based on attributes
        of the type of sampler.

        Args:
            **kwargs: Keyword arguments for configuring the sampler. The keys should
            match the expected configuration parameters.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid for
            the sampler type.

        Returns:
            None
        """
        assert self.sampler_config is not None
        for config_key in kwargs:
            assert (
                config_key in self.sampler_config  # type: ignore
            ), f"{config_key} not a valid configuration for the sampler of type \
                {self.sampler_type}"
            self.sampler_config[config_key] = kwargs[config_key]  # type: ignore

    def configure_perturbator(self, **kwargs) -> None:  # type: ignore
        """Configures the perturbator used for training the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                Possible keys include:
                for perturbation type 'adversarial':

                epsilon (float): Perturbation strength.

                data (tuple): Tuple of input data and target labels.

                loss_criterion (torch.nn.Module): Loss function to use.

                steps (int): Number of steps for perturbation.

                random_start (bool): Flag to  start with a random perturbation.

                sample_frequency (str): Frequency of sampling. Can be 'epoch' or 'step'.

                for perturbation type 'random':

                epsilon (float): Perturbation strength.

                sample_frequency (str): Frequency of sampling. Can be 'epoch' or 'step'.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        assert (
            self.perturb_type != "none"
        ), "Perturbator is initialized with None, \
            re-initialize with random or adversarial"

        for config_key in kwargs:
            assert (
                config_key in self.perturb_config  # type: ignore
            ), f"{config_key} not a valid configuration for the perturbator of \
                type {self.perturb_type}"
            self.perturb_config[config_key] = kwargs[config_key]  # type: ignore

    def configure_partial_connector(self, **kwargs) -> None:  # type: ignore
        """Configure the partial connector component for the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                k (int): 1/(Number of connections to keep).

                num_warm_epoch (int): Number of warm-up epochs.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
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
        """Configure the trainer component of the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                lr (float): Learning rate of the optimizer.

                arch_lr (float): Learning rate of architecture parameters.

                epochs (int): Number of training epochs.

                optim (str): Optimizer type. Can be 'sgd', 'adam', etc.

                arch_optim (str): Optimizer type of architecture parameters.

                optim_config (dict): Additional configuration of the optimizer.

                    momentum (float): Momentum for model's optimizer.

                    nesterov (bool): Flag to use Nesterov momentum for model's \
                        optimizer.

                    weight_decay (float): Weight decay for model's optimizer.

                arch_optim_config (dict): Additional configuration of the architecture \
                    optimizer.

                    weight_decay (float): Weight decay for architecture's optimizer.

                    beta (float): Beta for architecture's optimizer.

                scheduler (str): Scheduler's type.

                scheduler_config (dict): Additional configuration of the scheduler.

                criterion (str): Loss function type.

                batch_size (int): Batch size for training.

                learning_rate_min (float): Minimum learning rate.

                cutout (int): Enables cutout. If  cutout > 0, cutout is applied.

                cutout_length (int): Cutout length.

                train_portion (float): Portion of the training data to use for \
                    training the model's parameter.

                use_data_parallel (bool): Flag to use data parallelism.

                checkpointing_freq (int): Frequency of checkpointing.

                seed (int): Seed for random number generators.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        for config_key in kwargs:
            assert (
                config_key in self.trainer_config
            ), f"{config_key} not a valid configuration for the trainer"
            self.trainer_config[config_key] = kwargs[config_key]

    def configure_dropout(self, **kwargs) -> None:  # type: ignore
        """Configure the dropout module for the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                p (float): Dropout probability of the architecture parameters.

                p_min (float): Minimum dropout probability.

                anneal_frequency (str): Frequency of annealing. Can be 'epoch' or \
                    'step'.

                anneal_type (str): Type of annealing. Can be 'linear' or 'cosine'.

                max_iter (int): Maximum iterations for annealing.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        for config_key in kwargs:
            assert (
                config_key in self.dropout_config
            ), f"{config_key} not a valid configuration for the dropout module"
            self.dropout_config[config_key] = kwargs[config_key]

    def configure_lora(self, **kwargs) -> None:  # type: ignore
        """Configure the LoRA (Low-Rank Adaptation) module for the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                r (int): Rank for LoRA configuration.

                lora_dropout (float): Dropout rate for LoRA layers.

                lora_alpha (int): Scaling factor for LoRA layers.

                merge_weights (bool): Flag to merge LoRA weights.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        for config_key in kwargs:
            assert (
                config_key in self.lora_config
            ), f"{config_key} not a valid configuration for the lora layers"
            self.lora_config[config_key] = kwargs[config_key]

    def configure_oles(self, **kwargs) -> None:  # type: ignore
        """Configure the OLES (Operation-Level Early Stopping) module for the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                oles (bool): Flag to enable OLES. Defaults to False.

                calc_gm_score (bool): Flag to calculate GM score for OLES. Defaults to \
                    False.

                frequency (int): Accumalative value of GM score to check the threashold.

                threshold (float): Threshold of GM score to stop the training. \
                    Defaults to 0.4.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        for config_key in kwargs:
            assert (
                config_key in self.oles_config
            ), f"{config_key} not a valid configuration for the oles config"
            self.oles_config[config_key] = kwargs[config_key]

    def configure_regularization(self, **kwargs) -> None:  # type: ignore
        """Configure the regularization module for the supernet.

        There are three different types of regularizations in Configurable Optimizer:
            - FairDarts: FairDARTSRegularizationTerm
            - Flops: FLOPSRegularizationTerm
            - Drnas: DrNASRegularizationTerm.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                reg_weights (list[float]): List of weights for each regularization term.
                loss_weight (float): Weight for the loss term.
                active_reg_terms (list[str]): List of types of regularization terms.

                drnas_config (dict): Configuration for DRNAS regularization.
                    This dictionary can contain the following keys:
                        reg_scale (float): Scale for the regularization term.
                        reg_type (str): Type of regularization. Can be 'l1' or 'kl'.

                flops_config (dict): Configuration for FLOPS regularization.
                fairdarts_config (dict): Configuration for FairDARTS regularization.
        """
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
        """Configure the projection-based selection for the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                projection_interval (int): Interval for applying the projection while \
                    training.

                projection_criteria (str): Criteria for projection. Can be 'acc' or \
                    'loss'.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        for config_key in kwargs:
            assert config_key in self.pt_select_configs, (
                f"{config_key} not a valid configuration for the"
                + "perturbation based selection config"
            )
            self.pt_select_configs[config_key] = kwargs[config_key]

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

    def configure_extra(self, **config) -> None:  # type: ignore
        """Configure any extra settings for the supernet.
        Could be useful for tracking Weights & Biases metadata.

        Args:
            **config: Arbitrary keyword arguments.

        Returns:
            None
        """
        if self.extra_config is None:
            self.extra_config = config
        else:
            self.extra_config.update(config)

    def configure_early_stopper(self, **config: Any) -> None:
        """Configure the early stopping mechanism for the supernet.

        Args:
            **config: Arbitrary keyword arguments. Possible keys include:

                max_skip_normal (int): Maximum number of skip connections in \
                    normal cells

                max_skip_reduce (int): Maximum number of skip connections in \
                    reduction cells

                min_epochs (int): Minimum number of epochs to wait before stopping

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        if self.early_stopper_config is None:
            self.early_stopper_config = config
        else:
            self.early_stopper_config.update(config)

    def configure_synthetic_dataset(self, **config: Any) -> None:
        """Configure the synthetic dataset for the supernet.

        Args:
            **config: Arbitrary keyword arguments. Possible keys include:

                signal_width (int): Width of the signal Patch.

                shortcut_width (int): Width of the shortcut Patch.

                shortcut_strength (int): Probability of shourcut single being the \
                    valid single.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None
        """
        if self.synthetic_dataset_config is None:
            self.synthetic_dataset_config = config
        else:
            self.synthetic_dataset_config.update(config)

    def configure_dynamic_explorer(self, **kwargs) -> None:  # type: ignore
        """Configure the synthetic dataset for the supernet.

        Args:
            **kwargs: Arbitrary keyword arguments. Possible keys include:

                attention_weight (float): Initial attention weight for DAN.

                minimum_attention_weight (float): Minimum attention weight possible\
                    at the last schedule step.

        Raises:
            AssertionError: If any of the provided configuration keys are not valid.

        Returns:
            None

        """
        for config_key in kwargs:
            assert config_key in self.dynamic_exploration_config, (
                f"{config_key} not a valid configuration for the"
                + "dynamic exploration config"
            )
            self.dynamic_exploration_config[config_key] = kwargs[config_key]

    def get_run_description(self) -> str:
        """This method returns a string description of the run configuration.
        The description is used for tracking purposes in Weights & Biases.

        Args:
            None

        Returns:
            str: A string describing the run configuration.
        """
        run_configs = []
        run_configs.append(f"ss_{self.trainer_preset}")
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
        """Initialize the configuration for the trainer based on the NB201 search space.

        Args:
            None

        Returns:
            None
        """
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
        """Initialize the configuration for the trainer based on the DARTS search space.

        Args:
            None

        Returns:
            None
        """
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
        """Initialize the configuration for the trainer based on the NasBench1Shot1 \
            search space.

        Args:
            None

        Returns:
            None
        """
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
        """Initialize the configuration for the trainer based on the TransNasBench101 \
            search space.

        Args:
            None

        Returns:
            None
        """
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
