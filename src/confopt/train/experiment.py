from __future__ import annotations

import argparse
from collections import namedtuple
import json
import random
from typing import Any, Callable, Literal
import warnings

import numpy as np
import torch
from torch.backends import cudnn
import wandb

from confopt.dataset import get_dataset
from confopt.dataset.data import AbstractData
from confopt.enums import (
    CriterionType,
    DatasetType,
    OptimizerType,
    PerturbatorType,
    SamplerType,
    SchedulerType,
    SearchSpaceType,
)
from confopt.oneshot import (
    DrNASRegularizationTerm,
    Dropout,
    EarlyStopper,
    FairDARTSRegularizationTerm,
    FLOPSRegularizationTerm,
    LoRAToggler,
    PartialConnector,
    Pruner,
    RegularizationTerm,
    Regularizer,
    SDARTSPerturbator,
    SkipConnectionEarlyStopper,
    WeightEntangler,
)
from confopt.oneshot.archsampler import (
    BaseSampler,
    CompositeSampler,
    DARTSSampler,
    DRNASSampler,
    GDASSampler,
    ReinMaxSampler,
    SNASSampler,
)
from confopt.oneshot.dynamic_exploration import DynamicAttentionExplorer
from confopt.profile import (
    BaseProfile,
    DiscreteProfile,
    GDASProfile,
)
from confopt.searchspace import (
    BabyDARTSSearchSpace,
    DARTSGenotype,  # noqa: F401
    DARTSImageNetModel,
    DARTSModel,
    DARTSSearchSpace,
    NAS201Genotype,
    NASBench1Shot1SearchSpace,
    NASBench201Model,
    NASBench201SearchSpace,
    RobustDARTSSearchSpace,
    SearchSpace,
    TransNASBench101SearchSpace,
)
from confopt.train import ConfigurableTrainer, DiscreteTrainer
from confopt.train.projection import PerturbationArchSelection
from confopt.train.search_space_handler import SearchSpaceHandler
from confopt.utils import Logger, validate_model_to_load_value
from confopt.utils import distributed as dist_utils
from confopt.utils.time import check_date_format

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Experiment:
    """The Experiment class is responsible for managing the training and evaluation of
    the supernet and discrete models. It initializes the necessary components,
    manages the states to load, and handles the training process.

    Parameters:
        search_space (SearchSpace): The search space type used for the experiment.
        dataset (DatasetType): The dataset type used for the experiment.
        seed (int): The random seed for reproducibility of the runs.
        log_with_wandb (bool): Flag to enable logging with Weights & Biases.
        debug_mode (bool): Flag to enable debug mode, where we only do 5 steps for \
            each epoch.
        exp_name (str): The name of the experiment.
        dataset_domain (str | None): The domain of the dataset used for the Taskonomy \
            dataset. Valid values are 'class_object' and 'class_scene'.
        dataset_dir (str): The directory where the dataset is stored.
        api_dir (str): The directory where the API is stored to used when we are using \
            the benchmarks.
    """

    def __init__(
        self,
        search_space: SearchSpaceType,
        dataset: DatasetType,
        seed: int,
        log_with_wandb: bool = False,
        debug_mode: bool = False,
        exp_name: str = "test",
        dataset_domain: str | None = None,
        dataset_dir: str = "datasets",
        api_dir: str = "api",
    ) -> None:
        self.searchspace_type = search_space
        self.dataset = dataset
        self.dataset_domain = dataset_domain
        self.seed = seed
        self.log_with_wandb = log_with_wandb
        self.debug_mode = debug_mode
        self.exp_name = exp_name
        self.dataset_dir = dataset_dir
        self.api_dir = api_dir

    def _set_seed(self, rand_seed: int) -> None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        cudnn.benchmark = True
        torch.manual_seed(rand_seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(rand_seed)

    def init_ddp(self) -> None:
        """Initializes the distributed data parallel (DDP) environment.

        Args:
            None

        Returns:
            None
        """
        dist_utils.init_distributed()

    def cleanup_ddp(self) -> None:
        """Kills the distributed data parallel (DDP) process.

        Args:
            None

        Returns:
            None
        """
        dist_utils.cleanup()

    def train_supernet(
        self,
        profile: BaseProfile,
        model_to_load: str | int | None = None,
        exp_runtime_to_load: str | None = None,
        use_benchmark: bool = False,
    ) -> ConfigurableTrainer:
        """Trains a supernet using the given profile with options for loading previous
        runs.

        Args:
            profile (BaseProfile): Contains configurations for training the supernet,
            including component settings and architectural specifications.

            model_to_load (str | int | None): Specifies the training state to load the
            supernet from. Valid values are "last" or "best", representing the most
            recent or the best-performing model checkpoint, respectively.
            If an integer is provided, it represents the epoch number from which
            training should be continued.
            If `None`, then it starts training the model from scratch.

            exp_runtime_to_load (str | None): The particular experiment runtime to
            load the model from.If `None`, the model will be loaded from the last
            runtime.

            use_benchmark (bool): If `True`, uses a benchmark API for evaluation.

        Returns:
            ConfigurableTrainer: The trained supernet.
        """
        config = profile.get_config()
        run_name = profile.get_run_description()
        config["dataset"] = self.dataset.value

        assert hasattr(profile, "sampler_type")
        self.sampler_str = SamplerType(profile.sampler_type)
        self.perturbator_str = PerturbatorType(profile.perturb_type)
        self.is_partial_connection = profile.is_partial_connection
        self.dropout_p = profile.dropout
        self.edge_normalization = profile.is_partial_connection
        self.entangle_op_weights = profile.entangle_op_weights
        oles_config = config["oles"]

        return self._train_supernet(
            config=config,
            model_to_load=model_to_load,
            exp_runtime_to_load=exp_runtime_to_load,
            use_benchmark=use_benchmark,
            run_name=run_name,
            calc_gm_score=oles_config["calc_gm_score"],
            oles=oles_config["oles"],
            oles_frequency=oles_config["frequency"],
            oles_threshold=oles_config["threshold"],
        )

    def _init_wandb(self, run_name: str, config: dict) -> None:
        wandb.init(  # type: ignore
            name=run_name,
            project=(
                config.get("project_name", "Configurable_Optimizer")
                if config is not None
                else "Configurable_Optimizer"
            ),
            config=config,
        )

    def _train_supernet(
        self,
        config: dict | None = None,
        model_to_load: str | int | None = None,
        exp_runtime_to_load: str | None = None,
        use_benchmark: bool = False,
        run_name: str = "supernet_run",
        calc_gm_score: bool = False,
        oles: bool = False,
        oles_frequency: int = 20,
        oles_threshold: float = 0.4,
    ) -> ConfigurableTrainer:
        self._set_seed(self.seed)

        load_last_run = model_to_load is not None and not exp_runtime_to_load

        self.logger = Logger(
            log_dir="logs",
            exp_name=self.exp_name,
            search_space=self.searchspace_type.value,
            dataset=str(self.dataset.value),
            seed=self.seed,
            runtime=exp_runtime_to_load,
            use_supernet_checkpoint=True,
            last_run=load_last_run,
        )

        self.logger.log(
            "Logs and checkpoints will be saved in the following directory: "
            + self.logger.path(None)
        )
        config["save_dir"] = self.logger.path(None)  # type:ignore

        self._init_components(
            self.searchspace_type,
            self.sampler_str,
            self.perturbator_str,
            config=config,
            use_benchmark=use_benchmark,
        )
        if self.log_with_wandb:
            self._init_wandb(run_name, config)  # type: ignore

        trainer = self._initialize_configurable_trainer(
            config=config,  # type: ignore
            model_to_load=model_to_load,
        )

        config_str = json.dumps(config, indent=2, default=str)
        self.logger.log(
            f"Training the supernet with the following configuration: \n{config_str}"
        )

        trainer.train(
            search_space_handler=self.search_space_handler,  # type: ignore
            log_with_wandb=self.log_with_wandb,
            lora_warm_epochs=config["trainer"].get(  # type: ignore
                "lora_warm_epochs", 0
            ),
            calc_gm_score=calc_gm_score,
            oles=oles,
            oles_frequency=oles_frequency,
            oles_threshold=oles_threshold,
        )

        if self.log_with_wandb:
            wandb.finish()  # type: ignore

        return trainer

    def _init_components(
        self,
        searchspace_type: SearchSpaceType,
        sampler_type: SamplerType,
        perturbator_type: PerturbatorType,
        config: dict | None = None,
        use_benchmark: bool = False,
    ) -> None:
        if config is None:
            config = {}  # type : ignore
        self._set_search_space(searchspace_type, config.get("search_space", {}))
        self._set_sampler(sampler_type, config.get("sampler", {}))
        self._set_perturbator(perturbator_type, config.get("perturbator", {}))
        self._set_partial_connector(config.get("partial_connector", {}))
        self._set_dropout(config.get("dropout", {}))
        self._set_pruner(config.get("pruner", {}))
        self.benchmark_api: None | Any = None

        if use_benchmark:
            if (
                searchspace_type == SearchSpaceType.RobustDARTS
                and config.get("search_space", {}).get("space") == "s4"
            ):
                warnings.warn(
                    "Argument use_benchmark was set to True with s4 space of"
                    + " RobustDARTSSearchSpace. Consider setting it to False",
                    stacklevel=1,
                )
                self.benchmark_api = None
            else:
                self._set_benchmark_api(searchspace_type, config.get("benchmark", {}))
        else:
            self.benchmark_api = None

        self._set_lora_toggler(config.get("lora", {}), config.get("lora_extra", {}))
        self._set_weight_entangler()
        self._set_regularizer(config.get("regularization", {}))
        self._set_dynamic_explorer(config.get("dynamic_exploration", {}))
        self._set_profile(config)
        self._set_early_stopper(
            config["early_stopper"], config.get("early_stopper_config", {})
        )

    def _set_search_space(
        self,
        search_space: SearchSpaceType,
        config: dict,
    ) -> None:
        if search_space == SearchSpaceType.NB201:
            self.search_space = NASBench201SearchSpace(**config)
        elif search_space == SearchSpaceType.DARTS:
            self.search_space = DARTSSearchSpace(**config)
        elif search_space == SearchSpaceType.NB1SHOT1:
            self.search_space = NASBench1Shot1SearchSpace(**config)
        elif search_space == SearchSpaceType.TNB101:
            self.search_space = TransNASBench101SearchSpace(**config)
        elif search_space == SearchSpaceType.BABYDARTS:
            self.search_space = BabyDARTSSearchSpace(**config)
        elif search_space == SearchSpaceType.RobustDARTS:
            self.search_space = RobustDARTSSearchSpace(**config)

    def _set_benchmark_api(
        self,
        search_space: SearchSpaceType,
        config: dict,
    ) -> None:
        if search_space == SearchSpaceType.NB1SHOT1:
            from confopt.benchmark import NB101Benchmark

            self.benchmark_api = NB101Benchmark("full", self.api_dir)
        elif search_space == SearchSpaceType.NB201:
            from confopt.benchmark import NB201Benchmark

            self.benchmark_api = NB201Benchmark(self.api_dir)
        elif search_space in (SearchSpaceType.DARTS, SearchSpaceType.RobustDARTS):
            from confopt.benchmark import NB301Benchmark

            self.benchmark_api = NB301Benchmark(api_root_dir=self.api_dir, **config)
        elif search_space == SearchSpaceType.TNB101:
            from confopt.benchmark import TNB101Benchmark

            self.benchmark_api = TNB101Benchmark(self.api_dir)
        else:
            print(f"Benchmark does not exist for the {search_space.value} searchspace")
            self.benchmark_api = None

    def _set_sampler(
        self,
        sampler: SamplerType,
        config: dict,
    ) -> None:
        arch_params = self.search_space.arch_parameters
        self.sampler: BaseSampler | CompositeSampler | None = None

        def _get_sampler_class(sampler: SamplerType) -> Callable:
            if sampler == SamplerType.DARTS:
                return DARTSSampler
            if sampler == SamplerType.DRNAS:
                return DRNASSampler
            if sampler == SamplerType.GDAS:
                return GDASSampler
            if sampler == SamplerType.SNAS:
                return SNASSampler
            if sampler == SamplerType.REINMAX:
                return ReinMaxSampler

            raise ValueError(f"Illegal sampler {sampler} provided")

        if sampler == SamplerType.COMPOSITE:
            sub_samplers: list[BaseSampler] = []
            for _, sampler_config in config.items():
                sampler_type = sampler_config["sampler_type"]
                del sampler_config["sampler_type"]
                sampler_component = _get_sampler_class(sampler_type)(
                    **sampler_config, arch_parameters=arch_params
                )
                sub_samplers.append(sampler_component)
            self.sampler = CompositeSampler(sub_samplers, arch_parameters=arch_params)
        else:
            self.sampler = _get_sampler_class(sampler)(
                **config, arch_parameters=arch_params
            )

    def _set_perturbator(
        self,
        petubrator_type: PerturbatorType,
        pertub_config: dict,
    ) -> None:
        self.perturbator: SDARTSPerturbator | None = None
        if petubrator_type != PerturbatorType.NONE:
            self.perturbator = SDARTSPerturbator(
                **pertub_config,
                search_space=self.search_space,
                arch_parameters=self.search_space.arch_parameters,
                attack_type=petubrator_type.value,  # type: ignore
            )

    def _set_partial_connector(self, config: dict) -> None:
        self.partial_connector: PartialConnector | None = None
        if self.is_partial_connection:
            self.partial_connector = PartialConnector(**config)

    def _set_dropout(self, config: dict) -> None:
        self.dropout: Dropout | None = None
        if self.dropout_p is not None:
            self.dropout = Dropout(**config)

    def _set_weight_entangler(self) -> None:
        self.weight_entangler = WeightEntangler() if self.entangle_op_weights else None

    def _set_pruner(self, config: dict) -> None:
        self.pruner: Pruner | None = None
        if config:
            self.pruner = Pruner(
                searchspace=self.search_space,
                prune_epochs=config.get("prune_epochs", []),
                prune_fractions=config.get("prune_fractions", []),
            )

    def _set_lora_toggler(self, lora_config: dict, lora_extra: dict) -> None:
        if lora_config.get("r", 0) == 0:
            self.lora_toggler = None
            return

        toggle_epochs = lora_extra.get("toggle_epochs")
        toggle_probability = lora_extra.get("toggle_probability")
        if toggle_epochs is not None:
            assert min(toggle_epochs) > lora_extra.get(
                "warm_epochs"
            ), "The first toggle epoch should be after the warmup epochs"
            self.lora_toggler = LoRAToggler(
                searchspace=self.search_space,
                toggle_epochs=toggle_epochs,
                toggle_probability=toggle_probability,
            )
        else:
            self.lora_toggler = None

    def _set_regularizer(self, config: dict) -> None:
        if config is None or len(config["active_reg_terms"]) == 0:
            self.regularizer = None
            return

        reg_terms: list[RegularizationTerm] = []
        for term in config["active_reg_terms"]:
            if term == "drnas":
                reg_terms.append(DrNASRegularizationTerm(**config["drnas_config"]))
            elif term == "flops":
                reg_terms.append(FLOPSRegularizationTerm(**config["flops_config"]))
            elif term == "fairdarts":
                reg_terms.append(
                    FairDARTSRegularizationTerm(**config["fairdarts_config"])
                )

        self.regularizer = Regularizer(
            reg_terms=reg_terms,
            reg_weights=config["reg_weights"],
            loss_weight=config["loss_weight"],
        )

    def _set_dynamic_explorer(self, config: dict) -> None:
        self.dynamic_explorer: DynamicAttentionExplorer | None = None
        if config:
            self.dynamic_explorer = DynamicAttentionExplorer(
                self.search_space,
                total_epochs=config.get("total_epochs"),  # type: ignore
                attention_weight=config.get("attention_weight", 1),
                min_attention_weight=config.get("min_attention_weight", 1e-4),
            )

    def _set_profile(self, config: dict) -> None:
        assert self.sampler is not None

        self.search_space_handler = SearchSpaceHandler(
            sampler=self.sampler,
            edge_normalization=self.edge_normalization,
            partial_connector=self.partial_connector,
            perturbation=self.perturbator,
            dropout=self.dropout,
            weight_entangler=self.weight_entangler,
            lora_toggler=self.lora_toggler,
            lora_configs=config.get("lora"),
            pruner=self.pruner,
            is_arch_attention_enabled=config.get("is_arch_attention_enabled", False),
            regularizer=self.regularizer,
            use_auxiliary_skip_connection=config.get(
                "use_auxiliary_skip_connection", False
            ),
            dynamic_explorer=self.dynamic_explorer,
        )

    def _get_criterion(self, criterion_str: str) -> torch.nn.Module:
        criterion = CriterionType(criterion_str)
        if criterion == CriterionType.CROSS_ENTROPY:
            return torch.nn.CrossEntropyLoss()

        raise NotImplementedError

    def _get_optimizer(self, optim_str: str) -> Callable | None:
        optim = OptimizerType(optim_str)
        if optim == OptimizerType.ADAM:
            return torch.optim.Adam
        elif optim == OptimizerType.SGD:  # noqa: RET505
            return torch.optim.SGD
        if optim == OptimizerType.ASGD:
            return torch.optim.ASGD
        return None

    def _get_scheduler(
        self,
        scheduler_str: str,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        eta_min: float = 0.0,
        config: dict | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if config is None:
            config = {}
        scheduler = SchedulerType(scheduler_str)
        if scheduler == SchedulerType.CosineAnnealingLR:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=num_epochs,
                eta_min=eta_min,
            )
        elif scheduler == SchedulerType.CosineAnnealingWarmRestart:  # noqa: RET505
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=config.get("T_0", 10),
                T_mult=config.get("T_mult", 1),
                eta_min=eta_min,
            )
        return None

    def _set_early_stopper(
        self, early_stopper: str | None, config: dict | None
    ) -> None:
        self.early_stopper: None | EarlyStopper = None

        if early_stopper is not None:
            assert config is not None, (
                "The configurations for the EarlyStopper is empty. "
                + "Use profile.configure_early_stopper() to fix it."
            )
            if early_stopper == "skip_connection":
                self.early_stopper = SkipConnectionEarlyStopper(**config)
            else:
                raise ValueError(f"Earlyt stopping method {early_stopper} not known!")

    def train_discrete_model(
        self,
        profile: DiscreteProfile,
        model_to_load: str | int | None = None,
        exp_runtime_to_load: str | None = None,
        use_supernet_checkpoint: bool = False,
        use_expr_search_space: bool = False,
    ) -> DiscreteTrainer:
        """Trains a discrete model using the given profile with options for loading
        specific training states.

        Args:
            profile (DiscreteProfile): Contains configurations for training the model,
                including hyperparameters and architecture details.The genotype could be
                set in the profile, or the default genotype will be used.

            model_to_load (str | int | None): Specifies the training state to load.
                Acceptable string values are "last" or "best", representing the most
                recent or the best-performing model checkpoint, respectively.
                If an integer is provided, it represents the epoch number from which
                training should be continued.
                If `None`, behavior is determined by other parameters.

            exp_runtime_to_load (str | None): The experiment runtime to load the model
                from.
                If `None`, the model will be loaded from the most recent runtime.

            use_supernet_checkpoint (bool): If `True`, initializes the model's weights
                from a supernet checkpoint.
                If `False`, the model will use checkpoints from the discrete network
                instead.

            use_expr_search_space (bool): If `True`, gets the discretized model from
                self.search_space

        Returns:
            DiscreteTrainer: The trained discrete model.

        Behavior Notes:
            - If none of the parameters are provided the default profile genotype will
              be used.
            - The default genotype in the profile refers to the best architecture found
              after 50 epochs using the DARTS optimizer on the CIFAR-10 dataset within
              the DARTS search space.
            - Setting `use_supernet_checkpoint` to `True` allows loading from the
              supernet, while `False` defaults to using checkpoints from the discrete
              network.

        Example:
            >>> trainer = experiment.train_discrete_model(
                    profile=profile,
                    model_to_load="last",
                    exp_runtime_to_load=None,
                    use_supernet_checkpoint=True,
                    use_expr_search_space=False,
                )
        """
        train_config = profile.get_trainer_config()
        searchspace_config = profile.get_searchspace_config(self.dataset.value)
        genotype_str = profile.get_genotype()
        run_name = profile.get_run_description()
        extra_config = profile.get_extra_config()

        return self._train_discrete_model(
            searchspace_config=searchspace_config,
            extra_config=extra_config,
            train_config=train_config,
            model_to_load=model_to_load,
            exp_runtime_to_load=exp_runtime_to_load,
            use_supernet_checkpoint=use_supernet_checkpoint,
            use_expr_search_space=use_expr_search_space,
            genotype_str=genotype_str,
            run_name=run_name,
        )

    def get_discrete_model_from_genotype_str(
        self,
        search_space_str: str,
        genotype_str: str,
        searchspace_config: dict,
    ) -> torch.nn.Module:
        """Returns a discrete model based on the given genotype string.

        Args:
            search_space_str (str): The search space type.
            genotype_str (str): The genotype string to use for creating the discrete
            model.
            searchspace_config (dict): Configuration for the search space.

        Raises:
            ValueError: If the search space type is not recognized or if the dataset
            is not supported.

        Returns:
            torch.nn.Module: The discrete model.
        """
        if search_space_str == SearchSpaceType.NB201.value:
            searchspace_config["genotype"] = NAS201Genotype.str2structure(genotype_str)
            discrete_model = NASBench201Model(**searchspace_config)
        elif search_space_str == SearchSpaceType.DARTS.value:
            searchspace_config["genotype"] = eval(genotype_str)
            if self.dataset in (
                DatasetType.CIFAR10,
                DatasetType.CIFAR10_MODEL,
                DatasetType.CIFAR10_SUPERNET,
                DatasetType.CIFAR100,
                DatasetType.AIRCRAFT,
            ):
                discrete_model = DARTSModel(**searchspace_config)
            elif self.dataset in (DatasetType.IMGNET16, DatasetType.IMGNET16_120):
                discrete_model = DARTSImageNetModel(**searchspace_config)
            else:
                raise ValueError("undefined discrete model for this dataset.")
        else:
            raise ValueError("undefined discrete model for this search space.")

        return discrete_model

    def get_discrete_model_from_supernet(
        self,
    ) -> SearchSpace:
        """Returns a discrete model from experiment's supernet(search space).

        Args:
            None

        Raises:
            Exception: If the experiment does not have a search space or the
            search space is not a supernet.

        Returns:
            SearchSpace: A discrete model.
        """
        # A) Use the experiment's self.search_space of the experiment.
        if hasattr(self, "search_space"):
            if self.search_space.arch_parameters:
                model = self.search_space.discretize()
                return model  # type: ignore
            raise ValueError("need to be a supernet to be able to get the discrete net")
        raise Exception("Need a searchspace to be able to fetch a discrete model")

    # logger load_genotype function handles for both supernet and discrete model
    def get_genotype_str_from_checkpoint(
        self,
        model_to_load: str | int | None = None,
        use_supernet_checkpoint: bool = False,
    ) -> str:
        """Returns the genotype string from the checkpoint.

        Args:
            model_to_load (str | int | None): Specifies the training state to load.
            Can be "last", "best", or specific epoch.
            use_supernet_checkpoint (bool): If `True`, initializes the model's weights
            from a supernet checkpoint.

        Raises:
            ValueError: If `model_to_load` is not given

        Returns:
            str: The genotype string.
        """
        if model_to_load is not None:
            genotype_str = self.logger.load_genotype(
                model_to_load=model_to_load,
                use_supernet_checkpoint=use_supernet_checkpoint,
            )
            return genotype_str

        raise ValueError("is not a valid checkpoint.")

    def get_discrete_model(
        self,
        searchspace_config: dict,
        model_to_load: str | int | None = None,
        use_supernet_checkpoint: bool = False,
        use_expr_search_space: bool = False,
        genotype_str: str | None = None,
    ) -> tuple[torch.nn.Module, str]:
        """Returns a discrete model based on the given parameters.

        Args:
            searchspace_config (dict): Configuration for the search space.
            model_to_load (str | int | None): Specifies the training state to load. Can
            be "last", "best", or specific epoch.
            use_supernet_checkpoint (bool): If `True`, initializes the model's weights
            from a supernet checkpoint.
            use_expr_search_space (bool): If `True`, gets the discretized model from
            self.search_space.
            genotype_str (str | None): The genotype string to use for creating the
            discrete model.

        Returns:
            tuple[torch.nn.Module, str]: A tuple containing the discrete model and its
            genotype string.
        """
        # A) Use the experiment's self.search_space of the experiment.
        if use_expr_search_space:
            model = self.get_discrete_model_from_supernet()
            return model, self.search_space.get_genotype()  # type: ignore
        # B, C) Use a checkpoint (discrete net or supernet) to load the model.
        if model_to_load is not None:
            genotype_str = self.get_genotype_str_from_checkpoint(
                model_to_load=model_to_load,
                use_supernet_checkpoint=use_supernet_checkpoint,
            )
        # E) Use the default genotype which is set in the discrete_profile.
        # if you don't set the genotype in the profile the genotype
        # is the best model found by darts
        # elif genotype_str is None:
        # genotype_str = searchspace_config.get("genotype")
        # assert the according genotype_str is the one for the search space.
        elif genotype_str is None:
            raise ValueError("genotype cannot be empty")

        model = self.get_discrete_model_from_genotype_str(
            self.searchspace_type.value,
            genotype_str,  # type: ignore
            searchspace_config,
        )
        return model, genotype_str  # type: ignore

    def _get_dataset(
        self,
        cutout: int,
        cutout_length: int,
        train_portion: float,
        **kwargs: Any,
    ) -> AbstractData:
        return get_dataset(
            dataset=self.dataset,
            domain=self.dataset_domain,
            root=self.dataset_dir,
            cutout=cutout,  # type: ignore
            cutout_length=cutout_length,  # type: ignore
            train_portion=train_portion,  # type: ignore
            dataset_kwargs=kwargs,
        )

    # refactor the name to train
    def _train_discrete_model(
        self,
        searchspace_config: dict,
        extra_config: dict,
        train_config: dict,
        model_to_load: str | int | None = None,
        exp_runtime_to_load: str | None = None,
        use_supernet_checkpoint: bool = False,
        use_expr_search_space: bool = False,
        genotype_str: str | None = None,
        run_name: str = "discrete_run",
    ) -> DiscreteTrainer:
        # should not care where the model comes from => genotype should be a
        # different function
        self._set_seed(self.seed)

        load_last_run = model_to_load is not None and not exp_runtime_to_load
        self.logger = Logger(
            log_dir="logs",
            exp_name=self.exp_name,
            search_space=self.searchspace_type.value,
            dataset=str(self.dataset.value),
            seed=self.seed,
            runtime=exp_runtime_to_load,
            use_supernet_checkpoint=use_supernet_checkpoint,
            last_run=load_last_run,
        )

        # different options to train a discrete model:
        # A) Use the experiment's self.search_space of the experiment.
        # B) From a supernet checkpoint, load, and discretize to get the model.
        # C) From a discerete checkpoint, load the model.
        # D) pass a genotype from the prompt to build a discrete model.
        # E) just use the default genotype which is set in the discrete_profile.

        model, genotype_str = self.get_discrete_model(
            searchspace_config=searchspace_config,
            model_to_load=model_to_load,
            use_supernet_checkpoint=use_supernet_checkpoint,
            use_expr_search_space=use_expr_search_space,
            genotype_str=genotype_str,
        )
        model.to(device=DEVICE)

        n_params_model = sum(p.numel() for p in model.parameters())
        train_config["n_params_model"] = n_params_model

        # TODO: do i need this line?
        if use_supernet_checkpoint:
            model_to_load = None
            use_supernet_checkpoint = False
            self.logger.use_supernet_checkpoint = False
            self.logger.set_up_new_run()

        self.logger.save_genotype(genotype_str)
        train_config["genotype"] = genotype_str

        if train_config.get("use_ddp", False) is True:
            assert torch.distributed.is_initialized(), "DDP is not initialized!"
            world_size = dist_utils.get_world_size()
            train_config["lr"] *= world_size  # type: ignore
            train_config["learning_rate_min"] *= world_size  # type: ignore

        Arguments = namedtuple(  # type: ignore
            "Configure", " ".join(train_config.keys())  # type: ignore
        )
        trainer_arguments = Arguments(**train_config)  # type: ignore

        data = self._get_dataset(
            cutout=trainer_arguments.cutout,  # type: ignore
            cutout_length=trainer_arguments.cutout_length,  # type: ignore
            train_portion=trainer_arguments.train_portion,  # type: ignore
        )

        w_optimizer = self._get_optimizer(trainer_arguments.optim)(  # type: ignore
            model.parameters(),
            trainer_arguments.lr,  # type: ignore
            **trainer_arguments.optim_config,  # type: ignore
        )

        w_scheduler = self._get_scheduler(
            scheduler_str=trainer_arguments.scheduler,  # type: ignore
            optimizer=w_optimizer,
            num_epochs=trainer_arguments.epochs,  # type: ignore
            eta_min=trainer_arguments.learning_rate_min,  # type: ignore
            config=train_config.get("scheduler_config", {}),  # type: ignore
        )

        criterion = self._get_criterion(
            criterion_str=trainer_arguments.criterion  # type: ignore
        )

        trainer = DiscreteTrainer(
            model=model,
            data=data,
            model_optimizer=w_optimizer,
            scheduler=w_scheduler,
            criterion=criterion,
            logger=self.logger,
            batch_size=trainer_arguments.batch_size,  # type: ignore
            use_ddp=trainer_arguments.use_ddp,  # type: ignore
            print_freq=trainer_arguments.print_freq,  # type: ignore
            drop_path_prob=trainer_arguments.drop_path_prob,  # type: ignore
            aux_weight=trainer_arguments.auxiliary_weight,  # type: ignore
            model_to_load=model_to_load,
            checkpointing_freq=trainer_arguments.checkpointing_freq,  # type: ignore
            epochs=trainer_arguments.epochs,  # type: ignore
            debug_mode=self.debug_mode,
        )
        if self.log_with_wandb:
            config = {k: v for k, v in train_config.items()}
            config.update(extra_config)

            self._init_wandb(run_name, config=config)

        trainer.train(
            epochs=trainer_arguments.epochs,  # type: ignore
            log_with_wandb=self.log_with_wandb,
        )

        trainer.test(log_with_wandb=self.log_with_wandb)

        if self.log_with_wandb:
            wandb.finish()  # type: ignore

        return trainer

    def _initialize_configurable_trainer(
        self,
        config: dict,
        model_to_load: str | int | None = None,
    ) -> ConfigurableTrainer:
        Arguments = namedtuple(  # type: ignore
            "Configure", " ".join(config["trainer"].keys())  # type: ignore
        )
        trainer_arguments = Arguments(**config["trainer"])  # type: ignore

        criterion = self._get_criterion(
            criterion_str=trainer_arguments.criterion  # type: ignore
        )

        dataset_kwargs = (
            config.get("synthetic_dataset_config")
            if config.get("synthetic_dataset_config") is not None
            else {}
        )
        data = self._get_dataset(
            cutout=trainer_arguments.cutout,  # type: ignore
            cutout_length=trainer_arguments.cutout_length,  # type: ignore
            train_portion=trainer_arguments.train_portion,  # type: ignore
            **dataset_kwargs,
        )

        model = self.search_space

        w_optimizer = self._get_optimizer(trainer_arguments.optim)(  # type: ignore
            model.model_weight_parameters(),
            trainer_arguments.lr,  # type: ignore
            **config["trainer"].get("optim_config", {}),  # type: ignore
        )

        w_scheduler = self._get_scheduler(
            scheduler_str=trainer_arguments.scheduler,  # type: ignore
            optimizer=w_optimizer,
            num_epochs=trainer_arguments.epochs,  # type: ignore
            eta_min=trainer_arguments.learning_rate_min,  # type: ignore
            config=config["trainer"].get("scheduler_config", {}),  # type: ignore
        )

        if self.edge_normalization and hasattr(model, "beta_parameters"):
            arch_optimizer = self._get_optimizer(
                trainer_arguments.arch_optim  # type: ignore
            )(
                [*model.arch_parameters, *model.beta_parameters],
                lr=config["trainer"].get("arch_lr", 0.001),  # type: ignore
                **config["trainer"].get("arch_optim_config", {}),  # type: ignore
            )
        else:
            arch_optimizer = self._get_optimizer(
                trainer_arguments.arch_optim  # type: ignore
            )(
                model.arch_parameters,
                lr=config["trainer"].get("arch_lr", 0.001),  # type: ignore
                **config["trainer"].get("arch_optim_config", {}),  # type: ignore
            )

        trainer = ConfigurableTrainer(
            model=model,
            data=data,
            model_optimizer=w_optimizer,
            arch_optimizer=arch_optimizer,
            scheduler=w_scheduler,
            criterion=criterion,
            logger=self.logger,
            batch_size=trainer_arguments.batch_size,  # type: ignore
            use_data_parallel=trainer_arguments.use_data_parallel,  # type: ignore
            model_to_load=model_to_load,
            checkpointing_freq=trainer_arguments.checkpointing_freq,  # type: ignore
            epochs=trainer_arguments.epochs,  # type: ignore
            debug_mode=self.debug_mode,
            query_dataset=self.dataset.value,
            benchmark_api=self.benchmark_api,
            early_stopper=self.early_stopper,
        )

        return trainer

    def select_perturbation_based_arch(
        self,
        profile: BaseProfile,
        model_source: Literal["supernet", "arch_selection"] = "supernet",
        model_to_load: str | int | None = None,
        exp_runtime_to_load: str | None = None,
        log_with_wandb: bool = False,
        run_name: str = "darts-pt",
        src_folder_path: str | None = None,
    ) -> PerturbationArchSelection:
        """Creates and returns an architecture based on perturbation.

        Args:
            profile (BaseProfile): The profile containing the configuration for the
            experiment.
            model_source (str): The source of the model to load. Can be "supernet"
            or a PerturbationArchSelection object.
            model_to_load (str | int | None): The model to load. Can be "last",
            "best", or specific epoch.
            exp_runtime_to_load (str | None): The runtime to load the model from.
            log_with_wandb (bool): Flag to  log with wandb.
            run_name (str): The name of the run.
            src_folder_path (str | None): The source folder path of experiment's run.

        Raises:
            AttributeError: If an illegal model source is provided.
            AssertionError: If the model source is "arch_selection" and model_to_load
            is "best".

        Returns:
            PerturbationArchSelection: The architecture selection object.
        """
        # find pt_configs in the profile
        if model_to_load is not None:
            # self.searchspace is not trained
            last_run = False
            if not exp_runtime_to_load:
                last_run = True

            if model_source == "supernet":
                arch_selection = False
            elif model_source == "arch_selection":
                assert (
                    model_to_load != "best"
                ), "Cannot load best model for arch selection"
                arch_selection = True
            else:
                raise AttributeError("Illegal model source provided")

            self.logger = Logger(
                log_dir="logs",
                exp_name=self.exp_name,
                dataset=str(self.dataset.value),
                search_space=self.searchspace_type.value,
                seed=self.seed,
                runtime=exp_runtime_to_load,
                use_supernet_checkpoint=True,
                arch_selection=arch_selection,
                last_run=last_run,
                custom_log_path=src_folder_path,
            )
        else:
            assert (
                model_source == "supernet"
            ), "Model source can be arch_selection only with loading parameters"
            # use the self.logger
            # that is already in the experiment from last supernet training

        # initialize searchspace handler
        config = profile.get_config()

        self.sampler_str = SamplerType(profile.sampler_type)
        self.perturbator_str = PerturbatorType(profile.perturb_type)
        self.is_partial_connection = profile.is_partial_connection
        self.dropout_p = profile.dropout
        self.edge_normalization = profile.is_partial_connection
        self.entangle_op_weights = profile.entangle_op_weights

        self._init_components(
            self.searchspace_type,
            self.sampler_str,
            self.perturbator_str,
            config=config,
        )

        # Load model from trainer's init
        trainer = self._initialize_configurable_trainer(
            config=config,
            model_to_load=model_to_load,
        )
        search_space_handler = self.search_space_handler
        search_space_handler.adapt_search_space(trainer.model)

        # Load from supernet
        if model_source == "supernet":
            trainer._init_experiment_state(
                search_space_handler=search_space_handler,
                setup_new_run=False,
                warm_epochs=config["trainer"].get(  # type: ignore
                    "lora_warm_epochs", 0
                ),
            )

            # reroute logger
            self.logger = Logger(
                log_dir="logs",
                exp_name=self.exp_name,
                dataset=str(self.dataset.value),
                search_space=self.searchspace_type.value,
                seed=self.seed,
                use_supernet_checkpoint=True,
                arch_selection=True,
            )
            trainer.model_to_load = None
            # TODO: do we need to have the next line?
            trainer.start_epoch = 0
            trainer.logger = self.logger

        trainer._init_experiment_state()

        if log_with_wandb:
            self._init_wandb(run_name, config)

        arch_selector = PerturbationArchSelection(
            trainer,
            config["pt_selection"].get("projection_criteria", "acc"),
            config["pt_selection"].get("projection_interval", 10),
            log_with_wandb=log_with_wandb,
        )
        arch_selector.select_architecture()

        return arch_selector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fine tuning and training searched architectures", add_help=False
    )
    parser.add_argument(
        "--searchspace",
        default="nb201",
        help="search space in (darts, nb201, nb1shot1, tnb101)",
        type=str,
    )
    parser.add_argument(
        "--sampler",
        default="gdas",
        help="samplers in (darts, drnas, gdas, snas)",
        type=str,
    )
    parser.add_argument(
        "--perturbator",
        default="none",
        help="Type of perturbation in (none, random, adversarial)",
        type=str,
    )
    parser.add_argument(
        "--is_partial_connector",
        action="store_true",
        default=False,
        help="Enable/Disable partial connection",
    )
    parser.add_argument(
        "--dropout",
        default=None,
        help="Dropout probability. 0 <= p < 1.",
        type=float,
    )
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--logdir", default="./logs", type=str)
    parser.add_argument("--seed", default=444, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--exp_name", default="test", type=str)
    parser.add_argument(
        "--model_to_load",
        type=validate_model_to_load_value,
        help="if str, could be best, last. If int, load the checkpoint of that epoch",
        default=None,
    )
    parser.add_argument(
        "--use_supernet_checkpoint",
        action="store_true",
        default=False,
        help="If you would like to load from last, best, or a specific epoch",
    )
    parser.add_argument(
        "--runtime",
        # default="",
        help="if you want to start from in a certain runtime",
        type=str,
    )

    parser.add_argument(
        "--oles",
        action="store_true",
        default=False,
        help="freezes weights if it passes the threshold",
    )

    parser.add_argument(
        "--calc_gm_score",
        action="store_true",
        default=False,
        help="calculates gm scores during training the supernet",
    )

    args = parser.parse_args()
    IS_DEBUG_MODE = True
    log_with_wandb = IS_DEBUG_MODE is False

    searchspace = SearchSpaceType(args.searchspace)
    dataset = DatasetType(args.dataset)
    args.epochs = 3

    profile = GDASProfile(
        trainer_preset=searchspace.value,
        epochs=args.epochs,
        is_partial_connection=args.is_partial_connector,
        perturbation=args.perturbator,
        dropout=args.dropout,
        oles=args.oles,
        calc_gm_score=args.calc_gm_score,
    )

    config = profile.get_config()

    if args.runtime:
        assert check_date_format(args.runtime)

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=args.seed,
        log_with_wandb=log_with_wandb,
        debug_mode=IS_DEBUG_MODE,
        exp_name=args.exp_name,
    )

    # trainer = experiment.run_with_profile(
    #     profile,
    #     model_to_load=args.model_to_load,
    # )

    discrete_profile = DiscreteProfile(searchspace.value)

    exp_runtime_to_load = args.runtime if args.runtime != "" else None
    discret_trainer = experiment.train_discrete_model(
        discrete_profile,
        model_to_load=args.model_to_load,
        use_supernet_checkpoint=args.use_supernet_checkpoint,
        exp_runtime_to_load=exp_runtime_to_load,
    )

    if log_with_wandb:
        wandb.finish()  # type: ignore
