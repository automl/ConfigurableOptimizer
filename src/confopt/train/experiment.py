from __future__ import annotations

import argparse
from collections import namedtuple
from enum import Enum
import random
from typing import Callable

from fvcore.common.checkpoint import Checkpointer
import numpy as np
import torch
from torch.backends import cudnn
import wandb

from confopt.dataset import (
    CIFAR10Data,
    CIFAR100Data,
    ImageNet16Data,
    ImageNet16120Data,
)
from confopt.oneshot.archsampler import (
    DARTSSampler,
    DRNASSampler,
    GDASSampler,
    SNASSampler,
)
from confopt.oneshot.dropout import Dropout
from confopt.oneshot.partial_connector import PartialConnector
from confopt.oneshot.perturbator import SDARTSPerturbator
from confopt.profiles import (
    DiscreteProfile,
    GDASProfile,
    ProfileConfig,
)
from confopt.searchspace import (
    DARTSSearchSpace,
    NASBench1Shot1SearchSpace,
    NASBench201SearchSpace,
    TransNASBench101SearchSpace,
)
from confopt.train import ConfigurableTrainer, DiscreteTrainer, Profile
from confopt.utils import Logger

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# TODO Change this to real data
ADVERSERIAL_DATA = (
    torch.randn(2, 3, 32, 32).to(DEVICE),
    torch.randint(0, 9, (2,)).to(DEVICE),
)


class SearchSpaceType(Enum):
    DARTS = "darts"
    NB201 = "nb201"
    NB1SHOT1 = "nb1shot1"
    TNB101 = "tnb101"


class SamplerType(Enum):
    DARTS = "darts"
    DRNAS = "drnas"
    GDAS = "gdas"
    SNAS = "snas"


class PerturbatorType(Enum):
    RANDOM = "random"
    ADVERSERIAL = "adverserial"
    NONE = "none"


PERTUB_DEFAULT_EPSILON = 0.03
PERTUBRATOR_NONE = PerturbatorType("none")


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMGNET16 = "imgnet16"
    IMGNET16_120 = "imgnet16_120"


N_CLASSES = {
    DatasetType.CIFAR10: 10,
    DatasetType.CIFAR100: 100,
    DatasetType.IMGNET16_120: 120,
}


class CriterionType(Enum):
    CROSS_ENTROPY = "cross_entropy"


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ASGD = "asgd"


class Experiment:
    def __init__(
        self,
        search_space: SearchSpaceType,
        dataset: DatasetType,
        seed: int,
        is_wandb_log: bool = False,
        debug_mode: bool = False,
        exp_name: str = "test",
    ) -> None:
        self.search_space_str = search_space
        self.dataset_str = dataset
        self.seed = seed
        self.is_wandb_log = is_wandb_log
        self.debug_mode = debug_mode
        self.exp_name = exp_name

    def set_seed(self, rand_seed: int) -> None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        cudnn.benchmark = True
        torch.manual_seed(rand_seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(rand_seed)

    def run_with_profile(
        self,
        profile: ProfileConfig,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
    ) -> ConfigurableTrainer:
        config = profile.get_config()

        assert hasattr(profile, "sampler_type")
        self.sampler_str = SamplerType(profile.sampler_type)
        self.perturbator_str = PerturbatorType(profile.perturb_type)
        self.is_partial_connection = profile.is_partial_connection
        self.dropout = profile.dropout
        self.edge_normalization = profile.is_partial_connection
        assert sum([load_best_model, load_saved_model, (start_epoch > 0)]) <= 1
        return self.runner(
            config,
            start_epoch,
            load_saved_model,
            load_best_model,
        )

    def runner(
        self,
        config: dict | None = None,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
    ) -> ConfigurableTrainer:
        assert sum([load_best_model, load_saved_model, (start_epoch > 0)]) <= 1

        self.set_seed(self.seed)

        if load_saved_model or load_best_model or start_epoch > 0:
            self.logger = Logger(
                log_dir="logs",
                seed=self.seed,
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
                last_run=True,
            )
        else:
            self.logger = Logger(
                log_dir="logs",
                seed=self.seed,
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
            )

        self._enum_to_objects(
            self.search_space_str,
            self.sampler_str,
            self.perturbator_str,
            config=config,
        )
        if self.is_wandb_log:
            wandb.init(  # type: ignore
                project=(
                    config.get("project_name", "Configurable_Optimizer")
                    if config is not None
                    else "Configurable_Optimizer"
                ),
                config=config,
            )

        Arguments = namedtuple(  # type: ignore
            "Configure", " ".join(config["trainer"].keys())  # type: ignore
        )
        arg_config = Arguments(**config["trainer"])  # type: ignore

        criterion = self._get_criterion(
            criterion_str=arg_config.criterion  # type: ignore
        )

        data = self._get_dataset(self.dataset_str)(
            root="datasets",
            cutout=arg_config.cutout,  # type: ignore
            train_portion=arg_config.train_portion,  # type: ignore
        )

        model = self.search_space

        w_optimizer = self._get_optimizer(arg_config.optim)(  # type: ignore
            model.model_weight_parameters(),
            arg_config.lr,  # type: ignore
            **config["trainer"].get("optim_config", {}),  # type: ignore
        )

        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=w_optimizer,
            T_max=float(arg_config.epochs),  # type: ignore
            eta_min=arg_config.learning_rate_min,  # type: ignore
        )

        if self.edge_normalization and hasattr(model, "beta_parameters"):
            arch_optimizer = self._get_optimizer(arg_config.arch_optim)(  # type: ignore
                [*model.arch_parameters, *model.beta_parameters],
                lr=config["trainer"].get("arch_lr", 0.001),  # type: ignore
                **config["trainer"].get("arch_optim_config", {}),  # type: ignore
            )
        else:
            arch_optimizer = self._get_optimizer(arg_config.arch_optim)(  # type: ignore
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
            batch_size=arg_config.batch_size,  # type: ignore
            use_data_parallel=arg_config.use_data_parallel,  # type: ignore
            load_saved_model=load_saved_model,
            load_best_model=load_best_model,
            start_epoch=start_epoch,
            checkpointing_freq=arg_config.checkpointing_freq,  # type: ignore
            epochs=arg_config.epochs,  # type: ignore
            debug_mode=self.debug_mode,
        )

        trainer.train(
            profile=self.profile,  # type: ignore
            epochs=arg_config.epochs,  # type: ignore
            is_wandb_log=self.is_wandb_log,
            lora_warm_epochs=config["trainer"].get(  # type: ignore
                "lora_warm_epochs", 0
            ),
        )

        return trainer

    def _enum_to_objects(
        self,
        search_space_enum: SearchSpaceType,
        sampler_enum: SamplerType,
        perturbator_enum: PerturbatorType,
        config: dict | None = None,
    ) -> None:
        if config is None:
            config = {}  # type : ignore
        self.set_search_space(search_space_enum, config.get("search_space", {}))
        self.set_sampler(sampler_enum, config.get("sampler", {}))

        self.set_perturbator(perturbator_enum, config.get("perturbator", {}))

        self.set_partial_connector(config.get("partial_connector", {}))
        self.set_dropout(config.get("dropout", {}))
        self.set_profile(config)

    def set_search_space(
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

    def set_sampler(
        self,
        sampler: SamplerType,
        config: dict,
    ) -> None:
        arch_params = self.search_space.arch_parameters
        if sampler == SamplerType.DARTS:
            self.sampler = DARTSSampler(**config, arch_parameters=arch_params)
        elif sampler == SamplerType.DRNAS:
            self.sampler = DRNASSampler(**config, arch_parameters=arch_params)
        elif sampler == SamplerType.GDAS:
            self.sampler = GDASSampler(**config, arch_parameters=arch_params)
        elif sampler == SamplerType.SNAS:
            self.sampler = SNASSampler(**config, arch_parameters=arch_params)

    def set_perturbator(
        self,
        petubrator_enum: PerturbatorType,
        pertub_config: dict,
    ) -> None:
        if petubrator_enum != PerturbatorType.NONE:
            self.perturbator = SDARTSPerturbator(
                **pertub_config,
                search_space=self.search_space,
                arch_parameters=self.search_space.arch_parameters,
                attack_type=petubrator_enum.value,
            )
        else:
            self.perturbator = None

    def set_partial_connector(self, config: dict) -> None:
        if self.is_partial_connection:
            self.partial_connector = PartialConnector(**config)
        else:
            self.partial_connector = None

    def set_dropout(self, config: dict) -> None:
        if self.dropout is not None:
            self.dropout = Dropout(**config)
        else:
            self.dropout = None

    def set_profile(self, config: dict) -> None:
        assert self.sampler is not None

        self.profile = Profile(
            sampler=self.sampler,
            edge_normalization=self.edge_normalization,
            partial_connector=self.partial_connector,
            perturbation=self.perturbator,
            dropout=self.dropout,
            lora_configs=config.get("lora", None),
        )

    def _get_dataset(self, dataset: DatasetType) -> Callable | None:
        if dataset == DatasetType.CIFAR10:
            return CIFAR10Data
        elif dataset == DatasetType.CIFAR100:  # noqa: RET505
            return CIFAR100Data
        elif dataset == DatasetType.IMGNET16:
            return ImageNet16Data
        elif dataset == DatasetType.IMGNET16_120:
            return ImageNet16120Data
        return None

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

    def run_discrete_model_with_profile(
        self,
        profile: DiscreteProfile,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
    ) -> DiscreteTrainer:
        config = profile.get_trainer_config()

        return self.run_discrete_model(
            config,
            start_epoch,
            load_saved_model,
            load_best_model,
        )

    def run_discrete_model(
        self,
        arg_config: dict | None = None,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
    ) -> DiscreteTrainer:
        assert sum([load_best_model, load_saved_model, (start_epoch > 0)]) <= 1

        self.set_seed(self.seed)

        if not hasattr(self, "search_space"):
            self.set_search_space(
                self.search_space_str,
                arg_config.get("search_space", {}),  # type: ignore
            )

        if load_saved_model or load_best_model or start_epoch > 0:
            self.logger = Logger(
                log_dir="logs",
                seed=self.seed,
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
                last_run=True,
            )
        else:
            self.logger = Logger(
                log_dir="logs",
                seed=self.seed,
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
                last_run=False,
            )

        if "search_space" in arg_config:  # type: ignore
            searched_arch_params = self.search_space.arch_parameters
            self.set_search_space(
                self.search_space_str, arg_config.get("search_space")  # type: ignore
            )
            self.search_space.set_arch_parameters(searched_arch_params)

        Arguments = namedtuple(  # type: ignore
            "Configure", " ".join(arg_config.keys())  # type: ignore
        )
        arg_config = Arguments(**arg_config)  # type: ignore

        data = self._get_dataset(self.dataset_str)(
            root="datasets",
            cutout=arg_config.cutout,  # type: ignore
            train_portion=arg_config.train_portion,  # type: ignore
        )

        model = self.search_space
        w_optimizer = self._get_optimizer(arg_config.optim)(  # type: ignore
            model.model_weight_parameters(),
            arg_config.lr,  # type: ignore
            **arg_config.optim_config,  # type: ignore
        )

        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=w_optimizer,
            T_max=float(arg_config.epochs),  # type: ignore
            eta_min=arg_config.learning_rate_min,  # type: ignore
        )

        criterion = self._get_criterion(
            criterion_str=arg_config.criterion  # type: ignore
        )

        trainer = DiscreteTrainer(
            model=model,
            data=data,
            model_optimizer=w_optimizer,
            scheduler=w_scheduler,
            criterion=criterion,
            logger=self.logger,
            batch_size=arg_config.batch_size,  # type: ignore
            use_data_parallel=arg_config.use_data_parallel,  # type: ignore
            load_saved_model=load_saved_model,
            load_best_model=load_best_model,
            start_epoch=start_epoch,
            checkpointing_freq=arg_config.checkpointing_freq,  # type: ignore
            epochs=arg_config.epochs,  # type: ignore
        )

        trainer.train(
            epochs=arg_config.epochs,  # type: ignore
            is_wandb_log=self.is_wandb_log,
        )

        trainer.test(is_wandb_log=self.is_wandb_log)

        return trainer

    def initialize_from_last_run(
        self,
        profile_config: ProfileConfig,
        last_search_run_time: str = "NOT_VALID",
    ) -> None:
        if last_search_run_time != "NOT_VALID":
            run_time = last_search_run_time
            last_run_logger = Logger(
                log_dir="logs",
                seed=self.seed,
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
                run_time=run_time,
            )
        else:
            last_run_logger = Logger(
                log_dir="logs",
                seed=self.seed,
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
                last_run=True,
            )
        config = profile_config.get_config()
        self.set_search_space(self.search_space_str, config.get("search_space"))
        checkpoint_dir = last_run_logger.path(mode="checkpoints")
        checkpointer = Checkpointer(
            model=self.search_space,
            save_dir=checkpoint_dir,
            save_to_disk=True,
        )
        last_info = last_run_logger.path("last_checkpoint")
        info = checkpointer._load_file(f=last_info)
        self.search_space.load_state_dict(info["model"])
        print("=> loading SEARCH checkpoint of the last-info", str(last_info))
        last_run_logger.close()


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
        help="Type of perturbation in (none, random, adverserial)",
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
    parser.add_argument("--exp_name", default="test", type=str)
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        default=False,
        help="Load the best model found from the previous run",
    )
    parser.add_argument(
        "--load_saved_model",
        action="store_true",
        default=False,
        help="Load the last saved model in the last run of training them",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        help="Specify the start epoch to continue the training of the model from the \
        previous run",
        type=int,
    )
    args = parser.parse_args()
    IS_DEBUG_MODE = True
    is_wandb_log = IS_DEBUG_MODE is False

    searchspace = SearchSpaceType(args.searchspace)
    dataset = DatasetType(args.dataset)

    profile = GDASProfile(
        epochs=args.epochs,
        is_partial_connection=args.is_partial_connector,
        perturbation=args.perturbator,
        dropout=args.dropout,
    )

    config = profile.get_config()

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=args.seed,
        is_wandb_log=is_wandb_log,
        debug_mode=IS_DEBUG_MODE,
        exp_name=args.exp_name,
    )

    trainer = experiment.run_with_profile(
        profile,
        start_epoch=args.start_epoch,
        load_saved_model=args.load_saved_model,
        load_best_model=args.load_best_model,
    )

    profile = DiscreteProfile()
    discret_trainer = experiment.run_discrete_model_with_profile(
        profile,
        start_epoch=args.start_epoch,
        load_saved_model=args.load_saved_model,
        load_best_model=args.load_best_model,
    )

    if is_wandb_log:
        wandb.finish()  # type: ignore
