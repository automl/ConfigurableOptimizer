from __future__ import annotations

from confopt.enums import DatasetType

from .data import (
    AbstractData,
    CIFAR10Data,
    CIFAR10ModelDataset,
    CIFAR10SupernetDataset,
    CIFAR100Data,
    FGVCAircraftDataset,
    ImageNet16Data,
    ImageNet16120Data,
    SyntheticData,
    TaskonomyClassObjectData,
    TaskonomyClassSceneData,
)


def get_taskonomy_dataset(domain: str) -> type[AbstractData]:
    if domain == "class_object":
        return TaskonomyClassObjectData
    elif domain == "class_scene":  # noqa: RET505
        return TaskonomyClassSceneData
    raise ValueError("Invalid domain for Taskonomy dataset")


def get_dataset(
    dataset: DatasetType,
    domain: str | None,
    root: str,
    cutout: int,
    cutout_length: int,
    train_portion: float = 1.0,
    dataset_kwargs: dict | None = None,
) -> AbstractData:
    dataset_cls: type[AbstractData] = CIFAR10Data
    if dataset == DatasetType.CIFAR10:
        dataset_cls = CIFAR10Data
    elif dataset == DatasetType.CIFAR10_SUPERNET:
        dataset_cls = CIFAR10SupernetDataset
    elif dataset == DatasetType.CIFAR10_MODEL:
        dataset_cls = CIFAR10ModelDataset
    elif dataset == DatasetType.CIFAR100:
        dataset_cls = CIFAR100Data
    elif dataset == DatasetType.IMGNET16:
        dataset_cls = ImageNet16Data
    elif dataset == DatasetType.IMGNET16_120:
        dataset_cls = ImageNet16120Data
    elif dataset == DatasetType.TASKONOMY:
        assert domain is not None, "Domain should be provided for Taskonomy dataset"
        dataset_cls = get_taskonomy_dataset(domain)
    elif dataset == DatasetType.AIRCRAFT:
        dataset_cls = FGVCAircraftDataset
    elif dataset == DatasetType.SYNTHETIC:
        dataset_cls = SyntheticData
    else:
        raise ValueError("Invalid dataset")

    if dataset_kwargs is None:
        dataset_kwargs = {}
    return dataset_cls(
        root=root,
        cutout=cutout,
        cutout_length=cutout_length,
        train_portion=train_portion,
        **dataset_kwargs,
    )


__all__ = [
    "AbstractData",
    "CIFAR10Data",
    "CIFAR100Data",
    "FGVCAircraftDataset",
    "ImageNet16Data",
    "ImageNet16120Data",
    "TaskonomyClassObjectData",
    "TaskonomyClassSceneData",
    "get_dataset",
    "SyntheticData",
]
