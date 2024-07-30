from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.transforms import Compose

import confopt.utils.distributed as dist_utils

from .imgnet16 import ImageNet16

DS = Tuple[Union[Dataset, None], Union[Sampler, None]]


class CUTOUT:
    def __init__(self, length: int):
        self.length = length

    def __repr__(self) -> str:
        return "{name}(length={length})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)  # type: ignore
        mask = mask.expand_as(img)  # type: ignore
        img *= mask
        return img


class AbstractData(ABC):
    def __init__(self, root: str, train_portion: float = 1.0) -> None:
        self.root = root
        self.train_portion = train_portion
        if train_portion == 1:
            self.shuffle = True
        else:
            self.shuffle = False

    @abstractmethod
    def build_datasets(self) -> tuple[DS, DS, DS]:
        ...

    @abstractmethod
    def get_transforms(self) -> tuple[Compose, Compose]:
        ...

    @abstractmethod
    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        ...

    def get_dataloaders(
        self,
        batch_size: int = 64,
        n_workers: int = 2,
        use_distributed_sampler: bool = False,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader]:
        (
            (train_data, train_sampler),
            (val_data, val_sampler),
            (test_data, test_sampler),
        ) = self.build_datasets()

        if use_distributed_sampler:
            rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
            choose_sampler = (
                lambda data, sampler: sampler if sampler is not None else data
            )
            train_sampler = DistributedSampler(
                choose_sampler(train_data, train_sampler),
                num_replicas=world_size,
                rank=rank,
            )
            val_sampler = DistributedSampler(
                choose_sampler(val_data, val_sampler),
                num_replicas=world_size,
                rank=rank,
            )
            test_sampler = DistributedSampler(
                choose_sampler(test_data, test_sampler),
                num_replicas=world_size,
                rank=rank,
            )

        train_queue = DataLoader(
            train_data,  # type: ignore
            batch_size=batch_size,
            pin_memory=True,
            sampler=train_sampler,
            num_workers=n_workers,
            shuffle=self.shuffle,
        )
        if val_data is not None:
            valid_queue = DataLoader(
                val_data,
                batch_size=batch_size,
                pin_memory=True,
                sampler=val_sampler,
                num_workers=n_workers,
            )
        else:
            valid_queue = None

        test_queue = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=n_workers,
            sampler=test_sampler,
        )

        return train_queue, valid_queue, test_queue


class CIFARData(AbstractData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, train_portion)
        self.cutout = cutout

    def get_transforms(self) -> tuple[Compose, Compose]:
        lists = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),  # type: ignore
        ]

        if self.cutout > 0:
            lists += [CUTOUT(self.cutout)]
        train_transform = transforms.Compose(lists)

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),  # type: ignore
            ]
        )

        return train_transform, test_transform

    def build_datasets(self) -> tuple[DS, DS, DS]:
        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets(
            self.root, train_transform, test_transform
        )

        if self.train_portion < 1:
            num_train = len(train_data)  # type: ignore
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split]
            )
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]
            )
            return (
                (train_data, train_sampler),
                (train_data, val_sampler),
                (test_data, None),
            )

        return (train_data, None), (None, None), (test_data, None)


class ImageNetData(AbstractData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, train_portion)
        self.cutout = cutout
        self.mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        self.std = [x / 255 for x in [63.22, 61.26, 65.09]]

    def get_transforms(self) -> tuple[Compose, Compose]:
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]
        if self.cutout > 0:
            lists += [CUTOUT(self.cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )

        return train_transform, test_transform

    def build_datasets(self) -> tuple[DS, DS, DS]:
        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets(
            self.root, train_transform, test_transform
        )

        if self.train_portion > 0:
            num_train = len(train_data)  # type: ignore
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split]
            )
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]
            )
            return (
                (train_data, train_sampler),
                (train_data, val_sampler),
                (test_data, None),
            )

        return (train_data, None), (None, None), (test_data, None)


class CIFAR10Data(CIFARData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, cutout, train_portion)
        self.mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255 for x in [63.0, 62.1, 66.7]]

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )

        assert len(train_data) == 50000
        assert len(test_data) == 10000
        return train_data, test_data


class CIFAR100Data(CIFARData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, cutout, train_portion)
        self.mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255 for x in [63.0, 62.1, 66.7]]

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )

        assert len(train_data) == 50000
        assert len(test_data) == 10000
        return train_data, test_data


class ImageNet16Data(ImageNetData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167
        assert len(test_data) == 50000
        return train_data, test_data


class ImageNet16120Data(ImageNetData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700
        assert len(test_data) == 6000

        return train_data, test_data
