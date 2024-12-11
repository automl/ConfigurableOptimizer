from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Callable, Tuple, Union

import numpy as np
from skimage import io
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.transforms import Compose

from confopt.dataset import load_ops
import confopt.utils.distributed as dist_utils

from .imgnet16 import ImageNet16

DS = Tuple[Union[Dataset, None], Union[Sampler, None]]
DOMAIN_DATA_SOURCE = {
    "rgb": ("rgb", "png", "rgb"),
    "autoencoder": ("rgb", "png", "autoencoder"),
    "class_object": ("class_object", "npy", "class_object"),
    "class_scene": ("class_scene", "npy", "class_places"),
    "normal": ("normal", "png", "normal"),
    "room_layout": ("room_layout", "npy", "room_layout"),
    "segmentsemantic": ("segmentsemantic", "png", "segmentsemantic"),
    "jigsaw": ("rgb", "png", "jigsaw"),
}
TASKONOMY_TRAIN_FILENAMES_FINAL5K = "train_filenames_final5k.json"
TASKONOMY_TEST_FILENAMES_FINAL5K = "test_filenames_final5k.json"


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
            choose_sampler = lambda data, sampler: (
                sampler if sampler is not None else data
            )
            train_sampler = DistributedSampler(
                choose_sampler(train_data, train_sampler),
                num_replicas=world_size,
                rank=rank,
                shuffle=self.shuffle,
            )
            if val_data is not None:
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

        test_queue = DataLoader(
            test_data,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=n_workers,
            sampler=test_sampler,
        )

        return train_queue, valid_queue, test_queue


class CIFARData(AbstractData):
    def __init__(
        self, root: str, cutout: int, cutout_length: int, train_portion: float = 1.0
    ):
        super().__init__(root, train_portion)
        self.cutout = cutout
        self.cutout_length = cutout_length

    def get_transforms(self) -> tuple[Compose, Compose]:
        lists = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),  # type: ignore
        ]

        if self.cutout > 0:
            lists += [CUTOUT(self.cutout_length)]
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
    def __init__(
        self, root: str, cutout: int, cutout_length: int, train_portion: float = 1.0
    ):
        super().__init__(root, train_portion)
        self.cutout = cutout
        self.cutout_length = cutout_length
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
            lists += [CUTOUT(self.cutout_length)]
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
    def __init__(
        self, root: str, cutout: int, cutout_length: int, train_portion: float = 1.0
    ):
        super().__init__(root, cutout, cutout_length, train_portion)
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
    def __init__(
        self, root: str, cutout: int, cutout_length: int, train_portion: float = 1.0
    ):
        super().__init__(root, cutout, cutout_length, train_portion)
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
    def __init__(
        self, root: str, cutout: int, cutout_length: int, train_portion: float = 1.0
    ):
        super().__init__(root, cutout, cutout_length, train_portion)

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167
        assert len(test_data) == 50000
        return train_data, test_data


class ImageNet16120Data(ImageNetData):
    def __init__(
        self, root: str, cutout: int, cutout_length: int, train_portion: float = 1.0
    ):
        super().__init__(root, cutout, cutout_length, train_portion)

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700
        assert len(test_data) == 6000

        return train_data, test_data


class TaskonomyDataset(Dataset):
    def __init__(
        self,
        templates: list[str],
        dataset_dir: str,
        domain: str,
        target_load_fn: Callable,
        target_load_kwargs: dict,
        transform: Callable | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.domain = domain
        self.label_type = DOMAIN_DATA_SOURCE[domain][1]
        self.all_templates = (
            templates  # load_ops.get_all_templates(dataset_dir, json_path)
        )
        self.target_load_kwargs = target_load_kwargs
        self.target_load_fn = target_load_fn
        self.transform = transform

    def __len__(self) -> int:
        return len(self.all_templates)

    def __getitem__(
        self, idx: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | np.ndarray]:
        try:
            if torch.is_tensor(idx):
                idx = idx.item()  # type: ignore
            template = os.path.join(self.dataset_dir, self.all_templates[idx])
            image = io.imread(".".join([template.format(domain="rgb"), "png"]))
            label = self.get_label(template)
            sample = {"image": image, "label": label}
            if self.transform:
                sample = self.transform(sample)
        except Exception as err:  # noqa: BLE001
            template = os.path.join(self.dataset_dir, self.all_templates[idx])
            raise Exception(
                f"Error for img {'.'.join([template.format(domain='rgb'), 'png'])}"
            ) from err
        return sample["image"], sample["label"]

    def get_label(self, template: str) -> np.ndarray:
        template = template.replace("{domain}", "{domain_task}", 1)
        label_path = ".".join(
            [
                template.format(
                    domain_task=DOMAIN_DATA_SOURCE[self.domain][0],
                    domain=DOMAIN_DATA_SOURCE[self.domain][2],
                ),
                DOMAIN_DATA_SOURCE[self.domain][1],
            ]
        )
        label = self.target_load_fn(label_path, **self.target_load_kwargs)
        return label


class TaskonomyData(AbstractData):
    def __init__(
        self,
        train_portion: float,
        dataset_dir: str,
        domain: str,
        target_load_fn: Callable,
        num_classes: int,
        target_dim: int,
        data_split_dir: str,
    ) -> None:
        super().__init__(dataset_dir, train_portion)
        self.dataset_dir = dataset_dir
        self.domain = domain
        self.label_type = DOMAIN_DATA_SOURCE[domain][1]
        self.target_load_fn = target_load_fn
        self.num_classes = num_classes
        self.target_dim = target_dim
        self.data_split_dir = data_split_dir

    def load_datasets(
        self,
        # TODO: Remove the unused argument
        root: str,  # noqa: ARG002
        train_transform: load_ops.Compose,
        test_transform: load_ops.Compose,
    ) -> tuple[Dataset, Dataset]:
        train_templates = load_ops.get_all_templates(
            self.dataset_dir,
            os.path.join(self.data_split_dir, TASKONOMY_TRAIN_FILENAMES_FINAL5K),
        )
        target_load_kwargs = {
            "selected": self.target_dim < self.num_classes,
            "final5k": "final5k" in TASKONOMY_TRAIN_FILENAMES_FINAL5K,
        }
        train_data = TaskonomyDataset(
            templates=train_templates,
            dataset_dir=self.dataset_dir,
            domain=self.domain,
            target_load_fn=self.target_load_fn,
            target_load_kwargs=target_load_kwargs,
            transform=train_transform,
        )
        test_templates = load_ops.get_all_templates(
            self.dataset_dir,
            os.path.join(self.data_split_dir, TASKONOMY_TEST_FILENAMES_FINAL5K),
        )
        test_data = TaskonomyDataset(
            templates=test_templates,
            dataset_dir=self.dataset_dir,
            domain=self.domain,
            target_load_fn=self.target_load_fn,
            target_load_kwargs=target_load_kwargs,
            transform=test_transform,
        )
        return train_data, test_data

    def get_transforms(self) -> tuple[load_ops.Compose, load_ops.Compose]:
        normal_params = {
            "mean": [0.5224, 0.5222, 0.5221],
            "std": [0.2234, 0.2235, 0.2236],
            "inplace": False,
        }
        train_transform = load_ops.Compose(
            self.domain,
            [
                load_ops.ToPILImage(),
                load_ops.Resize([256, 256]),  # (1024, 1024)
                load_ops.RandomHorizontalFlip(0.5),
                load_ops.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                load_ops.ToTensor(),
                load_ops.Normalize(**normal_params),
            ],
        )
        test_transform = load_ops.Compose(
            self.domain,
            [
                load_ops.ToPILImage(),
                load_ops.Resize([256, 256]),
                load_ops.ToTensor(),
                load_ops.Normalize(**normal_params),
            ],
        )
        return train_transform, test_transform

    def build_datasets(self) -> tuple[DS, DS, DS]:
        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets("", train_transform, test_transform)

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


class TaskonomyClassObjectData(TaskonomyData):
    def __init__(
        self,
        root: str = "datasets",
        dataset_dir: str = "taskonomydata_mini",
        train_portion: float = 0.5,
        # TODO: Remove the unused argument
        cutout: int = -1,  # noqa: ARG002
        cutout_length: int = 16,  # noqa: ARG002
        num_classes: int = 1000,
        target_dim: int = 75,
        data_split_dir: str = "final5K_splits",
    ) -> None:
        super().__init__(
            train_portion=train_portion,
            dataset_dir=os.path.join(root, dataset_dir),
            domain="class_object",
            target_load_fn=load_ops.load_class_object_logits,
            num_classes=num_classes,
            target_dim=target_dim,
            data_split_dir=os.path.join(root, dataset_dir, data_split_dir),
        )


class TaskonomyClassSceneData(TaskonomyData):
    def __init__(
        self,
        root: str = "datasets",
        dataset_dir: str = "taskonomydata_mini",
        train_portion: float = 0.5,
        # TODO: Remove the unused argument
        cutout: int = -1,  # noqa: ARG002
        cutout_length: int = 16,  # noqa: ARG002
        num_classes: int = 365,
        target_dim: int = 47,
        data_split_dir: str = "final5K_splits",
    ) -> None:
        super().__init__(
            dataset_dir=os.path.join(root, dataset_dir),
            train_portion=train_portion,
            domain="class_scene",
            target_load_fn=load_ops.load_class_scene_logits,
            num_classes=num_classes,
            target_dim=target_dim,
            data_split_dir=os.path.join(root, dataset_dir, data_split_dir),
        )
