##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from __future__ import annotations

import hashlib
import os
import pickle
import sys
from typing import Any

import gdown
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision.transforms import Compose


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()  # noqa: S324
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: dict[str, Any]) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)  # type: ignore


def check_integrity(fpath: str, md5: str | None = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True

    return check_md5(fpath, md5)


class ImageNet16(data.Dataset):
    # http://image-net.org/download-images
    # https://drive.google.com/drive/folders/1NE63Vdo2Nia0V7LK1CdybRLjBFY72w40
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_checksums = [
        ["train_data_batch_1", "27846dcaa50de8e21a7d1a35f30f0e91"],
        ["train_data_batch_2", "c7254a054e0e795c69120a5727050e3f"],
        ["train_data_batch_3", "4333d3df2e5ffb114b05d2ffc19b1e87"],
        ["train_data_batch_4", "1620cdf193304f4a92677b695d70d10f"],
        ["train_data_batch_5", "348b3c2fdbb3940c4e9e834affd3b18d"],
        ["train_data_batch_6", "6e765307c242a1b3d7d5ef9139b48945"],
        ["train_data_batch_7", "564926d8cbf8fc4818ba23d2faac7564"],
        ["train_data_batch_8", "f4755871f718ccb653440b9dd0ebac66"],
        ["train_data_batch_9", "bb6dd660c38c58552125b1a92f86b5d4"],
        ["train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"],
    ]
    valid_checksums = [
        ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
    ]

    drive_file_ids = {
        "train_data_batch_1": "1qd9Fkg7MdIe3MMbHtIJC8eZ8OsWcqPYA",
        "train_data_batch_2": "1pQBJ9exwpSG2E7m6aVvcOlJBGRhVhtg9",
        "train_data_batch_3": "175we9AOjnGam0j4sG5Vn0SHFyBvyv2Ia",
        "train_data_batch_4": "1FNBkOavsAP6Hvi7-41yLZwojdWfPub-R",
        "train_data_batch_5": "1HujB1GyiBjrSdAA0he5kZtkO9WEDAwCn",
        "train_data_batch_6": "1_vaYBQpbP6bx-G0_EiNysohqOJBJ_Ept",
        "train_data_batch_7": "1JwQk4TE21KqvrfvnOfVcdqnEB32ULWTr",
        "train_data_batch_8": "1T00JaN09RlNZPod8dQnF_Xdz0BWtjCWr",
        "train_data_batch_9": "1fB2JYSZRfd8uKfKLBO9P3mn9HWIWtMOH",
        "train_data_batch_10": "19Qvrqt-wi0UOCZBwI_Jw6-bLIXCYcPyl",
        "val_data": "1LQNICeSrwwE2KdDxc9Z9FXmi7N4HKUsA",
    }

    def __init__(
        self,
        root: str,
        train: bool,
        transform: Compose,
        use_num_of_class_only: int | None = None,
    ):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        self.dataset_folder = "imagenet"

        self.download_dataset()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        downloaded_list = self.train_checksums if self.train else self.valid_checksums
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for _i, (file_name, _checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, self.dataset_folder, file_name)

            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)  # type: ignore
        self.data = self.data.transpose((0, 2, 3, 1))  # type: ignore   #convert to HWC
        if use_num_of_class_only is not None:
            assert_error_msg = f"Invalid use_num_of_class_only {use_num_of_class_only}"
            assert isinstance(use_num_of_class_only, int), assert_error_msg
            assert use_num_of_class_only > 0, assert_error_msg
            assert use_num_of_class_only < 1000, assert_error_msg

            new_data, new_targets = [], []
            for datapoint, target in zip(self.data, self.targets):
                if 1 <= target <= use_num_of_class_only:
                    new_data.append(datapoint)
                    new_targets.append(target)
            self.data = new_data
            self.targets = new_targets
        #    self.mean.append(entry['mean'])
        # self.mean = np.vstack(self.mean).reshape(-1, 3, 16, 16)
        # self.mean = np.mean(np.mean(np.mean(self.mean, axis=0), axis=1), axis=1)
        # print ('Mean : {:}'.format(self.mean))
        # temp      = self.data - np.reshape(self.mean, (1, 1, 1, 3))
        # std_data  = np.std(temp, axis=0)
        # std_data  = np.mean(np.mean(std_data, axis=0), axis=0)
        # print ('Std  : {:}'.format(std_data))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self.data)} images,"
            + f" {len(set(self.targets))} classes)"
        )

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index] - 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_checksums + self.valid_checksums:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.dataset_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download_dataset(self) -> None:
        root = self.root
        dataset_folder = self.dataset_folder
        folder_path = os.path.join(root, dataset_folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for file_name, file_id in self.drive_file_ids.items():
            file_path = os.path.join(root, dataset_folder, file_name)
            if not os.path.exists(file_path):
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    output=file_path,
                )


"""
if __name__ == '__main__':
  train = ImageNet16('datasets/imagenet', True , None)
  valid = ImageNet16('datasets/imagenet', False, None)

  print ( len(train) )
  print ( len(valid) )
  image, label = train[111]
  trainX = ImageNet16('~/.torch/cifar.python/ImageNet16', True , None, 200)
  validX = ImageNet16('~/.torch/cifar.python/ImageNet16', False , None, 200)
  print ( len(trainX) )
  print ( len(validX) )
"""
