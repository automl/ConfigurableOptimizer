from __future__ import annotations

import collections

# import transforms3d
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T  # noqa: N812
from torchvision.transforms import functional as F  # noqa: N812

from confopt.dataset.synset import synset as raw_synset

Sequence = collections.abc.Sequence
Iterable = collections.abc.Iterable


lib_dir = (Path(__file__).parent / "..").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

##############
# Helper fns #
##############


def read_json(file_path: Path | str) -> dict[str, Any]:
    current_path = os.getcwd()
    file_path = os.path.join(current_path, file_path)
    with open(file_path) as json_file:
        data: dict[str, Any] = json.load(json_file)
    return data


#######################
# Image transform fns #
#######################

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",  # type: ignore[attr-defined]
    Image.BILINEAR: "PIL.Image.BILINEAR",  # type: ignore[attr-defined]
    Image.BICUBIC: "PIL.Image.BICUBIC",  # type: ignore[attr-defined]
    Image.LANCZOS: "PIL.Image.LANCZOS",  # type: ignore[attr-defined]
    Image.HAMMING: "PIL.Image.HAMMING",  # type: ignore[attr-defined]
    Image.BOX: "PIL.Image.BOX",  # type: ignore[attr-defined]
}

TASK_NAMES = [
    "autoencoder",
    "class_object",
    "class_scene",
    "normal",
    "jigsaw",
    "room_layout",
    "segmentsemantic",
]


class Compose(T.Compose):
    def __init__(self, task_name: str, transforms: list) -> None:
        self.transforms = transforms
        self.task_name = task_name
        assert task_name in TASK_NAMES

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            sample = t(sample, self.task_name)
        return sample


class Cutout:
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def cutout(self, img: torch.Tensor) -> torch.Tensor:
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
        mask = mask.expand_as(img)  # type: ignore[attr-defined]
        img *= mask
        return img

    def __repr__(self) -> str:
        return "{name}(length={length})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def __call__(self, sample: dict, task_name: str) -> dict:
        image, label = sample["image"], sample["label"]
        if task_name in ["class_object", "class_scene", "room_layout"]:
            return {"image": self.cutout(image), "label": label}

        raise ValueError(f"task name {task_name} not available!")


class Resize(T.Resize):
    def __init__(
        self,
        input_size: int | list,
        target_size: int | list | None = None,
        interpolation: F.InterpolationMode = (
            Image.BILINEAR  # type: ignore[attr-defined]
        ),
    ) -> None:
        assert isinstance(input_size, int) or (
            isinstance(input_size, Iterable) and len(input_size) == 2
        )
        if target_size:
            assert isinstance(target_size, int) or (
                isinstance(target_size, Iterable) and len(target_size) == 2
            )
        self.input_size = input_size
        self.target_size = target_size if target_size is not None else input_size
        self.interpolation = interpolation

    def __call__(self, sample: dict, task_name: str) -> dict:
        """Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        image, label = sample["image"], sample["label"]

        if task_name in ["autoencoder", "normal"]:
            return {
                "image": F.resize(
                    image, self.input_size, self.interpolation  # type: ignore
                ),
                "label": F.resize(label, self.target_size, self.interpolation),
            }  # type: ignore
        elif task_name == "segmentsemantic":  # noqa: RET505
            return {
                "image": F.resize(
                    image, self.input_size, self.interpolation  # type: ignore
                ),
                "label": F.resize(
                    label, self.target_size, Image.NEAREST  # type: ignore[attr-defined]
                ),
            }  # type: ignore
        elif task_name in ["class_object", "class_scene", "room_layout", "jigsaw"]:
            return {
                "image": F.resize(
                    image, self.input_size, self.interpolation  # type: ignore
                ),
                "label": label,
            }

        raise ValueError(f"task name {task_name} not available!")

    def __repr__(self) -> str:
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return (
            self.__class__.__name__
            + "(input_size={}, target_size={}, interpolation={})".format(
                self.input_size, self.target_size, interpolate_str
            )
        )


class ToPILImage(T.ToPILImage):
    def __init__(self, mode: str | None = None) -> None:
        self.mode = mode

    def __call__(self, sample: dict, task_name: str) -> dict:
        """.

        Args:
            sample (Tensor or numpy.ndarray): {'image': image_to_convert,
                'label': npy/png label}
            task_name (str): task name in ['autoencoder', 'class_object']

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        image, label = sample["image"], sample["label"]
        if task_name in ["autoencoder", "normal"] or task_name == "segmentsemantic":
            return {
                "image": F.to_pil_image(image, self.mode),
                "label": F.to_pil_image(label, self.mode),
            }
        elif task_name in [  # noqa: RET505
            "class_object",
            "class_scene",
            "room_layout",
            "jigsaw",
        ]:
            return {"image": F.to_pil_image(image, self.mode), "label": label}
        else:
            raise ValueError(f"task name {task_name} not available!")


class ToTensor(T.ToTensor):
    def __init__(self, new_scale: Image.Image | np.ndarray | None = None) -> None:
        self.new_scale = new_scale

    def __call__(self, sample: dict, task_name: str) -> dict:
        image, label = sample["image"], sample["label"]
        if task_name in ["autoencoder", "normal"]:
            image = F.to_tensor(image).float()
            label = F.to_tensor(label).float()
            if self.new_scale:
                min_val, max_val = self.new_scale  # type: ignore
                label *= max_val - min_val
                label += min_val
        elif task_name == "segmentsemantic":
            image = F.to_tensor(image).float()
            label = torch.tensor(np.array(label), dtype=torch.uint8)
        elif task_name in ["class_object", "class_scene", "room_layout"]:
            image = F.to_tensor(image).float()
            label = torch.FloatTensor(label)
        else:
            raise ValueError(f"task name {task_name} not available!")
        if self.new_scale:
            min_val, max_val = self.new_scale  # type: ignore
            image *= max_val - min_val
            image += min_val
        return {"image": image, "label": label}


class Normalize(T.Normalize):
    def __init__(
        self, mean: list[float], std: list[float], inplace: bool = False
    ) -> None:
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample: dict, task_name: str) -> dict:
        tensor, label = sample["image"], sample["label"]
        # if task_name in ["segmentsemantic"]:
        #     raise TypeError("['segmentsemantic'] cannot apply normalize")
        if task_name in ["autoencoder"]:
            return {
                "image": F.normalize(tensor, self.mean, self.std, self.inplace),
                "label": F.normalize(label, self.mean, self.std, self.inplace),
            }
        elif task_name in [  # noqa: RET505
            "normal",
            "segmentsemantic",
        ] or task_name in [
            "class_object",
            "class_scene",
            "room_layout",
        ]:
            return {
                "image": F.normalize(tensor, self.mean, self.std, self.inplace),
                "label": label,
            }
        else:
            raise ValueError(f"task name {task_name} not available!")


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """.
    Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict, task_name: str) -> dict:
        random_num = random.random()
        image, label = sample["image"], sample["label"]
        if random_num < self.p:
            if task_name in ["autoencoder", "segmentsemantic"]:
                return {"image": F.hflip(image), "label": F.hflip(label)}
            elif task_name in ["class_object", "class_scene", "jigsaw"]:  # noqa: RET505
                return {"image": F.hflip(image), "label": label}
            elif task_name in ["normal", "room_layout"]:
                raise ValueError(f"task name {task_name} not available!")
            else:
                raise ValueError(f"task name {task_name} not available!")
        else:
            return sample


class RandomGrayscale(T.RandomGrayscale):
    """Randomly convert image to grayscale with a probability of p (default 0.1).

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def __call__(self, sample: dict, task_name: str) -> dict:
        """Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        image, label = sample["image"], sample["label"]
        num_output_channels = 1 if image.mode == "L" else 3
        if random.random() < self.p:
            if task_name in ["jigsaw"]:
                return {
                    "image": F.to_grayscale(
                        image, num_output_channels=num_output_channels
                    ),
                    "label": label,
                }
            else:  # noqa: RET505
                raise ValueError(f"task name {task_name} not available!")
        else:
            return sample


class ColorJitter(T.ColorJitter):
    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
    ) -> None:
        self.brightness = self._check_input(brightness, "brightness")  # type: ignore
        self.contrast = self._check_input(contrast, "contrast")  # type: ignore
        self.saturation = self._check_input(saturation, "saturation")  # type: ignore
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )  # type: ignore

    def __call__(self, sample: dict, task_name: str) -> dict:
        # t = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        image, label = sample["image"], sample["label"]
        if task_name in ["autoencoder"]:
            return {
                "image": self.forward(image),  # type: ignore
                "label": self.forward(label),
            }  # type: ignore
        elif task_name == "segmentsemantic" or task_name in [  # noqa: RET505
            "class_object",
            "class_scene",
            "room_layout",
            "normal",
            "jigsaw",
        ]:
            return {"image": self.forward(image), "label": label}  # type: ignore
        else:
            raise ValueError(f"task name {task_name} not available!")


########################
# Classification Tasks #
########################


def np_softmax(logits: np.ndarray) -> np.ndarray:
    maxs = np.amax(logits, axis=-1)
    softmax = np.exp(logits - np.expand_dims(maxs, axis=-1))
    sums = np.sum(softmax, axis=-1)
    softmax = softmax / np.expand_dims(sums, -1)
    return softmax


def load_class_object_logits(
    label_path: str,
    selected: bool = False,
    normalize: bool = True,
    final5k: bool = False,
) -> np.ndarray:
    """Class 1000 ImageNet Ground Truth
    :param label_path: path to a specific label file
    :param selected: original taskonomy data: 1000 cls -> 100 cls
    :param final5k: transnas-bench data: 100 cls -> 75 cls
    :param normalize: normalize the remaining 75 cls label (sum to be 1.0)
    :return: target: the target probability
        (returned as np.array with dim=(1000/100/75,)).
    """
    try:
        logits = np.load(label_path)
    except:
        print(f"corrupted: {label_path}!")
        raise
    lib_data_dir = os.path.abspath(os.path.dirname(__file__))
    if selected:  # original taskonomy data: 1000 cls -> 100 cls
        selection_file = (
            os.path.join(lib_data_dir, "class_object_final5k.npy")
            if final5k
            else os.path.join(lib_data_dir, "class_object_selected.npy")
        )
        selection = np.load(selection_file)
        logits = logits[selection.astype(bool)]
        if normalize:
            logits = logits / logits.sum()
    target = np.asarray(logits)
    return target


def load_class_object_label(
    label_path: str, selected: bool = False, final5k: bool = False
) -> np.ndarray:
    """Class 1000 ImageNet Ground Truth
    :param label_path: path to a specific label file
    :param selected: original taskonomy data: 1000 cls -> 100 cls
    :param final5k: transnas-bench data: 100 cls -> 75 cls
    :return: target: the target label (returned as np.array with dim=(1,)).
    """
    logits = load_class_object_logits(
        label_path, selected=selected, normalize=False, final5k=final5k
    )
    target = np.asarray(logits.argmax())
    return target


def load_class_scene_logits(
    label_path: str,
    selected: bool = False,
    normalize: bool = True,
    final5k: bool = False,
) -> np.ndarray:
    """Class Scene Ground Truth
    :param label_path: path to a specific label file
    :param selected: original taskonomy data: 365 cls -> 63 cls
    :param final5k: transnas-bench data: 63 cls -> 47 cls
    :param normalize: normalize the remaining 47 cls label (sum to be 1.0)
    :return: target: the target probability
        (returned as np.array with dim=(365/63/47,)).
    """
    try:
        logits = np.load(label_path)
    except:  # noqa: E722
        raise FileNotFoundError(f"corrupted: {label_path}!")  # noqa: B904
    lib_data_dir = os.path.abspath(os.path.dirname(__file__))
    if selected:
        selection_file = (
            os.path.join(lib_data_dir, "class_scene_final5k.npy")
            if final5k
            else os.path.join(lib_data_dir, "class_scene_selected.npy")
        )
        selection = np.load(selection_file)
        logits = logits[selection.astype(bool)]
        if normalize:
            logits = logits / logits.sum()
    target = np.asarray(logits)
    return target


def load_class_scene_label(
    label_path: str, selected: bool = False, final5k: bool = False
) -> np.ndarray:
    """Class Scene Ground Truth
    :param label_path: path to a specific label file
    :param selected: original taskonomy data: 365 cls -> 63 cls
    :param final5k: transnas-bench data: 63 cls -> 47 cls
    :return: target: the target label (returned as np.array with dim=(1,)).
    """
    logits = load_class_scene_logits(
        label_path, selected=selected, normalize=False, final5k=final5k
    )
    target = np.asarray(logits.argmax())
    return target


def get_synset(task: str, selected: bool = True, final5k: bool = False) -> list[str]:
    """Get class names for classification tasks
    :param task: task in ['class_object', 'class_scene']
    :param selected: original taskonomy data with reduced classes
    :param final5k: transnas-bench data with reduced classes
    :return: synset: the synset names (returned as list of str).
    """
    if task == "class_scene":
        selection_file = (
            "class_scene_final5k.npy" if final5k else "class_scene_selected.npy"
        )
        selection = np.load((Path(__file__).parent / selection_file).resolve())
        with open((Path(__file__).parent / "class_scene_names.json").resolve()) as fp:
            synset_scene = [x[3:] for x in json.load(fp)]
            if selected:
                synset_scene = [x for x, y in zip(synset_scene, selection) if y == 1.0]
        synset = synset_scene
    elif task == "class_object":
        synset_object = [" ".join(i.split(" ")[1:]) for i in raw_synset]
        selection_file = (
            "class_object_final5k.npy" if final5k else "class_object_selected.npy"
        )
        selection = np.load((Path(__file__).parent / selection_file).resolve())
        if selected:
            synset_object = [x for x, y in zip(synset_object, selection) if y == 1.0]
        synset = synset_object
    else:
        raise ValueError(
            f"{task} not in ['class_object', 'class_scene'], cannot get_synset"
        )
    return synset


#####################################
# Room Layout (code from taskonomy) #
#####################################


########################
# Autoencoder & Normal #
########################


################
# Segmentation #
################


##########
# jigsaw #
##########


def get_all_templates(dataset_dir: str, filenames_path: str) -> list[str]:
    """Get all templates.
    :param dataset_dir: the dir containing the taskonomy dataset
    :param filenames_path: /path/to/json_file for train/val/test_filenames
        (specifies which buildings to include)
    :return: a list of absolute paths of all templates
        e.g. "{building}/{domain}/point_0_view_0_domain_{domain}".
    """
    building_lists = read_json(filenames_path)["filename_list"]
    all_template_paths: list = []
    for building in building_lists:
        all_template_paths += read_json(os.path.join(dataset_dir, building))
    for i, path in enumerate(all_template_paths):
        f_split = path.split(".")
        if f_split[-1] in ["npy", "png"]:
            all_template_paths[i] = ".".join(f_split[:-1])
    return all_template_paths
