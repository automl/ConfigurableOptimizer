from __future__ import annotations

from typing import Iterable

import torch

from .checkpoints import (
    copy_checkpoint,
    save_checkpoint,
)
from .logger import Logger, prepare_logger
from .normalize_params import normalize_params
from .time import get_time_as_string


class AverageMeter:
    """Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py.
    """

    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Iterable = (1,)
) -> list[float]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob

        # mask = torch.nn.Parameter(
        #     torch.cuda.FloatTensor(x.size(0), 1, 1, 1, dtype=torch.float32).bernoulli
        # _(keep_prob
        #     )
        # ).to(device=x.device)
        mask = torch.nn.Parameter(
            torch.empty(x.size(0), 1, 1, 1, dtype=torch.float32).bernoulli_(keep_prob)
        ).to(device=x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def get_num_classes(dataset: str) -> int:
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "imgnet16_120" or dataset == "imgnet16":
        num_classes = 120
    else:
        raise ValueError("dataset is not defined.")
    return num_classes


__all__ = [
    "calc_accuracy",
    "save_checkpoint",
    "load_checkpoint",
    "copy_checkpoint",
    "get_machine_info",
    "get_time_as_string",
    "prepare_logger",
    "Logger",
    "BaseProfile",
    "get_device",
    "normalize_params",
]
