from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch import nn

from .checkpoints import (
    copy_checkpoint,
    save_checkpoint,
)
from .distributed import (
    cleanup,
    get_device,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
)
from .logger import Logger, prepare_logger
from .normalize_params import normalize_params
from .time import get_runtime, get_time_as_string


class ExperimentCheckpointLoader:
    @classmethod
    def load_checkpoint(
        cls,
        logger: Logger,
        src: Literal["last", "best", "epoch"],
        epoch: int | None = None,
    ) -> nn.Module:
        assert src in [
            "last",
            "best",
            "epoch",
        ], "src must be 'last', 'best', or 'epoch'"

        if src == "best":
            path = logger.path("best_model")
        elif src == "last":
            path = logger.path("last_checkpoint")
        elif src == "epoch":
            assert epoch is not None, "epoch argument must be given when src is 'epoch'"
            path = logger.path("checkpoints")
            path = "{}/{}_{:07d}.pth".format(path, "model", epoch)

        logger.log(f"Loading checkpoint {path}")
        return torch.load(path, map_location=torch.device("cpu"))


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


def freeze(m: torch.nn.Module) -> None:
    for param in m.parameters():
        param.requires_grad_(False)


def preserve_gradients_in_module(
    m: torch.nn.Module,
    ignored_modules: tuple[torch.nn.Module],
    oles_ops: list[torch.nn.Module],
) -> None:
    if isinstance(m, ignored_modules):
        return

    if not isinstance(m, tuple(oles_ops)):
        return

    if not hasattr(m, "pre_grads"):
        m.pre_grads = []

    for param in m.parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.detach().cpu()
            m.pre_grads.append(g)


def clear_grad_cosine(m: torch.nn.Module) -> None:
    if not hasattr(m, "avg"):
        return
    m.pre_grads.clear()
    m.avg = 0
    m.count = 0


def calc_layer_alignment_score(layer_gradients: list[torch.Tensor]) -> float:
    scale = len(layer_gradients) * (len(layer_gradients) - 1) / 2
    score = 0
    for i in range(len(layer_gradients)):
        for j in range(i + 1, len(layer_gradients)):
            g, g_ = layer_gradients[i], layer_gradients[j]
            numerator = torch.dot(g, g_)
            denominator = g.norm(p=2.0) * g_.norm(p=2.0)
            score += numerator / denominator
    assert score.shape == torch.Size([])  # type: ignore
    return score.item() / scale  # type: ignore


def reset_gm_score_attributes(module: torch.nn.Module) -> None:
    if hasattr(module, "count"):
        module.count = 0
    if hasattr(module, "avg"):
        module.avg = 0
    if hasattr(module, "pre_grads"):
        module.pre_grads.clear()
    if hasattr(module, "running_sim"):
        module.running_sim.reset()


def set_ops_to_prune(model: torch.nn.Module, mask: torch.Tensor) -> None:
    from confopt.searchspace.common.mixop import OperationBlock, OperationChoices

    assert isinstance(model, (OperationBlock, OperationChoices))
    assert len(mask) == len(model.ops)
    for op, mask_val in zip(model.ops, mask):
        if not torch.is_nonzero(mask_val):
            freeze(op)
            op.is_pruned = True


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


__all__ = [
    "calc_accuracy",
    "save_checkpoint",
    "load_checkpoint",
    "copy_checkpoint",
    "init_distributed",
    "cleanup",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "get_machine_info",
    "get_time_as_string",
    "get_runtime",
    "prepare_logger",
    "Logger",
    "BaseProfile",
    "get_device",
    "normalize_params",
    "calc_layer_alignment_score",
    "reset_gm_score_attributes",
    "set_ops_to_prune",
]
