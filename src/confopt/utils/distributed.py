from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    print("Initializing distributed training.")
    if "SLURM_PROCID" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        rank = int(os.environ["SLURM_PROCID"])
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    else:
        print("Not using distributed training.")
        local_rank, rank, world_size = 0, 0, 1

    return local_rank, rank, world_size


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_local_rank() -> int:
    if dist.is_initialized():
        return int(os.environ["SLURM_LOCALID"])
    return 0


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", get_local_rank())
    return torch.device("cpu")


def print_on_master_only(is_master: bool) -> None:
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def master_print(*args, **kwargs) -> None:  # type:ignore #noqa: A001
        if is_master or args[0].startswith("ALL"):
            builtin_print(*args, **kwargs)

    __builtin__.print = master_print
