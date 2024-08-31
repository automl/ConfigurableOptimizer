from __future__ import annotations

import torch
from torch import nn
from typing_extensions import TypeAlias

LRSchedulerType: TypeAlias = torch.optim.lr_scheduler.LRScheduler
OptimizerType: TypeAlias = torch.optim.Optimizer


def configure_optimizer(
    optimizer_old: OptimizerType, optimizer_new: OptimizerType
) -> OptimizerType:
    # Get the state dictionaries of the old and new optimizers
    old_state_dict = optimizer_old.state_dict()
    # new_state_dict = optimizer_new.state_dict()

    # Iterate over the parameters in the new optimizer
    for i, p in enumerate(optimizer_new.param_groups[0]["params"]):
        # If the parameter does not have a 'raw_id', copy the state directly
        if not hasattr(p, "raw_id") and not hasattr(p, "optimizer_id"):
            optimizer_new.state[p] = optimizer_old.state[p]
            p.optimizer_id = i
            continue

        # Find the old state using the 'optimizer_id' attribute
        optimizer_id = p.optimizer_id
        if optimizer_id in old_state_dict["state"]:
            if optimizer_id != i:
                state_old = optimizer_new.state_dict()["state"].get(optimizer_id, {})
            else:
                state_old = old_state_dict["state"].get(p.optimizer_id, {})
            state_new = optimizer_new.state[p]

            state_new["momentum_buffer"] = state_old["momentum_buffer"]
            # Copy momentum buffer if it exists
            if "momentum_buffer" in state_old:
                if hasattr(p, "t") and p.t == "bn":
                    get_momentum_buffer_bn(p, state_new)

                elif hasattr(p, "t") and p.t == "conv":
                    get_momentum_buffer_conv(p, state_new)

            if optimizer_id != i:
                p.ref_id = p.optimizer_id
                p.optimizer_id = i

    return optimizer_new


def get_momentum_buffer_bn(p: nn.Parameter, state_new: dict) -> torch.Tensor:
    for index in p.out_index:
        state_new["momentum_buffer"] = torch.cat(
            [
                state_new["momentum_buffer"],
                state_new["momentum_buffer"][index].clone(),
            ],
            dim=0,
        )
    del p.t, p.raw_id, p.out_index
    return state_new["momentum_buffer"]


def get_momentum_buffer_conv(p: nn.Parameter, state_new: dict) -> torch.Tensor:
    if hasattr(p, "in_index"):
        for index in p.in_index:
            state_new["momentum_buffer"] = torch.cat(
                [
                    state_new["momentum_buffer"],
                    state_new["momentum_buffer"][:, index, :, :].clone(),
                ],
                dim=1,
            )
        del p.in_index
    if hasattr(p, "out_index"):
        for index in p.out_index:
            state_new["momentum_buffer"] = torch.cat(
                [
                    state_new["momentum_buffer"],
                    state_new["momentum_buffer"][index, :, :, :].clone(),
                ],
                dim=0,
            )
        del p.out_index
    del p.t, p.raw_id
    return state_new["momentum_buffer"]


def configure_scheduler(
    scheduler_old: LRSchedulerType,
    scheduler_new: LRSchedulerType,
) -> LRSchedulerType:
    """Configures a new LR-scheduler using state information from an old scheduler.
    Code originates from github.com/xiangning-chen/DrNAS/blob/master/net2wider.py.
    """
    scheduler_new.load_state_dict(scheduler_old.state_dict())
    return scheduler_new
