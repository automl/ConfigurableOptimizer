from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.distributions import Dirichlet, kl_divergence
import torch.nn.functional as F  # noqa: N812

from confopt.oneshot.base import OneShotComponent
from confopt.searchspace.common import (
    DrNASRegTermSupport,
    FairDARTSRegTermSupport,
    FLOPSRegTermSupport,
    SearchSpace,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Regularizer(OneShotComponent):
    def __init__(
        self,
        reg_terms: list[RegularizationTerm],
        reg_weights: list[float],
        loss_weight: float,
    ) -> None:
        super().__init__()

        assert len(reg_weights) == len(
            reg_terms
        ), "Length of reg_weights must match reg_terms"

        self.reg_terms = reg_terms
        self.reg_weights = reg_weights
        self.loss_weight = loss_weight

    def add_reg_terms(self, model: SearchSpace, loss: torch.Tensor) -> torch.Tensor:
        reg_terms = 0.0
        for term, weight in zip(self.reg_terms, self.reg_weights):
            reg_terms += weight * term.loss(model)

        return loss * self.loss_weight + reg_terms


class RegularizationTerm(ABC):
    @abstractmethod
    def loss(self, model: SearchSpace) -> torch.Tensor:
        ...


class DrNASRegularizationTerm(RegularizationTerm):
    def __init__(self, reg_scale: float, reg_type: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reg_scale = reg_scale
        self.reg_type = reg_type

    def loss(self, model: SearchSpace) -> torch.Tensor:
        assert isinstance(
            model, DrNASRegTermSupport
        ), "Model must be of type DrNASRegTermSupport"

        anchors = model.get_drnas_anchors()
        arch_parameters = model.arch_parameters

        assert len(arch_parameters) == len(anchors), (
            "There should be same number of anchors"
            + " as the number of arch parameters in the model"
        )

        if self.reg_type == "kl":
            loss = self.loss_kl(anchors, arch_parameters)
        elif self.reg_type == "l2":
            loss = self.loss_l2(arch_parameters)
        else:
            raise ValueError(f"Unknown regularization type: {self.reg_type}")
        return loss

    def loss_l2(self, arch_parameters: list[nn.Parameter]) -> torch.Tensor:
        l2_reg_terms = []
        for alphas in arch_parameters:
            l2_reg_terms.append(alphas.norm())

        return self.reg_scale * sum(l2_reg_terms)

    def loss_kl(
        self, anchors: torch.Tensor, arch_parameters: list[nn.Parameter]
    ) -> torch.Tensor:
        kl_reg_terms = []
        for anchor, alphas in zip(anchors, arch_parameters):
            cons = F.elu(alphas) + 1
            q = Dirichlet(cons)
            p = anchor
            kl_reg_terms.append(torch.sum(kl_divergence(q, p)))

        return self.reg_scale * sum(kl_reg_terms)


class FLOPSRegularizationTerm(RegularizationTerm):
    def loss(self, model: SearchSpace) -> torch.Tensor:
        assert isinstance(model, FLOPSRegTermSupport)

        return model.get_weighted_flops()


class FairDARTSRegularizationTerm(RegularizationTerm):
    """Computes the regularization term for FairDARTS. Taken from
    https://github.com/xiaomi-automl/FairDARTS/blob/master/fairdarts/separate_loss.py.
    """

    def loss(self, model: SearchSpace) -> torch.Tensor:
        assert isinstance(
            model, FairDARTSRegTermSupport
        ), "Model must be of type FairDARTSRegTermSupport"

        arch_parameters = model.get_fair_darts_arch_parameters()

        if arch_parameters is None:
            return torch.tensor(0.0, requires_grad=False).to(DEVICE)

        arch_parameters = torch.cat(arch_parameters, dim=0)
        loss = -F.l1_loss(
            arch_parameters, torch.tensor(0.5, requires_grad=False).to(DEVICE)
        )

        return loss
