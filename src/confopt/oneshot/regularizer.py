from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.distributions import Dirichlet, kl_divergence
import torch.nn.functional as F  # noqa: N812

from confopt.oneshot.base_component import OneShotComponent
from confopt.searchspace.common import (
    DrNASRegTermSupport,
    FairDARTSRegTermSupport,
    FLOPSRegTermSupport,
    SearchSpace,
)


class Regularizer(OneShotComponent):
    def __init__(
        self,
        reg_terms: list[RegularizationTerm],
        reg_weights: list[float],
        loss_weight: float,
    ) -> None:
        super().__init__()

        assert (
            sum([loss_weight, *reg_weights]) == 1.0
        ), f"Sum of loss_weight ({loss_weight}) and reg_weights \
        ({reg_weights}) must be 1.0"
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
    def __init__(self, reg_scale: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reg_scale = reg_scale

    def loss(self, model: SearchSpace) -> torch.Tensor:
        assert isinstance(
            model, DrNASRegTermSupport
        ), "Model must be of type DrNASRegTermSupport"

        anchors = model.get_drnas_anchors()
        arch_parameters = model.arch_parameters

        if anchors[1] is None:
            anchors = [anchors[0]]
            assert (
                len(arch_parameters) == 1
            ), "Only one set of arch_parameters is expected"

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
            return torch.tensor(0.0, requires_grad=False).cuda()

        arch_parameters = torch.cat(arch_parameters, dim=0)
        loss = -F.l1_loss(
            arch_parameters, torch.tensor(0.5, requires_grad=False).cuda()
        )

        return loss
