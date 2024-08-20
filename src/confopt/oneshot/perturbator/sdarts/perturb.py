from __future__ import annotations

from typing import Any, Callable, Literal

import torch
from torch.optim.optimizer import Optimizer, required

from confopt.oneshot.perturbator import BasePerturbator
from confopt.searchspace.common import SearchSpace


class SDARTSPerturbator(BasePerturbator):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        epsilon: float = 0.03,
        search_space: SearchSpace | None = None,
        data: tuple[torch.Tensor, torch.Tensor] | None = None,
        loss_criterion: torch.nn.modules.loss._Loss | None = None,
        attack_type: Literal["random", "adversarial"] = "random",
        steps: int = 7,
        random_start: bool = True,
        sample_frequency: Literal["epoch", "step"] = "step",
    ) -> None:
        super().__init__(
            arch_parameters=arch_parameters, sample_frequency=sample_frequency
        )
        self.epsilon = epsilon

        assert attack_type in [
            "random",
            "adversarial",
        ], "attack_type must be either 'random' or 'adversarial'"
        self.attack_type = attack_type

        # Initialize variables for adversarial attack
        if self.attack_type == "adversarial":
            assert search_space is not None, "search_space should not be None"

            assert data is not None, "data should not be None"

        if search_space is not None:
            self.model = search_space
        if data is not None:
            self.X, self.target = data
        self.steps = steps
        self.random_start = random_start

        if loss_criterion is None:
            self.loss_criterion = torch.nn.CrossEntropyLoss()
        else:
            self.loss_criterion = loss_criterion

    def perturb_alphas(self, arch_parameters: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.attack_type == "random":
            perturbed_alphas = []
            for alpha in arch_parameters:
                perturbed_alphas.append(
                    alpha.clone().data.add_(
                        torch.zeros_like(alpha).uniform_(-self.epsilon, self.epsilon)
                    )
                )

            self.clip(perturbed_alphas)
            return perturbed_alphas

        return self.perturb_linf_pgd_alpha(
            self.model,
            self.loss_criterion,
            self.X,
            self.target,
            arch_parameters,
            self.epsilon,
            self.steps,
            self.random_start,
        )

    def perturb_random(self, alpha: torch.Tensor, epsilon: float) -> torch.Tensor:
        alpha.data.add_(torch.zeros_like(alpha).uniform_(-epsilon, epsilon))
        return alpha

    def perturb_linf_pgd_alpha(
        self,
        model: torch.nn.Module,
        loss_criterion: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        alphas: list[torch.Tensor],
        epsilon: float,
        steps: int = 7,
        random_start: bool = True,
    ) -> list[torch.Tensor]:
        saved_params = [p.clone() for p in alphas]
        optimizer = LinfSGD(alphas, lr=2 * epsilon / steps)
        with torch.no_grad():
            _, logits = model(X)
            loss_before = loss_criterion(logits, y)

        if random_start:
            for p in alphas:
                p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
            self.clip(model.arch_parameters)

        for _ in range(steps):
            optimizer.zero_grad()
            model.zero_grad()
            __, logits = model(X)
            loss = loss_criterion(logits, y)
            loss.backward()
            optimizer.step()
            diff = [
                (alphas[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))
            ]
            for i, p in enumerate(alphas):
                p.data.copy_(diff[i] + saved_params[i])
            self.clip(model.arch_parameters)

        optimizer.zero_grad()
        model.zero_grad()
        with torch.no_grad():
            _, logits = model(X)
            loss_after = loss_criterion(logits, y)
        if loss_before > loss_after:
            for i, p in enumerate(alphas):
                p.data.copy_(saved_params[i])

        return alphas

    def clip(self, alphas: list[torch.Tensor]) -> None:
        for p in alphas:
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())


class LinfSGD(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Any = required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure: Callable | None = None) -> torch.Tensor:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = torch.sign(p.grad.data)
                # d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = d_p.add(momentum, buf) if nesterov else buf

                p.data.add_(d_p, alpha=-group["lr"])

        return loss
