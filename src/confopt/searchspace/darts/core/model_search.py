from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from confopt.searchspace.common.mixop import OperationChoices

from .genotypes import PRIMITIVES, Genotype
from .operations import OPS, FactorizedReduce, ReLUConvBN

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MixedOp(nn.Module):
    def __init__(self, C: int, stride: int):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            # if "pool" in primitive:
            #     op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        return sum(w * op(x) for w, op in zip(weights, self._ops))  # type: ignore


class Cell(nn.Module):
    def __init__(
        self,
        steps: int,
        multiplier: int,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
    ):
        super().__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(
                C_prev_prev, C, 1, 1, 0, affine=False
            )  # type: ignore
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                ops = MixedOp(C, stride)._ops
                op = OperationChoices(ops, is_reduction_cell=reduction)
                self._ops.append(op)

    def forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights: list[torch.Tensor],
        beta_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _i in range(self._steps):
            s = sum(
                (beta_weights[offset + j] if beta_weights is not None else 1)
                * self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class Network(nn.Module):
    def __init__(
        self,
        C: int = 16,
        num_classes: int = 10,
        layers: int = 8,
        criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss,
        steps: int = 4,
        multiplier: int = 4,
        stem_multiplier: int = 3,
        edge_normalization: bool = False,
    ) -> None:
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.edge_normalization = edge_normalization
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_parameters()

    def new(self) -> Network:
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion
        ).to(DEVICE)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        for x, y in zip(model_new.beta_parameters(), self.beta_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        for _i, cell in enumerate(self.cells):
            if self.edge_normalization:
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                    n = 3
                    start = 2
                    weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
                    for _i in range(self._steps - 1):
                        end = start + n
                        tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                        start = end
                        n += 1
                        weights2 = torch.cat([weights2, tw2], dim=0)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                    n = 3
                    start = 2
                    weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
                    for _i in range(self._steps - 1):
                        end = start + n
                        tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                        start = end
                        n += 1
                        weights2 = torch.cat([weights2, tw2], dim=0)
                s0, s1 = s1, cell(s0, s1, weights, weights2)
            else:
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return torch.squeeze(out, dim=(-1, -2)), logits  # type: ignore

    def _loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        return self._criterion(logits, target)  # type: ignore

    def _initialize_parameters(self) -> None:
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops).to(DEVICE))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops).to(DEVICE))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

        self.betas_normal = nn.Parameter(1e-3 * torch.randn(k).to(DEVICE))
        self.betas_reduce = nn.Parameter(1e-3 * torch.randn(k).to(DEVICE))
        self._betas = [
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self) -> list[torch.nn.Parameter]:
        return self._arch_parameters  # type: ignore

    def beta_parameters(self) -> list[torch.nn.Parameter]:
        return self._betas

    def genotype(self) -> Genotype:
        def _parse(weights: list[torch.Tensor]) -> list[tuple[str, int]]:
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k]
                        for k in range(len(W[x]))  # type: ignore
                        if k != PRIMITIVES.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index("none") and (
                            k_best is None or W[j][k] > W[j][k_best]
                        ):
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))  # type: ignore
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype
