from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F  # noqa: N812

from confopt.searchspace.common.mixop import OperationChoices
from confopt.searchspace.darts.core.genotypes import DARTSGenotype
from confopt.searchspace.darts.core.operations import FactorizedReduce, ReLUConvBN

from .operations import OPS, drop_path

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MixedOp(nn.Module):
    def __init__(self, C: int, stride: int, primitives: list[str]) -> None:
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(
        self,
        primitives: dict[str, list[list[str]]],
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
        self.primitives = primitives[
            "primitives_reduct" if reduction else "primitives_normal"
        ]

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                ops = MixedOp(C, stride, self.primitives[edge_index])._ops
                op = OperationChoices(ops, is_reduction_cell=reduction)
                self._ops.append(op)
                edge_index += 1

    def forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights: list[torch.Tensor],
        beta_weights: torch.Tensor | None = None,
        drop_prob: float = 0.2,
    ) -> torch.Tensor:
        if beta_weights is not None:
            return self.edge_normalization_forward(s0, s1, weights, beta_weights)

        return self._forward(s0, s1, weights, drop_prob)

    def _forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights: list[torch.Tensor],
        drop_prob: float = 0.0,
    ) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _i in range(self._steps):
            if drop_prob > 0.0 and self.training:
                s = sum(
                    drop_path(self._ops[offset + j](h, weights[offset + j]), drop_prob)
                    for j, h in enumerate(states)
                )
            else:
                s = sum(
                    self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states)
                )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)

    def edge_normalization_forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights: list[torch.Tensor],
        beta_weights: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Edge normalization is not implemented yet for \
                Robust-DARTS Search Space"
        )


class Network(nn.Module):
    def __init__(
        self,
        primitives: dict[str, list],
        C: int = 16,
        num_classes: int = 10,
        layers: int = 8,
        criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss,
        steps: int = 4,
        multiplier: int = 4,
        stem_multiplier: int = 3,
        drop_path_prob: float = 0.0,
        edge_normalization: bool = False,
    ):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self.drop_path_prob = drop_path_prob
        self.edge_normalization = edge_normalization
        self.primitives = primitives

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
                primitives,
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
            primitives=self.primitives,
            C=self._C,
            num_classes=self._num_classes,
            layers=self._layers,
            criterion=self._criterion,
            steps=self._steps,
            multiplier=self.stem_multiplier,
            stem_multiplier=self._stem_multiplier,
            drop_path_prob=self.drop_path_prob,
            edge_normalization=self.edge_normalization,
        ).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def sample(self, alphas: torch.Tensor) -> torch.Tensor:
        # Replace this function on the fly to change the sampling method
        return F.softmax(alphas, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.edge_normalization:
            return self.edge_normalized_forward(x)

        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)

        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.sample(self.alphas_reduce)
            else:
                weights = self.sample(self.alphas_normal)

            s0, s1 = s1, cell(s0, s1, weights, drop_prob=self.drop_path_prob)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return out.view(out.size(0), -1), logits

    def edge_normalized_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Edge normalization is not yet implemented for \
                Robust-DARTS Search Space"
        )

    def _loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        return self._criterion(logits, target)

    def _initialize_parameters(self) -> None:
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.primitives["primitives_normal"][0])

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

    def arch_parameters(self) -> list[Variable]:
        return self._arch_parameters

    def genotype(self) -> DARTSGenotype:
        def _parse(weights: torch.Tensor, normal: bool = True) -> list[tuple[str, int]]:
            primitives = self.primitives[
                "primitives_normal" if normal else "primitives_reduct"
            ]

            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()

                try:
                    edges = sorted(
                        range(i + 2),
                        key=lambda x: -max(
                            W[x][k]
                            for k in range(len(W[x]))
                            if k != primitives[x].index("none")
                        ),
                    )[:2]
                except (
                    ValueError
                ):  # This error happens when the 'none' op is not present in the ops
                    edges = sorted(
                        range(i + 2),
                        key=lambda x: -max(W[x][k] for k in range(len(W[x]))),
                    )[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if ("none" in primitives[j]) and (
                            k != primitives[j].index("none")
                        ):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                        elif k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((primitives[start + j][k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True
        )
        gene_reduce = _parse(
            F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False
        )

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = DARTSGenotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype

    def beta_parameters(self) -> list[torch.nn.Parameter]:
        """Get a list containing the beta parameters of partial connection used for
        edge normalization.

        Returns:
            list[torch.Tensor]: A list containing the beta parameters for the model.
        """
        return self._betas
