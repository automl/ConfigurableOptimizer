from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.mixop import AuxiliarySkipConnection
from confopt.searchspace.darts.core.operations import (
    OPS,
    FactorizedReduce,
    Identity,
    ReLUConvBN,
)
from confopt.utils import drop_path

from .genotypes import Genotype

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class AuxilaryNetworkSkipConnection(nn.Module):
    def __init__(self, operation: nn.Module, stride: int, C: int | None = None) -> None:
        super().__init__()
        self.operation = operation
        self.aux_skip = AuxiliarySkipConnection(
            stride=stride, C_in=C, C_out=C, affine=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aux_skip(x) + self.operation(x)


class Cell(nn.Module):
    def __init__(
        self,
        genotype: Genotype,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
        use_auxiliary_skip_connection: bool = False,
    ) -> None:
        super().__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        self.use_auxiliary_skip_connection = use_auxiliary_skip_connection
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(
        self, C: int, op_names: list, indices: list, concat: tuple, reduction: bool
    ) -> None:
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            if self.use_auxiliary_skip_connection:
                op = AuxilaryNetworkSkipConnection(op, stride, C)

            self._ops += [op]
        self._indices = indices

    def forward(
        self, s0: torch.Tensor, s1: torch.Tensor, drop_prob: float
    ) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C: int, num_classes: int) -> None:
        """Assuming input size 8x8."""
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),  # noqa: PD002
            nn.AvgPool2d(
                5, stride=3, padding=0, count_include_pad=False
            ),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # noqa: PD002
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),  # noqa: PD002
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C: int, num_classes: int) -> None:
        """Assuming input size 14x14."""
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),  # noqa: PD002
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # noqa: PD002
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE:This batchnorm was omitted in my earlier implementation due to a typo
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),  # noqa: PD002
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):
    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        auxiliary: bool,
        genotype: Genotype,
        drop_path_prob: float = 0.0,
        use_auxiliary_skip_connection: bool = False,
        # TODO: Verify that 0. is the correct default value for drop_path_prob
    ) -> None:
        super().__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob

        stem_multiplier = 3
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
                genotype,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                use_auxiliary_skip_connection,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3 and self._auxiliary and self.training:
                logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits_aux, logits


class NetworkImageNet(nn.Module):
    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        auxiliary: bool,
        genotype: Genotype,
        drop_path_prob: float = 0.0,
        # TODO: Verify that 0. is the correct default value for drop_path_prob
    ) -> None:
        super().__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),  # noqa: PD002
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            # TODO: code has been taken from the original repo
            # ruff's error on inplace=True should be fixed
            nn.ReLU(inplace=True),  # noqa: PD002
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        logits_aux = None
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3 and self._auxiliary and self.training:
                logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits_aux, logits
