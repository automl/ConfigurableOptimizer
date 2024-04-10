from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from confopt.searchspace.common.mixop import OperationBlock, OperationChoices
from confopt.utils import drop_path
from confopt.utils.normalize_params import normalize_params

from .genotypes import BABY_PRIMITIVES, PRIMITIVES, DARTSGenotype
from .model import NetworkCIFAR, NetworkImageNet
from .operations import OPS, FactorizedReduce, Identity, ReLUConvBN

NUM_CIFAR_CLASSES = 10
NUM_CIFAR100_CLASSES = 100
NUM_IMAGENET_CLASSES = 120
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MixedOp(nn.Module):
    def __init__(self, C: int, stride: int, primitives: list[str] = PRIMITIVES):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C, stride, False)
            # TODO: is it okay to remove this?
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
        primitives: list[str] = PRIMITIVES,
    ):
        """Neural Cell for DARTS.

        Represents a neural cell used in DARTS.

        Args:
            steps (int): Number of steps in the cell.
            multiplier (int): Multiplier for channels in the cell.
            C_prev_prev (int): Number of channels in the previous-previous cell.
            C_prev (int): Number of channels in the previous cell.
            C (int): Number of channels in the current cell.
            reduction (bool): Whether the cell is a reduction cell.
            reduction_prev (bool): Whether the previous cell is a reduction cell.
            primitives (list): The list of primitives to use for generating cell.

        Attributes:
            preprocess0(nn.Module): Preprocess for input from previous-previous cell.
            preprocess1(nn.Module): Preprocess for input from previous cell.
            _ops(nn.ModuleList): List of operations in the cell.
            reduction(bool): Whether the cell is a reduction cell (True) or
                             a normal cell (False).
        """
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
                ops = MixedOp(C, stride, primitives)._ops
                op = OperationChoices(ops, is_reduction_cell=reduction)
                self._ops.append(op)

    def forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights: list[torch.Tensor] | None = None,
        beta_weights: torch.Tensor | None = None,
        drop_prob: float | None = 0.2,
    ) -> torch.Tensor:
        """Forward pass of the cell.

        Args:
            s0 (torch.Tensor): First input tensor to the model.
            s1 (torch.Tensor): Second input tensor to the model.
            weights (list[torch.Tensor]): Alpha weights to the edges.
            beta_weights (torch.Tensor): Beta weights for the edge.
            drop_prob: (float|None): the droping probability of a path (for discrete).

        Returns:
            torch.Tensor: state ouptut from the cell
        """
        if weights is None:
            return self.discrete_model_forward(s0, s1, drop_prob)
        if beta_weights is not None:
            return self.edge_normalization_forward(s0, s1, weights, beta_weights)

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)

    def discrete_model_forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        drop_prob: float | None = None,
    ) -> torch.Tensor:
        if self._indices is None or self._concat is None:
            raise ValueError(
                "could not do forward pass for discrete darts search space"
            )

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
            if self.training and drop_prob is not None and drop_prob > 0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

    def edge_normalization_forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        weights: list[torch.Tensor],
        beta_weights: torch.Tensor,
    ) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _i in range(self._steps):
            s = sum(
                beta_weights[offset + j] * self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)

    def _discretize(self, genotype: DARTSGenotype) -> None:
        # TODO: create indices and concat
        if self.reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        C = self.preprocess1.op[1].out_channels
        self._compile(C, op_names, indices, concat, self.reduction)

        # for i, edge in enumerate(self._ops):
        #     max_idx = torch.argmax(weights[i], dim=-1)
        #     self._ops[i] = edge.ops[max_idx]  # type: ignore

    def _compile(
        self,
        C: int,
        op_names: list[str],
        indices: list[int],
        concat: range,
        reduction: bool,
    ) -> None:
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices


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
        discretized: bool = False,
        is_baby_darts: bool = False,
    ) -> None:
        """Implementation of DARTS search space's network model.

        Args:
            C (int): Number of channels. Defaults to 16.
            num_classes (int): Number of output classes. Defaults to 10.
            layers (int): Number of layers in the network. Defaults to 8.
            criterion (nn.modules.loss._Loss): Loss function. Defaults to nn.CrossEntropyLoss.
            steps (int): Number of steps in the search space cell. Defaults to 4.
            multiplier (int): Multiplier for channels in the cells. Defaults to 4.
            stem_multiplier (int): Stem multiplier for channels. Defaults to 3.
            edge_normalization (bool): Whether to use edge normalization. Defaults to False.
            discretized (bool): Whether supernet is discretized to only have one operation on
            each edge or not.
            is_baby_darts (bool): Controls which primitive list to use

        Attributes:
            stem (nn.Sequential): Stem network composed of Conv2d and BatchNorm2d layers.
            cells (nn.ModuleList): List of cells in the search space.
            global_pooling (nn.AdaptiveAvgPool2d): Global pooling layer.
            classifier (nn.Linear): Linear classifier layer.
            alphas_normal (nn.Parameter): Parameter for normal cells' alpha values.
            alphas_reduce (nn.Parameter): Parameter for reduction cells' alpha values.
            arch_parameters (list[nn.Parameter]): List of parameter for architecture alpha values.
            betas_normal (nn.Parameter): Parameter for normal cells' beta values.
            betas_reduce (nn.Parameter): Parameter for normal cells' beta values.
            beta_parameters (list[nn.Parameter]): List of parameter for architecture alpha values.
            discretized (bool): Whether the network is dicretized or not

        Note:
            This is a custom neural network model with various hyperparameters and
            architectural choices.
        """  # noqa: E501
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.edge_normalization = edge_normalization
        self.discretized = discretized
        self.mask: None | list[torch.Tensor] = None
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        self.is_baby_darts = is_baby_darts
        if is_baby_darts:
            self.primitives = BABY_PRIMITIVES
        else:
            self.primitives = PRIMITIVES

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
                self.primitives,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_parameters()

    def new(self) -> Network:
        """Get a new object with same arch and beta parameters.

        Return:
            Network: A torch module with same arch and beta parameters as this model.
        """
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion
        ).to(DEVICE)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        for x, y in zip(model_new.beta_parameters(), self.beta_parameters()):
            x.data.copy_(y.data)
        return model_new

    def sample(self, alphas: torch.Tensor) -> torch.Tensor:
        # Replace this function on the fly to change the sampling method
        return F.softmax(alphas, dim=-1)

    def forward(
        self, x: torch.Tensor, drop_prob: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network model.

        Args:
            x (torch.Tensor): Input x tensor to the model.
            drop_prob (float|None): the droping probability of a path.


        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - The output tensor after the forward pass.
            - The logits tensor produced by the model.
        """
        if self.discretized:
            return self.discrete_model_forward(x, drop_prob=drop_prob)
        if self.edge_normalization:
            return self.edge_normalization_forward(x)

        s0 = s1 = self.stem(x)
        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.sample(self.alphas_reduce)
                if self.mask is not None:
                    weights = normalize_params(weights, self.mask[1])
            else:
                weights = self.sample(self.alphas_normal)
                if self.mask is not None:
                    weights = normalize_params(weights, self.mask[0])
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return torch.squeeze(out, dim=(-1, -2)), logits

    def discrete_model_forward(
        self,
        inputs: torch.Tensor,
        drop_prob: float | None = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s0 = s1 = self.stem(inputs)
        for _i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, drop_prob=drop_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return torch.squeeze(out, dim=(-1, -2)), logits

    def edge_normalization_forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: normalization of alphas

        s0 = s1 = self.stem(inputs)
        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.sample(self.alphas_reduce)
                if self.mask is not None:
                    weights = normalize_params(weights, self.mask[1])
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
                weights = self.sample(self.alphas_normal)
                if self.mask is not None:
                    weights = normalize_params(weights, self.mask[0])
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

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return torch.squeeze(out, dim=(-1, -2)), logits

    def _loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss for the given input data and target.

        Args:
            x (torch.Tensor): Input data.
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Computed loss value.

        """
        logits = self(x)
        return self._criterion(logits, target)  # type: ignore

    def _initialize_parameters(self) -> None:
        """Initialize architectural and beta parameters for the cell.

        This function initializes the architectural and beta parameters required for
        the neural cell.
        """
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.primitives)

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
        """Get a list containing the architecture parameters or alphas.

        Returns:
            list[torch.Tensor]: A list containing the architecture parameters, such as
            alpha values.
        """
        return self._arch_parameters  # type: ignore

    def beta_parameters(self) -> list[torch.nn.Parameter]:
        """Get a list containing the beta parameters of partial connection used for
        edge normalization.

        Returns:
            list[torch.Tensor]: A list containing the beta parameters for the model.
        """
        return self._betas

    def genotype(self) -> DARTSGenotype:
        """Get the genotype of the model, representing the architecture.

        Returns:
            Structure: An object representing the genotype of the model, which describes
            the architectural choices in terms of operations and connections between
            nodes.
        """

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
                        if k != self.primitives.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != self.primitives.index("none") and (
                            k_best is None or W[j][k] > W[j][k_best]
                        ):
                            k_best = k
                    gene.append((self.primitives[k_best], j))  # type: ignore
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = DARTSGenotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype

    def _prune(self, op_sparsity: float, wider: int | None = None) -> None:
        """Discretize architecture parameters to enforce sparsity.

        Args:
            op_sparsity (float): The desired sparsity level, represented as a
            fraction of operations to keep.
            wider (int): If provided, this parameter determines how much wider the
            search space should be by multiplying the number of channels by this factor.

        Note:
            This method enforces sparsity in the architecture parameters by zeroing out
            a fraction of the smallest values, as specified by the `op_sparsity`
            parameter.
            It modifies the architecture parameters in-place to achieve the desired
            sparsity.
        """
        # TODO: should be removed from prune function to a seperate function
        # TODO: write wider function for ops
        self.mask = []
        for _name, module in self.named_modules():
            if isinstance(module, (OperationBlock, OperationChoices)):
                module.change_op_channel_size(wider)

        top_k = int(op_sparsity * len(self.primitives))
        for p in self._arch_parameters:
            sorted_arch_params, _ = torch.sort(p.data, dim=1, descending=True)
            thresholds = sorted_arch_params[:, :top_k]
            mask = p.data >= thresholds

            p.data *= mask.float()
            p.data[~mask].requires_grad = False
            if p.data[~mask].grad:
                p.data[~mask].grad.zero_()

            self.mask.append(mask)

    def _discretize(self) -> NetworkCIFAR | NetworkImageNet:
        genotype = self.genotype()
        if (
            self._num_classes == NUM_CIFAR_CLASSES
            or self._num_classes == NUM_CIFAR100_CLASSES
        ):
            discrete_model = NetworkCIFAR(
                C=self._C,
                num_classes=self._num_classes,
                layers=self._layers,
                auxiliary=False,
                genotype=genotype,
            )
        elif self._num_classes == NUM_IMAGENET_CLASSES:
            discrete_model = NetworkImageNet(
                C=self._C,
                num_classes=self._num_classes,
                layers=self._layers,
                auxiliary=False,
                genotype=genotype,
            )
        else:
            raise ValueError(
                "number of classes is not a valid number of any of the datasets"
            )
        #         discrete_model = Network(
        #             C=self._C,
        #             num_classes=self._num_classes,
        #             layers=self._layers,
        #             criterion=self._criterion,  # TODO: what is this
        #             steps=self._steps,
        #             multiplier=self._multiplier,
        #             stem_multiplier=self.stem[-1].num_features,  # type: ignore
        #             edge_normalization=False,
        #             discretized=True,
        #             is_baby_darts=self.is_baby_darts,
        #         )
        #         for cell in discrete_model.cells:
        #             if cell.reduction:
        #                 cell._discretize(genotype)  # type: ignore
        #             else:
        #                 cell._discretize(genotype)  # type: ignore
        #         discrete_model._arch_parameters = None  # type: ignore

        discrete_model.to(next(self.parameters()).device)

        return discrete_model

    def model_weight_parameters(self) -> list[nn.Parameter]:
        params = set(self.parameters())
        params -= set(self._betas)
        if self._arch_parameters != [None]:
            params -= set(self.alphas_reduce)
            params -= set(self.alphas_normal)
        return list(params)
