from __future__ import annotations

from reinmax import reinmax as rmx
import torch

from confopt.oneshot.archsampler import GDASSampler


class ReinMaxSampler(GDASSampler):
    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        tau = self.tau_curr.to(alpha.device)  # type: ignore
        hardwts, _ = rmx(alpha, tau[0])
        index = hardwts.max(-1, keepdim=True)[1]  # noqa: F841

        return hardwts
