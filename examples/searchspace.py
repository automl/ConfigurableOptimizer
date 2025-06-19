from __future__ import annotations

import torch
from confopt.searchspace import NASBench201SearchSpace

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    supernet = NASBench201SearchSpace()
    compiled_model = torch.compile(supernet)  # PyTorch 2.0 only

    x = torch.randn(2, 3, 32, 32).to(device)
    out, logits = supernet(x)

    print(out.shape)
    print(logits.shape)
    print("Done.")
