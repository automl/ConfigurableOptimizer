from __future__ import annotations

import torch.nn as nn  # noqa: PLR0402


def get_modules_of_type(module: nn.Module, module_type: type) -> list[nn.Module]:
    modules = []
    for m in module.modules():
        if isinstance(m, module_type):
            modules.append(m)
    return modules
