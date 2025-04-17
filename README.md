# Configurable Optimizer

[![confopt](https://i.postimg.cc/xCsTZyrM/diagram-20250403.png)](https://postimg.cc/7G2kGzZZ)

Break down one-shot optimizers into their core ideas, modularize them, and then search the space of optimizers for the best one.

[![API Docs](https://img.shields.io/badge/docs-API-blue)](https://automl.github.io/ConfigurableOptimizer/index.html)


## Installation and Development
First, install the dependencies required for development and testing in your environment. You can skip running the last line as it might take long.

```
conda create -n confopt python=3.9
conda activate confopt
pip install -e ".[dev, test]"
pip install -e ".[benchmark]"
```

Install the precommit hooks
```
pre-commit install
```

Run the tests
```
pytest tests
```

Run with the slow benchmark tests
```
pytest --benchmark tests
```

Try running an example
```
python examples/demo_light.py
```

This project uses `mypy` for type checking, `ruff` for linting, and `black` for formatting. VSCode extensions can be found for each of these tools. The pre-commit hooks check for `mypy`/`ruff`/`black` errors and won't let you commit until you fix the issues. The pre-commit hooks also checks for proper commit message format.

The easiest way to ensure that the commits are well formatted is to commit using `cz commit` instead of `git commit`.

## Getting Started

We define modular, differentiable NAS components within our library. Below is a snippet that demonstrates how we run a vanilla-DARTS experiment. We recommend exploring the [demo-notebook](examples/notebooks/demo_notebook.ipynb) for a hands-on experience with the library.


```python 
from confopt.profile import DARTSProfile
from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType

profile = DARTSProfile(
        searchspace_type=SearchSpaceType.DARTS,
        epochs=3
)

experiment = Experiment(
    search_space=SearchSpaceType.DARTS,
    dataset=DatasetType.CIFAR10,
)

experiment.train_supernet(profile)

```

## Reproduce

To reproduce our workflow for the paper- 
***`confopt`** : A Library for Implementation and Evaluation of Gradient-based One-Shot NAS Methods*, checkout the **[reproducing docs](scripts/benchsuite_experiments/README.md)**.
