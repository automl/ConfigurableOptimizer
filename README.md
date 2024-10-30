# Configurable Optimizer

Break down one-shot optimizers into their core ideas, modularize them, and then search the space of optimizers for the best one.

## Installation and Development
First, install the dependencies required for development and testing in your environment.

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
python examples/searchspace.py
```

This project uses `mypy` for type checking, `ruff` for linting, and `black` for formatting. VSCode extensions can be found for each of these tools. The pre-commit hooks check for `mypy`/`ruff`/`black` errors and won't let you commit until you fix the issues. The pre-commit hooks also checks for proper commit message format.

The easiest way to ensure that the commits are well formatted is to commit using `cz commit` instead of `git commit`.

To download the Taskonomy dataset from Stanford urls run:
```
bash scripts/download_taskonomy_stanford.sh
```
Or alternatively from the EPFL urls:
```
bash scripts/download_taskonomy_epfl.sh
```
Extract the ```taskonomydata_mini.zip``` into ```datasets/``` directory. Eventually you would have a directory ```datasets/taskonomydata_mini``` which contains all images to be sampled from the each of the domains along with their data splits.