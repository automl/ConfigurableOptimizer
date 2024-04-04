# Configurable Optimzer

Break down one-shot optimizers into their core ideas, modularize them, and then search the space of optimizers for the best one.

## Installation and Development
First, install the dependencies required for development and testing in your environment.

```
conda create -n confopt python=3.9
conda activate confopt
pip install -e ".[benchmark, dev, test]"
```

Install the precommit hooks
```
pre-commit install
```

Run the tests
```
pytest tests
```

Try running an example
```
python examples/searchspace.py
```

This project uses `mypy` for type checking, `ruff` for linting, and `black` for formatting. VSCode extensions can be found for each of these tools. The pre-commit hooks check for `mypy`/`ruff`/`black` errors and won't let you commit until you fix the issues. The pre-commit hooks also checks for proper commit message format.

The easiest way to ensure that the commits are well formatted is to commit using `cz commit` instead of `git commit`.
