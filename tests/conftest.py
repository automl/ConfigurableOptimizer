from typing import Any
import pytest

def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--benchmark", action="store", help="Run tests marked with the provided marker"
    )

    parser.addoption(
        "--experimental", action="store", help="Run tests marked with the provided marker"
    )

def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "env(name): mark test to run only on named environment"
    )

def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    if config.getoption("--benchmark") or config.getoption("--experimental"):
        return
    skip_benchmark = pytest.mark.skip(
        reason="Test is skipped unless the benchmark marker is provided"
    )
    skip_experimental = pytest.mark.skip(
        reason="Test is skipped unless the under-dev marker is provided"
    )
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
        if "experimental" in item.keywords:
            item.add_marker(skip_experimental)