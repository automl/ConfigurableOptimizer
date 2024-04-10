from __future__ import annotations

from collections import namedtuple

from .logger import Logger


def dict2config(xdict: dict, logger: Logger) -> tuple:
    assert isinstance(xdict, dict), f"invalid type : {type(xdict)}"
    Arguments = namedtuple(  # type: ignore
        "Configure", " ".join(xdict.keys())  # type:ignore
    )
    content = Arguments(**xdict)
    if hasattr(logger, "log"):
        logger.log(f"{content}")
    return content
