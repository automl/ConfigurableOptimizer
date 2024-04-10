from __future__ import annotations

import re
import time


def get_time_as_string() -> str:
    """Gets the current date and time as a string."""
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return time_str


def check_date_format(input_string: str) -> bool:
    pattern = r"^\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}.\d{3}$"
    return re.match(pattern, input_string) is not None


def get_runtime() -> str:
    current_time = time.time()
    milliseconds = int((current_time - int(current_time)) * 1000)

    runtime = (
        time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime(current_time))
        + f".{milliseconds:03d}"
    )
    return runtime
