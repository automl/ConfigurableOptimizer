from __future__ import annotations

import re
import time


def get_time_as_string() -> str:
    """Gets the current date and time as a string."""
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return time_str


def check_date_format(input_string: str) -> bool:
    pattern = r"^\d{4}-\d{2}-[A-Za-z]{3}-\d{2}:\d{2}:\d{2}$"
    return re.match(pattern, input_string) is not None
