"""
File modify from https://github.com/open-mmlab/mmcv
License File Available at:
https://github.com/open-mmlab/mmcv/blob/master/LICENSE
"""
from enum import Enum


class Priority(Enum):
    """Hook priority levels.
    +------------+------------+
    | Level      | Value      |
    +============+============+
    | HIGHEST    | 0          |
    +------------+------------+
    | VERY_HIGH  | 10         |
    +------------+------------+
    | HIGH       | 30         |
    +------------+------------+
    | NORMAL     | 50         |
    +------------+------------+
    | LOW        | 70         |
    +------------+------------+
    | VERY_LOW   | 90         |
    +------------+------------+
    | LOWEST     | 100        |
    +------------+------------+
    """

    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 30
    NORMAL = 50
    LOW = 70
    VERY_LOW = 90
    LOWEST = 100


def get_priority(priority):
    """Get priority value.
    Args:
        priority (int or str or :obj:`Priority`): Priority.
    Returns:
        int: The priority value.
    """
    if isinstance(priority, str):
        return Priority[priority.upper()].value
    else:
        raise TypeError('priority must be an integer or Priority enum value')
