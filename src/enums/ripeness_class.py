from enum import Enum


class RipenessClass(str, Enum):
    GREEN = "green"
    RED = "red"
    TURNING = "turning"
    BROWN = "brown"
