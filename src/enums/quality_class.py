"""Enum de clases de calidad de pimientos."""

from enum import Enum


class QualityClass(str, Enum):
    CRACKED = "cracked"
    DECAY = "decay"
    DEHYDRATED = "dehydrated"
    GOOD = "good"
    SUNSCALD = "sunscald"
