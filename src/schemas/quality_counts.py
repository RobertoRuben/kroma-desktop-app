"""Conteo de frutos por clase de calidad."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.enums import QualityClass


class QualityCounts(BaseModel):
    """Conteo de frutos por clase de calidad."""

    model_config = ConfigDict(frozen=True)

    cracked: int = 0
    decay: int = 0
    dehydrated: int = 0
    good: int = 0
    sunscald: int = 0

    @classmethod
    def empty(cls) -> QualityCounts:
        return cls()

    def total(self) -> int:
        return self.cracked + self.decay + self.dehydrated + self.good + self.sunscald

    def increment(self, quality: QualityClass) -> QualityCounts:
        """Return a new instance with the given class incremented by 1."""
        return self.model_copy(update={quality.value: getattr(self, quality.value) + 1})

    def get(self, quality: QualityClass) -> int:
        return getattr(self, quality.value)
