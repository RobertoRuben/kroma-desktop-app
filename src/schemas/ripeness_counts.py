from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.enums import RipenessClass


class RipnessCounts(BaseModel):
    """Conteo de frutos por clase de madurez."""

    model_config = ConfigDict(frozen=True)

    green: int = 0
    red: int = 0
    turning: int = 0
    brown: int = 0

    @classmethod
    def empty(cls) -> RipnessCounts:
        return cls()

    def total(self) -> int:
        return self.green + self.red + self.turning + self.brown

    def increment(self, ripeness: RipenessClass) -> RipnessCounts:
        """Return a new instance with the given class incremented by 1."""
        return self.model_copy(update={ripeness.value: getattr(self, ripeness.value) + 1})

    def get(self, ripeness: RipenessClass) -> int:
        return getattr(self, ripeness.value)
