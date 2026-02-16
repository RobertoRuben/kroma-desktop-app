"""Colores centralizados — unica fuente de verdad para ripeness y quality colors."""

from dataclasses import dataclass

from src.enums.quality_class import QualityClass
from src.enums.ripeness_class import RipenessClass


@dataclass(frozen=True)
class ClassColors:
    """Colores para una clase de clasificacion en distintos formatos."""
    bgr: tuple[int, int, int]     # OpenCV (anotacion de video)
    hex: str                       # UI hex color
    text_color: str                # Color de texto sobre el badge


# --- Ripeness ---

RIPENESS_COLORS: dict[RipenessClass, ClassColors] = {
    RipenessClass.GREEN: ClassColors(bgr=(0, 180, 0), hex="#43A047", text_color="#FFFFFF"),
    RipenessClass.RED: ClassColors(bgr=(0, 0, 220), hex="#E53935", text_color="#FFFFFF"),
    RipenessClass.TURNING: ClassColors(bgr=(0, 165, 255), hex="#FB8C00", text_color="#FFFFFF"),
    RipenessClass.BROWN: ClassColors(bgr=(42, 42, 165), hex="#8D6E63", text_color="#FFFFFF"),
}

ONNX_CLS_INDEX: dict[int, RipenessClass] = {
    0: RipenessClass.BROWN,
    1: RipenessClass.GREEN,
    2: RipenessClass.RED,
    3: RipenessClass.TURNING,
}

RIPENESS_BGR: dict[str, tuple[int, int, int]] = {
    cls.value: colors.bgr for cls, colors in RIPENESS_COLORS.items()
}

ALL_RIPENESS_CLASSES: list[RipenessClass] = [
    RipenessClass.GREEN,
    RipenessClass.RED,
    RipenessClass.TURNING,
    RipenessClass.BROWN,
]

# --- Quality ---

QUALITY_COLORS: dict[QualityClass, ClassColors] = {
    QualityClass.GOOD: ClassColors(bgr=(0, 180, 0), hex="#43A047", text_color="#FFFFFF"),
    QualityClass.CRACKED: ClassColors(bgr=(0, 100, 255), hex="#FF6D00", text_color="#FFFFFF"),
    QualityClass.DECAY: ClassColors(bgr=(0, 0, 180), hex="#D32F2F", text_color="#FFFFFF"),
    QualityClass.DEHYDRATED: ClassColors(bgr=(0, 140, 200), hex="#F9A825", text_color="#000000"),
    QualityClass.SUNSCALD: ClassColors(bgr=(80, 127, 255), hex="#FF7043", text_color="#FFFFFF"),
}

ONNX_QUALITY_INDEX: dict[int, QualityClass] = {
    0: QualityClass.CRACKED,
    1: QualityClass.DECAY,
    2: QualityClass.DEHYDRATED,
    3: QualityClass.GOOD,
    4: QualityClass.SUNSCALD,
}

QUALITY_BGR: dict[str, tuple[int, int, int]] = {
    cls.value: colors.bgr for cls, colors in QUALITY_COLORS.items()
}

ALL_QUALITY_CLASSES: list[QualityClass] = [
    QualityClass.GOOD,
    QualityClass.CRACKED,
    QualityClass.DECAY,
    QualityClass.DEHYDRATED,
    QualityClass.SUNSCALD,
]
