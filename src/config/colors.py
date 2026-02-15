"""Colores centralizados — unica fuente de verdad para ripeness colors."""

from dataclasses import dataclass

from src.enums.ripeness_class import RipenessClass


@dataclass(frozen=True)
class RipenessColors:
    """Colores para una clase de madurez en distintos formatos."""
    bgr: tuple[int, int, int]     # OpenCV (anotacion de video)
    hex: str                       # UI hex color
    text_color: str                # Color de texto sobre el badge


RIPENESS_COLORS: dict[RipenessClass, RipenessColors] = {
    RipenessClass.GREEN: RipenessColors(bgr=(0, 180, 0), hex="#43A047", text_color="#FFFFFF"),
    RipenessClass.RED: RipenessColors(bgr=(0, 0, 220), hex="#E53935", text_color="#FFFFFF"),
    RipenessClass.TURNING: RipenessColors(bgr=(0, 165, 255), hex="#FB8C00", text_color="#FFFFFF"),
    RipenessClass.BROWN: RipenessColors(bgr=(42, 42, 165), hex="#8D6E63", text_color="#FFFFFF"),
}

# Mapeo indice ONNX -> clase (salida del modelo de clasificacion)
ONNX_CLS_INDEX: dict[int, RipenessClass] = {
    0: RipenessClass.BROWN,
    1: RipenessClass.GREEN,
    2: RipenessClass.RED,
    3: RipenessClass.TURNING,
}

# Colores BGR para anotacion de video (acceso rapido)
RIPENESS_BGR: dict[str, tuple[int, int, int]] = {
    cls.value: colors.bgr for cls, colors in RIPENESS_COLORS.items()
}

# Orden de clases para UI
ALL_RIPENESS_CLASSES: list[RipenessClass] = [
    RipenessClass.GREEN,
    RipenessClass.RED,
    RipenessClass.TURNING,
    RipenessClass.BROWN,
]
