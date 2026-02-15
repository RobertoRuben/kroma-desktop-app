from pydantic import BaseModel, ConfigDict, Field
import numpy as np


class VideoInfo(BaseModel):
    """Informacion de un video cargado."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    rotation: int = 0
    hdr_transfer: str | None = None
    width: int = 0
    height: int = 0
    fps: int = 0
    total_frames: int = 0
    first_frame: np.ndarray | None = Field(default=None, exclude=True)
