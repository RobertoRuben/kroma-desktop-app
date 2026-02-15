from pydantic import BaseModel, ConfigDict, Field
import numpy as np

from src.enums import Direction, RipenessClass, SyncStatus


class CropInfo(BaseModel):
    """Metadata de un crop extraido durante el procesamiento."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    track_id: int
    frame_number: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    ripeness: RipenessClass
    ripeness_conf: float = Field(ge=0.0, le=1.0)
    direction: Direction = Direction.UNKNOWN
    sync_status: SyncStatus = SyncStatus.PENDING

    # Crop image stored separately (not serialized to JSON)
    image: np.ndarray | None = Field(default=None, exclude=True)
