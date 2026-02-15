from pydantic import BaseModel, ConfigDict, Field
import numpy as np

from src.schemas.crop_info import CropInfo
from src.schemas.ripeness_counts import RipnessCounts


class ProcessingResult(BaseModel):
    """Resultado completo de procesar un video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: str
    in_count: int = 0
    out_count: int = 0
    ripeness_counts: RipnessCounts = Field(default_factory=RipnessCounts.empty)
    in_ripeness: RipnessCounts = Field(default_factory=RipnessCounts.empty)
    out_ripeness: RipnessCounts = Field(default_factory=RipnessCounts.empty)
    frames_processed: int = 0
    total_frames: int = 0
    crops: list[CropInfo] = Field(default_factory=list)

    # Crop images stored separately (not serialized)
    crop_images: list[np.ndarray] = Field(default_factory=list, exclude=True)

    @property
    def total(self) -> int:
        return self.in_count + self.out_count
