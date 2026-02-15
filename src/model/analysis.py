from typing import TYPE_CHECKING
from sqlmodel import (
    BIGINT,
    Column,
    Field,
    ForeignKey,
    Relationship,
    SQLModel,
    String,
    Float,
    Boolean,
)
from .base import BaseTimestampMixin

if TYPE_CHECKING:
    from .production import ProductionUnit


class AnalysisRecord(BaseTimestampMixin, SQLModel, table=True):
    """Transactional record for a single pepper classification inference.

    Attributes:
        sync_id: Client-generated UUID (v4) for idempotent API synchronization.
        input_image_name: Name of the processed image file.
        fruit_id: Internal ID of the detected fruit within the frame.
        detection_accuracy: Confidence score for the object detection.
        quality_accuracy: Confidence score for the quality classification.
        maturity_accuracy: Confidence score for the maturity classification.
        is_synced: Boolean flag to track if the record reached the cloud API.
    """

    __tablename__ = "analysis_records"

    id: int | None = Field(default=None, sa_column=Column(BIGINT, primary_key=True))
    sync_id: str = Field(
        sa_column=Column(String(36), unique=True, index=True, nullable=False)
    )
    input_image_name: str = Field(sa_column=Column(String(500), nullable=False))
    fruit_id: int = Field(sa_column=Column(BIGINT, nullable=False))

    # ML Inference Results
    detection_accuracy: float = Field(sa_column=Column(Float, nullable=False))
    quality_accuracy: float = Field(sa_column=Column(Float, nullable=False))
    maturity_accuracy: float = Field(sa_column=Column(Float, nullable=False))

    production_unit_id: int = Field(
        sa_column=Column(
            BIGINT,
            ForeignKey("production_units.id", ondelete="CASCADE"),
            nullable=False,
        )
    )

    # Telemetry and Edge Control
    latitude: float = Field(sa_column=Column(Float, nullable=False))
    longitude: float = Field(sa_column=Column(Float, nullable=False))
    local_image_path: str = Field(sa_column=Column(String(1000), nullable=False))
    is_synced: bool = Field(
        default=False, sa_column=Column(Boolean, index=True, nullable=False)
    )

    production_unit: "ProductionUnit" = Relationship(back_populates="analysis_records")
