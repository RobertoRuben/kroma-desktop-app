from typing import TYPE_CHECKING
from sqlmodel import (
    BIGINT,
    Column,
    Field,
    ForeignKey,
    Relationship,
    SQLModel,
    UniqueConstraint,
)
from .base import BaseTimestampMixin

if TYPE_CHECKING:
    from .catalogs import AgriculturalCampaign, AgriculturalUnit
    from .analysis import AnalysisRecord


class ProductionUnit(BaseTimestampMixin, SQLModel, table=True):
    """Combines various master data into a single operational context.

    Attributes:
        id: Primary key (64-bit integer).
        agricultural_campaign_id: FK to the campaign.
        agricultural_unit_id: FK to the farm.
        module_id: Identifier for the specific production module.
        shift_id: Identifier for the work shift.
        batch_id: Identifier for the specific produce batch.
    """

    __tablename__ = "production_units"
    __table_args__ = (
        UniqueConstraint(
            "agricultural_campaign_id",
            "agricultural_unit_id",
            "module_id",
            "shift_id",
            "batch_id",
            name="ix_production_units_unique_context",
        ),
    )

    id: int | None = Field(default=None, sa_column=Column(BIGINT, primary_key=True))

    agricultural_campaign_id: int = Field(
        sa_column=Column(
            BIGINT,
            ForeignKey("agricultural_campaigns.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    agricultural_unit_id: int = Field(
        sa_column=Column(
            BIGINT,
            ForeignKey("agricultural_units.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    module_id: int = Field(sa_column=Column(BIGINT, nullable=False))
    shift_id: int = Field(sa_column=Column(BIGINT, nullable=False))
    batch_id: int = Field(sa_column=Column(BIGINT, nullable=False))

    agricultural_campaign: "AgriculturalCampaign" = Relationship(
        back_populates="production_units"
    )
    agricultural_unit: "AgriculturalUnit" = Relationship(
        back_populates="production_units"
    )
    analysis_records: list["AnalysisRecord"] = Relationship(
        back_populates="production_unit"
    )
