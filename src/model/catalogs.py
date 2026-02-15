from datetime import date
from typing import TYPE_CHECKING
from sqlmodel import BIGINT, Column, Field, Relationship, SQLModel, String
from .base import BaseTimestampMixin

if TYPE_CHECKING:
    from .production import ProductionUnit


class AgriculturalUnit(BaseTimestampMixin, SQLModel, table=True):
    """Master data for agricultural units (farms or fundos)."""

    __tablename__ = "agricultural_units"

    id: int | None = Field(default=None, sa_column=Column(BIGINT, primary_key=True))
    name: str = Field(sa_column=Column(String(255), unique=True, nullable=False))

    production_units: list["ProductionUnit"] = Relationship(
        back_populates="agricultural_unit"
    )


class AgriculturalCampaign(BaseTimestampMixin, SQLModel, table=True):
    """Master data for agricultural campaigns."""

    __tablename__ = "agricultural_campaigns"

    id: int | None = Field(default=None, sa_column=Column(BIGINT, primary_key=True))
    name: str = Field(sa_column=Column(String(255), unique=True, nullable=False))
    start_date: date = Field(nullable=False)
    end_date: date = Field(nullable=False)

    production_units: list["ProductionUnit"] = Relationship(
        back_populates="agricultural_campaign"
    )


class Module(BaseTimestampMixin, SQLModel, table=True):
    """Master data for production modules."""

    __tablename__ = "modules"

    id: int | None = Field(default=None, sa_column=Column(BIGINT, primary_key=True))
    name: str = Field(sa_column=Column(String(255), unique=True, nullable=False))

    production_units: list["ProductionUnit"] = Relationship(back_populates="module")


class Shift(BaseTimestampMixin, SQLModel, table=True):
    """Master data for work shifts."""

    __tablename__ = "shifts"

    id: int | None = Field(default=None, sa_column=Column(BIGINT, primary_key=True))
    name: str = Field(sa_column=Column(String(255), unique=True, nullable=False))

    production_units: list["ProductionUnit"] = Relationship(back_populates="shift")


class Batch(BaseTimestampMixin, SQLModel, table=True):
    """Master data for production batches."""

    __tablename__ = "batches"

    id: int | None = Field(default=None, sa_column=Column(BIGINT, primary_key=True))
    name: str = Field(sa_column=Column(String(255), unique=True, nullable=False))

    production_units: list["ProductionUnit"] = Relationship(back_populates="batch")
