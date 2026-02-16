"""Service layer for cascading production-unit queries.

Provides static methods that combine catalog lookups with filtering
through the ``production_units`` join table to support cascading
dropdown selection.

- **Campaign** (level 1): loaded directly from catalog (top-level).
- **Fundo** (level 2): filtered by selected campaign via ``production_units``.
- **Module** (level 3): filtered by campaign + fundo via ``production_units``.
- **Shift** (level 4): filtered by campaign + fundo + module via ``production_units``.
- **Batch** (level 5): filtered by all four parents via ``production_units``.
"""

from __future__ import annotations

import logging
import time

from sqlalchemy import distinct
from sqlmodel import select

from src.db import get_session
from src.model.catalogs import (
    AgriculturalCampaign,
    AgriculturalUnit,
    Batch,
    Module,
    Shift,
)
from src.model.production import ProductionUnit

logger = logging.getLogger(__name__)


class ProductionUnitService:
    """Cascading look-ups driven by the ``production_units`` table."""

    # ── 1. Campaigns (top-level, no filter) ────────────────

    @staticmethod
    def get_campaigns() -> list[dict]:
        """Return all campaigns from the catalog."""
        with get_session() as session:
            stmt = select(AgriculturalCampaign).order_by(AgriculturalCampaign.name)
            rows = session.exec(stmt).all()
            return [{"id": r.id, "name": r.name} for r in rows]

    # ── 2. Fundos filtered by campaign ─────────────────────

    @staticmethod
    def get_fundos(campaign_id: int) -> list[dict]:
        """Return fundos that appear in production_units for *campaign_id*."""
        with get_session() as session:
            sub = select(distinct(ProductionUnit.agricultural_unit_id)).where(
                ProductionUnit.agricultural_campaign_id == campaign_id
            )
            stmt = (
                select(AgriculturalUnit)
                .where(AgriculturalUnit.id.in_(sub))  # type: ignore[union-attr]
                .order_by(AgriculturalUnit.name)
            )
            rows = session.exec(stmt).all()
            return [{"id": r.id, "name": r.name} for r in rows]

    # ── 3. Modules filtered by campaign + fundo ────────────

    @staticmethod
    def get_modules(campaign_id: int, unit_id: int) -> list[dict]:
        with get_session() as session:
            sub = (
                select(distinct(ProductionUnit.module_id))
                .where(ProductionUnit.agricultural_campaign_id == campaign_id)
                .where(ProductionUnit.agricultural_unit_id == unit_id)
            )
            stmt = (
                select(Module)
                .where(Module.id.in_(sub))  # type: ignore[union-attr]
                .order_by(Module.name)
            )
            rows = session.exec(stmt).all()
            return [{"id": r.id, "name": r.name} for r in rows]

    # ── 4. Shifts filtered by campaign + fundo + module ────

    @staticmethod
    def get_shifts(campaign_id: int, unit_id: int, module_id: int) -> list[dict]:
        with get_session() as session:
            sub = (
                select(distinct(ProductionUnit.shift_id))
                .where(ProductionUnit.agricultural_campaign_id == campaign_id)
                .where(ProductionUnit.agricultural_unit_id == unit_id)
                .where(ProductionUnit.module_id == module_id)
            )
            stmt = (
                select(Shift)
                .where(Shift.id.in_(sub))  # type: ignore[union-attr]
                .order_by(Shift.name)
            )
            rows = session.exec(stmt).all()
            return [{"id": r.id, "name": r.name} for r in rows]

    # ── 5. Batches filtered by all four parents ────────────

    @staticmethod
    def get_batches(
        campaign_id: int,
        unit_id: int,
        module_id: int,
        shift_id: int,
    ) -> list[dict]:
        with get_session() as session:
            sub = (
                select(distinct(ProductionUnit.batch_id))
                .where(ProductionUnit.agricultural_campaign_id == campaign_id)
                .where(ProductionUnit.agricultural_unit_id == unit_id)
                .where(ProductionUnit.module_id == module_id)
                .where(ProductionUnit.shift_id == shift_id)
            )
            stmt = (
                select(Batch)
                .where(Batch.id.in_(sub))  # type: ignore[union-attr]
                .order_by(Batch.name)
            )
            rows = session.exec(stmt).all()
            return [{"id": r.id, "name": r.name} for r in rows]

    # ── 6. Find-or-create production unit ──────────────────

    @staticmethod
    def resolve_production_unit_id(
        campaign_id: int,
        unit_id: int,
        module_id: int,
        shift_id: int,
        batch_id: int,
    ) -> int | None:
        """Find or create the ``ProductionUnit`` for the exact 5-FK combo."""
        with get_session() as session:
            stmt = (
                select(ProductionUnit)
                .where(ProductionUnit.agricultural_campaign_id == campaign_id)
                .where(ProductionUnit.agricultural_unit_id == unit_id)
                .where(ProductionUnit.module_id == module_id)
                .where(ProductionUnit.shift_id == shift_id)
                .where(ProductionUnit.batch_id == batch_id)
            )
            existing = session.exec(stmt).first()
            if existing:
                logger.info("Found existing production_unit id=%s", existing.id)
                return existing.id

            # BIGINT PKs don't auto-increment in SQLite
            local_id = int(time.time() * 1_000_000)
            new_pu = ProductionUnit(
                id=local_id,
                agricultural_campaign_id=campaign_id,
                agricultural_unit_id=unit_id,
                module_id=module_id,
                shift_id=shift_id,
                batch_id=batch_id,
            )
            session.add(new_pu)
            session.commit()
            session.refresh(new_pu)
            logger.info("Created new production_unit id=%s", new_pu.id)
            return new_pu.id
