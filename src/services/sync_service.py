"""Master-data synchronization and analysis record upload service.

Responsibilities
----------------
1. **Pull catalogs** — download master data from the API and upsert into
   local SQLite tables (agricultural_units, agricultural_campaigns,
   modules, shifts, batches).
2. **Push analysis** — send locally stored ``AnalysisRecord`` rows that
   have ``is_synced = False`` to the ``/analysis/sync`` endpoint and mark
   them as synced when acknowledged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import httpx
from sqlmodel import select

from src.db import get_session
from src.model.analysis import AnalysisRecord
from src.model.catalogs import (
    AgriculturalCampaign,
    AgriculturalUnit,
    Batch,
    Module,
    Shift,
)
from src.model.production import ProductionUnit
from src.services.api_client import ApiClient

logger = logging.getLogger(__name__)


# ── Result DTOs ─────────────────────────────────────────────


@dataclass
class CatalogSyncResult:
    """Outcome of a catalog pull operation."""

    success: bool
    detail: str = ""
    agricultural_units: int = 0
    agricultural_campaigns: int = 0
    modules: int = 0
    shifts: int = 0
    batches: int = 0
    production_units: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class AnalysisSyncResult:
    """Outcome of pushing analysis records to the API."""

    success: bool
    total_sent: int = 0
    synced_count: int = 0
    failed_count: int = 0
    detail: str = ""


# ── Service ─────────────────────────────────────────────────


class SyncService:
    """Orchestrates data synchronization between local DB and remote API."""

    def __init__(self, access_token: str) -> None:
        self._api = ApiClient(access_token=access_token)

    # ────────────────────────────────────────────────────────
    # 1. Pull master-data catalogs
    # ────────────────────────────────────────────────────────

    def sync_catalogs(self) -> CatalogSyncResult:
        """Download all five catalogs and upsert into SQLite.

        Each catalog is fetched independently so that a failure in one
        does not block the others.
        """
        result = CatalogSyncResult(success=True)

        # Agricultural Units
        try:
            data = self._api.get("/agricultural-units")
            result.agricultural_units = self._upsert_agricultural_units(data)
        except Exception as exc:
            result.errors.append(f"agricultural_units: {exc}")
            logger.warning("Failed to sync agricultural_units: %s", exc)

        # Agricultural Campaigns
        try:
            data = self._api.get("/agricultural-campaigns")
            result.agricultural_campaigns = self._upsert_agricultural_campaigns(data)
        except Exception as exc:
            result.errors.append(f"agricultural_campaigns: {exc}")
            logger.warning("Failed to sync agricultural_campaigns: %s", exc)

        # Modules
        try:
            data = self._api.get("/modules")
            result.modules = self._upsert_modules(data)
        except Exception as exc:
            result.errors.append(f"modules: {exc}")
            logger.warning("Failed to sync modules: %s", exc)

        # Shifts
        try:
            data = self._api.get("/shifts")
            result.shifts = self._upsert_shifts(data)
        except Exception as exc:
            result.errors.append(f"shifts: {exc}")
            logger.warning("Failed to sync shifts: %s", exc)

        # Batches
        try:
            data = self._api.get("/batches")
            result.batches = self._upsert_batches(data)
        except Exception as exc:
            result.errors.append(f"batches: {exc}")
            logger.warning("Failed to sync batches: %s", exc)

        # Production Units (join table for cascading dropdowns)
        try:
            data = self._api.get("/production-units")
            result.production_units = self._upsert_production_units(data)
        except Exception as exc:
            result.errors.append(f"production_units: {exc}")
            logger.warning("Failed to sync production_units: %s", exc)

        if result.errors:
            result.success = len(result.errors) < 6  # partial success
            result.detail = f"{len(result.errors)} catalogo(s) con error."
        else:
            total = (
                result.agricultural_units
                + result.agricultural_campaigns
                + result.modules
                + result.shifts
                + result.batches
                + result.production_units
            )
            result.detail = f"{total} registros sincronizados correctamente."

        return result

    # ────────────────────────────────────────────────────────
    # 2. Push analysis records
    # ────────────────────────────────────────────────────────

    def sync_analysis_records(self) -> AnalysisSyncResult:
        """Send all un-synced AnalysisRecord rows to the API.

        The API endpoint ``POST /analysis/sync`` is **idempotent** — it
        accepts a list of records and returns the ``sync_id``s that are
        now persisted on the server.  We mark those rows locally.
        """
        with get_session() as session:
            stmt = select(AnalysisRecord).where(AnalysisRecord.is_synced == False)  # noqa: E712
            pending = session.exec(stmt).all()

            if not pending:
                return AnalysisSyncResult(
                    success=True,
                    detail="No hay registros pendientes de sincronizar.",
                )

            # Build payload
            payload = [
                {
                    "sync_id": r.sync_id,
                    "input_image_name": r.input_image_name,
                    "fruit_id": r.fruit_id,
                    "detection_accuracy": r.detection_accuracy,
                    "quality_grade_id": 1,  # placeholder — adapt as needed
                    "quality_accuracy": r.quality_accuracy,
                    "maturity_grade_id": 1,  # placeholder — adapt as needed
                    "maturity_accuracy": r.maturity_accuracy,
                    "production_unit_id": r.production_unit_id,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "registered_by_user_id": None,
                }
                for r in pending
            ]

            try:
                resp = self._api.post("/analysis/sync", json=payload)
                synced_ids: list[str] = resp.get("synced_ids", [])
                failed_ids: list[str] = resp.get("failed_ids", [])

                # Mark synced records
                for record in pending:
                    if record.sync_id in synced_ids:
                        record.is_synced = True
                        session.add(record)

                session.commit()

                return AnalysisSyncResult(
                    success=True,
                    total_sent=len(pending),
                    synced_count=len(synced_ids),
                    failed_count=len(failed_ids),
                    detail=f"{len(synced_ids)}/{len(pending)} registros sincronizados.",
                )

            except httpx.HTTPStatusError as exc:
                return AnalysisSyncResult(
                    success=False,
                    total_sent=len(pending),
                    detail=f"Error HTTP {exc.response.status_code}.",
                )
            except (httpx.ConnectError, httpx.TimeoutException):
                return AnalysisSyncResult(
                    success=False,
                    total_sent=len(pending),
                    detail="Sin conexion a internet.",
                )
            except Exception as exc:
                logger.exception("Analysis sync error")
                return AnalysisSyncResult(
                    success=False,
                    total_sent=len(pending),
                    detail=str(exc),
                )

    # ────────────────────────────────────────────────────────
    # Private upsert helpers
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _upsert_agricultural_units(data: list[dict]) -> int:
        count = 0
        with get_session() as session:
            for item in data:
                existing = session.get(AgriculturalUnit, item["id"])
                if existing:
                    existing.name = item["name"]
                    session.add(existing)
                else:
                    session.add(AgriculturalUnit(id=item["id"], name=item["name"]))
                count += 1
            session.commit()
        return count

    @staticmethod
    def _upsert_agricultural_campaigns(data: list[dict]) -> int:
        count = 0
        with get_session() as session:
            for item in data:
                existing = session.get(AgriculturalCampaign, item["id"])
                if existing:
                    existing.name = item["name"]
                    existing.start_date = date.fromisoformat(item["start_date"])
                    existing.end_date = date.fromisoformat(item["end_date"])
                    session.add(existing)
                else:
                    session.add(
                        AgriculturalCampaign(
                            id=item["id"],
                            name=item["name"],
                            start_date=date.fromisoformat(item["start_date"]),
                            end_date=date.fromisoformat(item["end_date"]),
                        )
                    )
                count += 1
            session.commit()
        return count

    @staticmethod
    def _upsert_modules(data: list[dict]) -> int:
        count = 0
        with get_session() as session:
            for item in data:
                existing = session.get(Module, item["id"])
                if existing:
                    existing.name = item["name"]
                    session.add(existing)
                else:
                    session.add(Module(id=item["id"], name=item["name"]))
                count += 1
            session.commit()
        return count

    @staticmethod
    def _upsert_shifts(data: list[dict]) -> int:
        count = 0
        with get_session() as session:
            for item in data:
                existing = session.get(Shift, item["id"])
                if existing:
                    existing.name = item["name"]
                    session.add(existing)
                else:
                    session.add(Shift(id=item["id"], name=item["name"]))
                count += 1
            session.commit()
        return count

    @staticmethod
    def _upsert_batches(data: list[dict]) -> int:
        count = 0
        with get_session() as session:
            for item in data:
                existing = session.get(Batch, item["id"])
                if existing:
                    existing.name = item["name"]
                    session.add(existing)
                else:
                    session.add(Batch(id=item["id"], name=item["name"]))
                count += 1
            session.commit()
        return count

    @staticmethod
    def _upsert_production_units(data: list[dict]) -> int:
        count = 0
        with get_session() as session:
            for item in data:
                existing = session.get(ProductionUnit, item["id"])
                if existing:
                    existing.agricultural_campaign_id = item["agricultural_campaign_id"]
                    existing.agricultural_unit_id = item["agricultural_unit_id"]
                    existing.module_id = item["module_id"]
                    existing.shift_id = item["shift_id"]
                    existing.batch_id = item["batch_id"]
                    session.add(existing)
                else:
                    session.add(
                        ProductionUnit(
                            id=item["id"],
                            agricultural_campaign_id=item["agricultural_campaign_id"],
                            agricultural_unit_id=item["agricultural_unit_id"],
                            module_id=item["module_id"],
                            shift_id=item["shift_id"],
                            batch_id=item["batch_id"],
                        )
                    )
                count += 1
            session.commit()
        return count

    # ────────────────────────────────────────────────────────
    # Helpers for local counts (UI display)
    # ────────────────────────────────────────────────────────

    @staticmethod
    def get_local_counts() -> dict[str, int]:
        """Return the number of rows in each catalog table."""
        with get_session() as session:
            return {
                "agricultural_units": len(session.exec(select(AgriculturalUnit)).all()),
                "agricultural_campaigns": len(
                    session.exec(select(AgriculturalCampaign)).all()
                ),
                "modules": len(session.exec(select(Module)).all()),
                "shifts": len(session.exec(select(Shift)).all()),
                "batches": len(session.exec(select(Batch)).all()),
            }

    @staticmethod
    def get_pending_analysis_count() -> int:
        """Return how many analysis records are waiting to be synced."""
        with get_session() as session:
            stmt = select(AnalysisRecord).where(AnalysisRecord.is_synced == False)  # noqa: E712
            return len(session.exec(stmt).all())
