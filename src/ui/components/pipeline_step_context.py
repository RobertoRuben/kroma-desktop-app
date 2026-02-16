"""Pipeline Step 0 — Production context selector with cascading combos.

Presents five searchable dropdowns (Campaña → Fundo → Módulo → Turno → Lote)
that cascade-filter via the ``production_units`` join table.
When all five are selected the resolved ``production_unit_id`` is passed
to the caller through the *on_context_selected* callback.
"""

from __future__ import annotations

import logging
from typing import Callable

import flet as ft

from src.services.production_unit_service import ProductionUnitService

logger = logging.getLogger(__name__)

# ── Cascade definition ─────────────────────────────────────

_LEVELS = [
    {"key": "campaign", "label": "Campaña", "icon": ft.Icons.CALENDAR_TODAY},
    {"key": "fundo", "label": "Fundo", "icon": ft.Icons.LANDSCAPE},
    {"key": "module", "label": "Módulo", "icon": ft.Icons.GRID_VIEW},
    {"key": "shift", "label": "Turno", "icon": ft.Icons.ACCESS_TIME},
    {"key": "batch", "label": "Lote", "icon": ft.Icons.INVENTORY_2},
]


class PipelineStepContext(ft.Column):
    """Card with 5 cascading searchable dropdowns + Continuar button."""

    def __init__(
        self,
        on_context_selected: Callable[[int], None],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._on_context_selected = on_context_selected

        # State: selected IDs per level
        self._selected: dict[str, int | None] = {lvl["key"]: None for lvl in _LEVELS}

        self._dropdowns: dict[str, ft.Dropdown] = {}
        self._build_controls()

    # ────────────────────────────────────────────────────────
    # Build UI
    # ────────────────────────────────────────────────────────

    def _build_controls(self) -> None:
        rows: list[ft.Control] = []

        rows.append(
            ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.AGRICULTURE, size=22, color=ft.Colors.PRIMARY),
                        ft.Text(
                            "Contexto de Producción",
                            size=18,
                            weight=ft.FontWeight.W_600,
                            color=ft.Colors.ON_SURFACE,
                        ),
                    ],
                    spacing=10,
                ),
                padding=ft.Padding(left=0, top=0, right=0, bottom=8),
            )
        )

        rows.append(
            ft.Text(
                "Seleccione la unidad de producción para esta evaluación.",
                size=13,
                color=ft.Colors.ON_SURFACE_VARIANT,
            )
        )

        for i, lvl in enumerate(_LEVELS):
            key = lvl["key"]

            dropdown = ft.Dropdown(
                label=lvl["label"],
                leading_icon=lvl["icon"],
                options=[],
                editable=True,
                enable_filter=True,
                enable_search=True,
                on_select=lambda e, k=key: self._on_dropdown_select(k, e),
                disabled=(i > 0),  # Only campaign enabled initially
                text_size=14,
                border_radius=10,
                content_padding=ft.Padding(left=12, top=14, right=12, bottom=14),
                filled=True,
                fill_color=ft.Colors.SURFACE_CONTAINER_LOW,
                expand=True,
            )

            self._dropdowns[key] = dropdown

            rows.append(
                ft.Container(
                    content=dropdown,
                    padding=ft.Padding(left=0, top=10, right=0, bottom=0),
                )
            )

        self._continue_btn = ft.ElevatedButton(
            "Continuar",
            icon=ft.Icons.ARROW_FORWARD,
            on_click=self._on_continue,
            disabled=True,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.PRIMARY,
                color=ft.Colors.ON_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            height=44,
        )

        rows.append(
            ft.Container(
                content=ft.Row(
                    [self._continue_btn],
                    alignment=ft.MainAxisAlignment.END,
                ),
                padding=ft.Padding(left=0, top=20, right=0, bottom=0),
            )
        )

        self.controls = [
            ft.Container(
                content=ft.Column(rows, spacing=4),
                padding=ft.Padding(left=28, top=24, right=28, bottom=24),
                border_radius=12,
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
                border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
                expand=True,
            )
        ]
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.expand = True

        # Pre-load campaign (top level only)
        self._load_options("campaign")

    # ────────────────────────────────────────────────────────
    # Data loading  (cascading via production_units)
    # ────────────────────────────────────────────────────────

    def _load_options(self, key: str) -> None:
        """Fetch options for *key* using parent selections."""
        svc = ProductionUnitService
        sel = self._selected

        try:
            if key == "campaign":
                items = svc.get_campaigns()
            elif key == "fundo":
                items = svc.get_fundos(sel["campaign"])  # type: ignore[arg-type]
            elif key == "module":
                items = svc.get_modules(sel["campaign"], sel["fundo"])  # type: ignore[arg-type]
            elif key == "shift":
                items = svc.get_shifts(sel["campaign"], sel["fundo"], sel["module"])  # type: ignore[arg-type]
            elif key == "batch":
                items = svc.get_batches(
                    sel["campaign"], sel["fundo"], sel["module"], sel["shift"]
                )  # type: ignore[arg-type]
            else:
                items = []
        except Exception:
            logger.exception("Error loading options for %s", key)
            items = []

        logger.info("Loaded %d options for %s (parents=%s)", len(items), key, sel)

        dd = self._dropdowns[key]
        dd.options = [
            ft.dropdown.Option(key=str(item["id"]), text=item["name"]) for item in items
        ]

    # ────────────────────────────────────────────────────────
    # Event handlers
    # ────────────────────────────────────────────────────────

    def _on_dropdown_select(self, key: str, e) -> None:
        """User selected a value — store it, reload children, reset downstream."""
        dd = self._dropdowns[key]
        value = dd.value

        if not value and hasattr(e, "data") and e.data:
            value = e.data
            dd.value = value

        logger.info("Dropdown select: key=%s, value=%s", key, value)

        if not value:
            return

        level_idx = [lvl["key"] for lvl in _LEVELS].index(key)

        # Store selection
        self._selected[key] = int(value)

        # Reset all downstream levels
        for child_lvl in _LEVELS[level_idx + 1 :]:
            ck = child_lvl["key"]
            self._selected[ck] = None
            self._dropdowns[ck].value = None
            self._dropdowns[ck].options = []
            self._dropdowns[ck].disabled = True

        # Enable & populate next level
        if level_idx + 1 < len(_LEVELS):
            next_key = _LEVELS[level_idx + 1]["key"]
            self._dropdowns[next_key].disabled = False
            self._load_options(next_key)

        # Enable Continue only when all five are selected
        all_selected = all(v is not None for v in self._selected.values())
        self._continue_btn.disabled = not all_selected

        try:
            self.page.update()  # type: ignore[union-attr]
        except Exception:
            pass

    def _on_continue(self, _) -> None:
        """All five selected — resolve (find-or-create) production_unit_id."""
        sel = self._selected
        pu_id = ProductionUnitService.resolve_production_unit_id(
            campaign_id=sel["campaign"],  # type: ignore[arg-type]
            unit_id=sel["fundo"],  # type: ignore[arg-type]
            module_id=sel["module"],  # type: ignore[arg-type]
            shift_id=sel["shift"],  # type: ignore[arg-type]
            batch_id=sel["batch"],  # type: ignore[arg-type]
        )
        if pu_id is None:
            logger.warning("No production_unit found for selection: %s", sel)
            return
        self._on_context_selected(pu_id)
