"""Sync controls — catalog sync button + analysis auto-sync."""

from __future__ import annotations

import logging
import threading

import flet as ft

from src.services.sync_service import SyncService

logger = logging.getLogger(__name__)


class CatalogSyncRow(ft.Container):
    """Row with 'Sincronizar Catalogos' button + spinner + status."""

    def __init__(self, sync_service: SyncService, page: ft.Page, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sync_service = sync_service
        self._page = page

        self._status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE)
        self._spinner = ft.ProgressRing(
            width=16, height=16, visible=False, color=ft.Colors.PRIMARY,
        )
        self._btn = ft.ElevatedButton(
            "Sincronizar Catalogos",
            icon=ft.Icons.CLOUD_DOWNLOAD,
            on_click=self._handle_sync,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.PRIMARY,
                color=ft.Colors.ON_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            height=38,
        )

        self.content = ft.Row(
            [self._btn, self._spinner, self._status],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.padding = ft.Padding(left=18, top=12, right=18, bottom=12)
        self.border_radius = 12
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_LOWEST
        self.border = ft.Border.all(1, ft.Colors.OUTLINE_VARIANT)

    def _handle_sync(self, _e) -> None:
        self._btn.disabled = True
        self._spinner.visible = True
        self._status.value = "Sincronizando..."
        self._status.color = ft.Colors.ON_SURFACE_VARIANT
        self._page.update()

        def _do_sync():
            result = self._sync_service.sync_catalogs()
            self._page.run_thread(lambda: self._on_done(result))

        threading.Thread(target=_do_sync, daemon=True).start()

    def _on_done(self, result) -> None:
        self._btn.disabled = False
        self._spinner.visible = False
        if result.success:
            self._status.value = result.detail
            self._status.color = ft.Colors.PRIMARY
        else:
            errors_summary = "; ".join(result.errors) if result.errors else result.detail
            self._status.value = f"Error: {errors_summary}"
            self._status.color = ft.Colors.ERROR
        self._page.update()


class AnalysisSyncStatus(ft.Container):
    """Analysis sync status + pending count display."""

    def __init__(self, sync_service: SyncService, page: ft.Page, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sync_service = sync_service
        self._page = page

        self._spinner = ft.ProgressRing(
            width=16, height=16, visible=False, color=ft.Colors.PRIMARY,
        )
        self._status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE)
        self._pending_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)

        self.content = ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED, size=18, color=ft.Colors.PRIMARY),
                        ft.Text(
                            "Sincronizacion de analisis", size=14,
                            weight=ft.FontWeight.W_600, color=ft.Colors.ON_SURFACE,
                        ),
                    ],
                    spacing=8,
                ),
                ft.Row(
                    [self._spinner, self._status],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self._pending_text,
            ],
            spacing=6,
        )
        self.padding = ft.Padding(left=14, top=12, right=14, bottom=12)
        self.border_radius = 10
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_LOW

    def auto_sync(self) -> None:
        """Trigger automatic sync after processing completes."""
        try:
            pending = SyncService.get_pending_analysis_count()
        except Exception:
            pending = 0

        if pending == 0:
            self._status.value = "Sin datos pendientes de sincronizacion."
            self._status.color = ft.Colors.ON_SURFACE_VARIANT
            self._refresh_counts()
            self._page.update()
            return

        self._spinner.visible = True
        self._status.value = f"Sincronizando {pending} registro(s)..."
        self._status.color = ft.Colors.ON_SURFACE_VARIANT
        self._page.update()

        def _do_sync():
            result = self._sync_service.sync_analysis_records()
            self._page.run_thread(lambda: self._on_done(result))

        threading.Thread(target=_do_sync, daemon=True).start()

    def _on_done(self, result) -> None:
        self._spinner.visible = False
        if result.success:
            self._status.value = f"Sincronizacion completada — {result.detail}"
            self._status.color = ft.Colors.PRIMARY
        else:
            self._status.value = f"Pendiente: {result.detail}"
            self._status.color = ft.Colors.TERTIARY
        self._refresh_counts()
        self._page.update()

    def _refresh_counts(self) -> None:
        try:
            pending = SyncService.get_pending_analysis_count()
        except Exception:
            pending = 0
        if pending > 0:
            self._pending_text.value = f"{pending} registro(s) pendientes de sincronizacion"
        else:
            self._pending_text.value = "No hay registros pendientes"
