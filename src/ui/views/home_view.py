"""Home view — main screen after login with sync controls and 4-step processing pipeline."""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

import flet as ft
import numpy as np

from src.config import (
    CLS_MODEL_PATH,
    DEFAULT_CONFIDENCE,
    DET_MODEL_PATH,
    DISPLAY_MAX_H,
    DISPLAY_MAX_W,
    TRACKER_CONFIG,
)
from src.model.auth import CachedUser
from src.schemas import ProcessingResult, VideoInfo
from src.services.sync_service import SyncService
from src.services.video_loader import VideoLoader
from src.ui.components.roi_canvas import ROICanvas
from src.ui.components.step_indicator import StepIndicator
from src.ui.theme import LIGHT
from src.ui.views.results_view import ResultsView
from src.utils.image_utils import frame_to_base64, resize_frame_for_display

logger = logging.getLogger(__name__)


class HomeView(ft.Column):
    """Main application view with catalog sync + 4-step video processing pipeline.

    Steps:
        0 — Cargar Video (file picker)
        1 — Zona de Interes (ROI canvas)
        2 — Procesamiento (progress bar + live counters)
        3 — Resultados (summary + crop gallery + auto analysis sync)
    """

    def __init__(
        self,
        user: CachedUser,
        page: ft.Page,
        on_logout: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._user = user
        self._page = page
        self._on_logout = on_logout
        self._sync_service = SyncService(access_token=user.access_token)

        self.spacing = 16
        self.expand = True

        # ── State ────────────────────────────────────────────
        self._video_info: VideoInfo | None = None
        self._result: ProcessingResult | None = None
        self._current_step = 0

        # ── Catalog sync controls ────────────────────────────
        self._catalog_status = ft.Text("", size=12)
        self._catalog_spinner = ft.ProgressRing(width=16, height=16, visible=False)
        self._sync_catalog_btn = ft.ElevatedButton(
            "Sincronizar",
            icon=ft.Icons.CLOUD_DOWNLOAD,
            on_click=self._handle_sync_catalogs,
            style=ft.ButtonStyle(
                bgcolor=LIGHT["primary"],
                color=LIGHT["primary_foreground"],
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            height=36,
        )
        self._counts_row = ft.Row(spacing=8, wrap=True)

        # ── Step indicator ───────────────────────────────────
        self._step_indicator = StepIndicator(palette=LIGHT, current=0)

        # ── File picker (Flet 0.80+ Service) ─────────────────
        self._file_picker = ft.FilePicker()
        page.services.append(self._file_picker)

        # ── Step 0: Load video ───────────────────────────────
        self._video_label = ft.Text(
            "Ningun video seleccionado",
            size=13,
            color=LIGHT["muted_foreground"],
            italic=True,
        )
        self._load_spinner = ft.ProgressRing(width=16, height=16, visible=False)
        self._load_status = ft.Text("", size=12)

        # ── GPU toggle ─────────────────────────────────────────
        self._use_gpu = False
        self._gpu_switch = ft.Switch(
            label="Usar GPU",
            value=False,
            on_change=self._handle_gpu_toggle,
            active_color=LIGHT["primary"],
        )
        self._gpu_status = ft.Text("", size=11, color=LIGHT["muted_foreground"])
        self._detect_gpu_availability()

        # ── Step 2: Processing ───────────────────────────────
        self._process_progress = ft.ProgressBar(
            visible=False,
            color=LIGHT["primary"],
            bgcolor=LIGHT["muted"],
        )
        self._process_status = ft.Text("", size=13)
        self._process_counters = ft.Text("", size=13, weight=ft.FontWeight.W_600)

        # ── Step 3: Analysis sync status ─────────────────────
        self._analysis_spinner = ft.ProgressRing(width=16, height=16, visible=False)
        self._analysis_status = ft.Text("", size=12)

        # ── Pending analysis indicator ───────────────────────
        self._pending_text = ft.Text("", size=12, color=LIGHT["muted_foreground"])

        # ── Dynamic content area (changes per step) ──────────
        self._step_content = ft.Column(spacing=10, expand=True)

        # ── Build layout ─────────────────────────────────────
        self.controls = [
            self._build_user_bar(),
            self._build_catalog_section(),
            self._build_pipeline_section(),
        ]

        self._refresh_counts()
        self._show_step(0)

    # ================================================================
    # Layout builders
    # ================================================================

    def _build_user_bar(self) -> ft.Container:
        initials = self._user.username[0].upper() if self._user.username else "?"
        return ft.Container(
            content=ft.Row(
                [
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Text(
                                    initials,
                                    size=16,
                                    weight=ft.FontWeight.BOLD,
                                    color=LIGHT["primary_foreground"],
                                    text_align=ft.TextAlign.CENTER,
                                ),
                                width=38,
                                height=38,
                                border_radius=19,
                                bgcolor=LIGHT["primary"],
                                alignment=ft.Alignment.CENTER,
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        self._user.username,
                                        size=15,
                                        weight=ft.FontWeight.W_600,
                                    ),
                                    ft.Text(
                                        self._user.role_name or "Sin rol",
                                        size=11,
                                        color=LIGHT["muted_foreground"],
                                    ),
                                ],
                                spacing=0,
                            ),
                        ],
                        spacing=12,
                    ),
                    ft.OutlinedButton(
                        "Cerrar Sesion",
                        icon=ft.Icons.LOGOUT,
                        on_click=self._handle_logout,
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=8),
                        ),
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding(left=20, top=12, right=20, bottom=12),
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
            border=ft.Border.all(1, LIGHT["border"]),
        )

    def _build_catalog_section(self) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Row(
                                [
                                    ft.Icon(ft.Icons.INVENTORY_2_OUTLINED, size=20, color=LIGHT["primary"]),
                                    ft.Text("Catalogos", size=16, weight=ft.FontWeight.W_600),
                                ],
                                spacing=8,
                            ),
                            ft.Row(
                                [
                                    self._catalog_spinner,
                                    self._catalog_status,
                                    self._sync_catalog_btn,
                                ],
                                spacing=10,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Text(
                        "Data maestra: unidades agricolas, campanas, modulos, turnos y lotes.",
                        size=12,
                        color=LIGHT["muted_foreground"],
                    ),
                    self._counts_row,
                ],
                spacing=10,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
            border=ft.Border.all(1, LIGHT["border"]),
        )

    def _build_pipeline_section(self) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Row(
                                [
                                    ft.Icon(ft.Icons.VIDEOCAM_OUTLINED, size=22, color=LIGHT["primary"]),
                                    ft.Text("Procesamiento de Video", size=16, weight=ft.FontWeight.W_600),
                                ],
                                spacing=8,
                            ),
                            ft.Row(
                                [
                                    self._gpu_switch,
                                    self._gpu_status,
                                ],
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self._step_indicator,
                    ft.Divider(height=1, color=LIGHT["border"]),
                    self._step_content,
                ],
                spacing=12,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
            border=ft.Border.all(1, LIGHT["border"]),
            expand=True,
        )

    # ================================================================
    # Step content builders
    # ================================================================

    def _show_step(self, step: int) -> None:
        """Switch the visible step content."""
        self._current_step = step
        try:
            self._step_indicator.set_step(step)
        except Exception:
            pass

        self._step_content.controls.clear()

        if step == 0:
            self._build_step_0()
        elif step == 1:
            self._build_step_1()
        elif step == 2:
            self._build_step_2()
        elif step == 3:
            self._build_step_3()

        try:
            self._page.update()
        except Exception:
            pass

    def _build_step_0(self) -> None:
        """Step 0 — Select and load a video file."""
        self._video_label.value = "Ningun video seleccionado"
        self._video_label.italic = True
        self._video_label.color = LIGHT["muted_foreground"]
        self._load_status.value = ""

        select_btn = ft.OutlinedButton(
            "Seleccionar Video",
            icon=ft.Icons.VIDEO_FILE_OUTLINED,
            on_click=self._handle_pick_file,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
        )

        self._step_content.controls = [
            ft.Text(
                "Seleccione un archivo de video para iniciar el procesamiento.",
                size=13,
                color=LIGHT["muted_foreground"],
            ),
            ft.Row(
                [select_btn, self._video_label],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Row(
                [self._load_spinner, self._load_status],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            # Video info (populated after loading)
        ]

    def _build_step_1(self) -> None:
        """Step 1 — Draw ROI on first frame."""
        if not self._video_info or self._video_info.first_frame is None:
            self._step_content.controls = [
                ft.Text("Error: no se pudo cargar el video.", color=LIGHT["destructive"]),
            ]
            return

        frame = self._video_info.first_frame
        resized, scale_x, scale_y = resize_frame_for_display(
            frame, DISPLAY_MAX_W, DISPLAY_MAX_H,
        )
        display_h, display_w = resized.shape[:2]
        image_b64 = frame_to_base64(resized)

        roi_canvas = ROICanvas(
            image_base64=image_b64,
            display_width=display_w,
            display_height=display_h,
            scale_x=scale_x,
            scale_y=scale_y,
            on_roi_confirmed=self._on_roi_confirmed,
            palette=LIGHT,
        )

        back_btn = ft.TextButton(
            "Volver",
            icon=ft.Icons.ARROW_BACK,
            on_click=lambda _: self._show_step(0),
        )

        info = self._video_info
        info_text = ft.Text(
            f"{os.path.basename(info.path)}  —  "
            f"{info.width}x{info.height}  |  {info.fps} FPS  |  "
            f"{info.total_frames} frames"
            f"{('  |  HDR: ' + info.hdr_transfer.upper()) if info.hdr_transfer else ''}",
            size=12,
            color=LIGHT["muted_foreground"],
        )

        self._step_content.controls = [
            ft.Row([back_btn, info_text], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            roi_canvas,
        ]

    def _build_step_2(self) -> None:
        """Step 2 — Processing in progress."""
        self._process_progress.visible = True
        self._process_progress.value = 0
        self._process_status.value = "Iniciando procesamiento..."
        self._process_status.color = LIGHT["muted_foreground"]
        self._process_counters.value = ""

        self._step_content.controls = [
            ft.Text(
                "Procesando video — deteccion, tracking y clasificacion...",
                size=13,
                color=LIGHT["muted_foreground"],
            ),
            self._process_progress,
            self._process_status,
            self._process_counters,
        ]

    def _build_step_3(self) -> None:
        """Step 3 — Results + automatic analysis sync."""
        if not self._result:
            self._step_content.controls = [
                ft.Text("No hay resultados disponibles.", color=LIGHT["muted_foreground"]),
            ]
            return

        results_view = ResultsView(
            result=self._result,
            crop_images=self._result.crop_images,
            palette=LIGHT,
            page=self._page,
            on_reset=lambda _: self._reset_pipeline(),
        )

        sync_row = ft.Row(
            [
                self._analysis_spinner,
                self._analysis_status,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._step_content.controls = [
            results_view,
            ft.Divider(height=1, color=LIGHT["border"]),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED, size=18, color=LIGHT["primary"]),
                                ft.Text(
                                    "Sincronizacion de datos de analisis",
                                    size=14,
                                    weight=ft.FontWeight.W_600,
                                ),
                            ],
                            spacing=8,
                        ),
                        sync_row,
                        self._pending_text,
                    ],
                    spacing=6,
                ),
                padding=ft.Padding(left=12, top=10, right=12, bottom=10),
                border_radius=8,
                bgcolor=LIGHT["secondary"],
            ),
        ]

    # ================================================================
    # GPU detection
    # ================================================================

    def _detect_gpu_availability(self) -> None:
        """Check if CUDA is available via onnxruntime."""
        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            cuda_available = "CUDAExecutionProvider" in available_providers
            if cuda_available:
                self._gpu_status.value = "CUDA disponible"
                self._gpu_status.color = LIGHT["primary"]
                self._gpu_switch.value = True
                self._use_gpu = True
            else:
                self._gpu_status.value = "Solo CPU"
                self._gpu_status.color = LIGHT["muted_foreground"]
                self._gpu_switch.value = False
                self._use_gpu = False
        except ImportError:
            self._gpu_status.value = "onnxruntime no instalado"
            self._gpu_status.color = LIGHT["destructive"]
            self._gpu_switch.value = False
            self._gpu_switch.disabled = True
            self._use_gpu = False

    def _handle_gpu_toggle(self, e) -> None:
        """Toggle GPU usage."""
        self._use_gpu = e.control.value
        self._gpu_status.value = "GPU activada" if self._use_gpu else "Solo CPU"
        self._gpu_status.color = LIGHT["primary"] if self._use_gpu else LIGHT["muted_foreground"]
        self._page.update()

    # ================================================================
    # Handlers
    # ================================================================

    # ── Step 0: File pick + video load ───────────────────────

    async def _handle_pick_file(self, _e) -> None:
        """Open native file picker and load the selected video."""
        files = await self._file_picker.pick_files(
            allowed_extensions=["mp4", "avi", "mov", "mkv"],
            file_type=ft.FilePickerFileType.CUSTOM,
            dialog_title="Seleccionar video",
        )
        if not files or len(files) == 0:
            return

        video_path = files[0].path
        if not video_path or not os.path.exists(video_path):
            self._load_status.value = "El archivo seleccionado no existe."
            self._load_status.color = LIGHT["destructive"]
            self._page.update()
            return

        name = os.path.basename(video_path)
        self._video_label.value = name
        self._video_label.italic = False
        self._video_label.color = LIGHT["foreground"]
        self._load_spinner.visible = True
        self._load_status.value = "Cargando video..."
        self._load_status.color = LIGHT["muted_foreground"]
        self._page.update()

        def _do_load():
            try:
                info = VideoLoader.load(video_path)
                self._page.run_thread(lambda: self._on_video_loaded(info))
            except Exception as exc:
                error_msg = str(exc)
                self._page.run_thread(
                    lambda: self._on_video_load_error(error_msg)
                )

        threading.Thread(target=_do_load, daemon=True).start()

    def _on_video_loaded(self, info: VideoInfo) -> None:
        self._video_info = info
        self._load_spinner.visible = False
        self._load_status.value = (
            f"{info.width}x{info.height} | {info.fps} FPS | {info.total_frames} frames"
        )
        self._load_status.color = LIGHT["primary"]
        self._page.update()

        # Auto-advance to step 1 (ROI)
        self._show_step(1)

    def _on_video_load_error(self, error: str) -> None:
        self._load_spinner.visible = False
        self._load_status.value = f"Error: {error}"
        self._load_status.color = LIGHT["destructive"]
        self._page.update()

    # ── Step 1: ROI confirmed ────────────────────────────────

    def _on_roi_confirmed(self, roi_points: list[tuple[int, int]]) -> None:
        """Called when user confirms ROI — starts processing."""
        self._roi_points = roi_points
        self._show_step(2)
        self._start_processing()

    # ── Step 2: Processing ───────────────────────────────────

    def _start_processing(self) -> None:
        """Launch video processing in a background thread."""
        info = self._video_info
        if not info:
            return

        use_gpu = self._use_gpu

        def _do_process():
            try:
                from src.processing.video_processor import VideoProcessor

                processor = VideoProcessor(
                    video_path=info.path,
                    model_path=DET_MODEL_PATH,
                    cls_model_path=CLS_MODEL_PATH,
                    tracker_config=TRACKER_CONFIG,
                    roi_points=self._roi_points,
                    use_gpu=use_gpu,
                    confidence=DEFAULT_CONFIDENCE,
                    hdr_transfer=info.hdr_transfer,
                )
                result = processor.process(
                    progress_callback=self._on_process_progress
                )
                self._page.run_thread(lambda: self._on_processing_done(result))
            except Exception as exc:
                logger.exception("Processing error")
                error_msg = str(exc)
                self._page.run_thread(
                    lambda: self._on_processing_error(error_msg)
                )

        threading.Thread(target=_do_process, daemon=True).start()

    def _on_process_progress(
        self, frame: int, total: int, in_count: int, out_count: int
    ) -> None:
        """Called from processing thread every N frames."""
        try:
            pct = frame / total if total > 0 else 0
            self._process_progress.value = pct
            self._process_status.value = f"Frame {frame}/{total} ({pct:.0%})"
            self._process_counters.value = f"IN: {in_count}   OUT: {out_count}"
            self._page.update()
        except Exception:
            pass

    def _on_processing_done(self, result: ProcessingResult) -> None:
        self._result = result
        self._process_progress.visible = False
        self._show_step(3)

        # Automatic analysis sync after processing
        self._auto_sync_analysis()

    def _on_processing_error(self, error: str) -> None:
        self._process_progress.visible = False
        self._process_status.value = f"Error: {error}"
        self._process_status.color = LIGHT["destructive"]
        self._page.update()

    # ── Step 3: Auto analysis sync ───────────────────────────

    def _auto_sync_analysis(self) -> None:
        """Automatically sync analysis records after processing."""
        try:
            pending = SyncService.get_pending_analysis_count()
        except Exception:
            pending = 0

        if pending == 0:
            self._analysis_status.value = "Sin datos pendientes de sincronizacion."
            self._analysis_status.color = LIGHT["muted_foreground"]
            self._refresh_counts()
            self._page.update()
            return

        self._analysis_spinner.visible = True
        self._analysis_status.value = f"Sincronizando {pending} registro(s)..."
        self._analysis_status.color = LIGHT["muted_foreground"]
        self._page.update()

        def _do_sync():
            result = self._sync_service.sync_analysis_records()
            self._page.run_thread(lambda: self._on_analysis_sync_done(result))

        threading.Thread(target=_do_sync, daemon=True).start()

    def _on_analysis_sync_done(self, result) -> None:
        self._analysis_spinner.visible = False

        if result.success:
            self._analysis_status.value = f"Sincronizacion completada — {result.detail}"
            self._analysis_status.color = LIGHT["primary"]
        else:
            self._analysis_status.value = f"Pendiente: {result.detail}"
            self._analysis_status.color = LIGHT["accent"]

        self._refresh_counts()
        self._page.update()

    # ── Pipeline reset ───────────────────────────────────────

    def _reset_pipeline(self) -> None:
        """Reset back to step 0 for a new video."""
        self._video_info = None
        self._result = None
        self._show_step(0)

    # ── Catalog sync ─────────────────────────────────────────

    def _handle_sync_catalogs(self, _e) -> None:
        self._sync_catalog_btn.disabled = True
        self._catalog_spinner.visible = True
        self._catalog_status.value = "Sincronizando..."
        self._catalog_status.color = LIGHT["muted_foreground"]
        self._page.update()

        def _do_sync():
            result = self._sync_service.sync_catalogs()
            self._page.run_thread(lambda: self._on_catalog_sync_done(result))

        threading.Thread(target=_do_sync, daemon=True).start()

    def _on_catalog_sync_done(self, result) -> None:
        self._sync_catalog_btn.disabled = False
        self._catalog_spinner.visible = False

        if result.success:
            self._catalog_status.value = result.detail
            self._catalog_status.color = LIGHT["primary"]
        else:
            errors_summary = "; ".join(result.errors) if result.errors else result.detail
            self._catalog_status.value = f"Error: {errors_summary}"
            self._catalog_status.color = LIGHT["destructive"]

        self._refresh_counts()
        self._page.update()

    # ── Logout ───────────────────────────────────────────────

    def _handle_logout(self, _e) -> None:
        from src.services.auth_service import AuthService

        AuthService().logout()
        self._on_logout()

    # ================================================================
    # Helpers
    # ================================================================

    def _refresh_counts(self) -> None:
        """Update catalog counts and pending analysis indicator."""
        try:
            counts = SyncService.get_local_counts()
            pending = SyncService.get_pending_analysis_count()
        except Exception:
            counts = {}
            pending = 0

        label_map = {
            "agricultural_units": "Unid. Agricolas",
            "agricultural_campaigns": "Campanas",
            "modules": "Modulos",
            "shifts": "Turnos",
            "batches": "Lotes",
        }

        self._counts_row.controls = [
            self._count_chip(label, counts.get(key, 0))
            for key, label in label_map.items()
        ]

        if pending > 0:
            self._pending_text.value = f"{pending} registro(s) pendientes de sincronizacion"
        else:
            self._pending_text.value = "No hay registros pendientes"

    @staticmethod
    def _count_chip(label: str, count: int) -> ft.Container:
        """Compact pill showing catalog name and row count."""
        has_data = count > 0
        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(
                        ft.Icons.CHECK_CIRCLE if has_data else ft.Icons.CIRCLE_OUTLINED,
                        size=14,
                        color=LIGHT["primary"] if has_data else LIGHT["muted_foreground"],
                    ),
                    ft.Text(
                        f"{label}: {count}",
                        size=11,
                        color=LIGHT["foreground"] if has_data else LIGHT["muted_foreground"],
                    ),
                ],
                spacing=4,
            ),
            padding=ft.Padding(left=8, top=4, right=10, bottom=4),
            border_radius=16,
            bgcolor=LIGHT["secondary"] if has_data else ft.Colors.TRANSPARENT,
            border=ft.Border.all(1, LIGHT["primary"] if has_data else LIGHT["border"]),
        )
