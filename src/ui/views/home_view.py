"""Home view — sidebar navigation + 4-step video processing pipeline."""

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

# ── Sidebar palette (always green, theme-independent) ─────
_SB_BG = LIGHT["sidebar_bg"]          # #4B8C6B
_SB_FG = LIGHT["sidebar_fg"]          # #FFFFFF
_SB_ACCENT = LIGHT["sidebar_accent"]  # #3E9C6B
_SB_BORDER = LIGHT["sidebar_border"]  # #3A5C4A
_SB_MUTED = "#FFFFFFB3"
_SB_SUBTLE = "#FFFFFF1A"

# ── Content area uses ft.Colors.* tokens (auto light/dark) ──
# ft.Colors.PRIMARY, ON_PRIMARY, SURFACE, ON_SURFACE,
# ON_SURFACE_VARIANT, SURFACE_CONTAINER_LOWEST, OUTLINE_VARIANT,
# ERROR, TERTIARY — all resolve per theme automatically.

_SIDEBAR_W = 220


class HomeView(ft.Row):
    """Main view: green sidebar + content area with navigation."""

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

        self.spacing = 0
        self.expand = True
        self.vertical_alignment = ft.CrossAxisAlignment.STRETCH

        # ── State ────────────────────────────────────────────
        self._video_info: VideoInfo | None = None
        self._result: ProcessingResult | None = None
        self._current_step = 0
        self._active_nav = 0

        # ── Catalog sync controls ────────────────────────────
        self._catalog_status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE)
        self._catalog_spinner = ft.ProgressRing(
            width=16, height=16, visible=False, color=ft.Colors.PRIMARY,
        )
        self._sync_catalog_btn = ft.ElevatedButton(
            "Sincronizar Catalogos",
            icon=ft.Icons.CLOUD_DOWNLOAD,
            on_click=self._handle_sync_catalogs,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.PRIMARY,
                color=ft.Colors.ON_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            height=38,
        )

        # ── Step indicator ───────────────────────────────────
        self._step_indicator = StepIndicator(palette=LIGHT, current=0)

        # ── File picker (Flet 0.80+ Service) ─────────────────
        self._file_picker = ft.FilePicker()
        page.services.append(self._file_picker)

        # ── Step 0: Load video ───────────────────────────────
        self._video_label = ft.Text(
            "Ningun video seleccionado",
            size=13,
            color=ft.Colors.ON_SURFACE_VARIANT,
            italic=True,
        )
        self._load_spinner = ft.ProgressRing(
            width=16, height=16, visible=False, color=ft.Colors.PRIMARY,
        )
        self._load_status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE)

        # ── GPU toggle ─────────────────────────────────────────
        self._use_gpu = False
        self._gpu_switch = ft.Switch(
            label="GPU",
            value=False,
            on_change=self._handle_gpu_toggle,
            active_color=ft.Colors.PRIMARY,
        )
        self._gpu_status = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self._detect_gpu_availability()

        # ── Step 2: Processing ───────────────────────────────
        self._process_progress = ft.ProgressBar(
            visible=False,
            color=ft.Colors.PRIMARY,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
        )
        self._process_status = ft.Text("", size=13, color=ft.Colors.ON_SURFACE)
        self._process_counters = ft.Text(
            "", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.ON_SURFACE,
        )

        # ── Step 3: Analysis sync status ─────────────────────
        self._analysis_spinner = ft.ProgressRing(
            width=16, height=16, visible=False, color=ft.Colors.PRIMARY,
        )
        self._analysis_status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE)

        # ── Pending analysis indicator ───────────────────────
        self._pending_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)

        # ── Dynamic content area (changes per step) ──────────
        self._step_content = ft.Column(spacing=12, expand=True)

        # ── Page content area ──────────────────────────────────
        self._main_content = ft.Column(
            spacing=20,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )

        # ── Build layout ─────────────────────────────────────
        self.controls = [
            self._build_sidebar(),
            ft.Container(
                content=self._main_content,
                expand=True,
                padding=ft.Padding(left=28, top=24, right=28, bottom=24),
            ),
        ]

        self._show_nav(0)
        self._show_step(0)

    # ================================================================
    # Sidebar (green, theme-independent)
    # ================================================================

    def _build_sidebar(self) -> ft.Container:
        initials = self._user.username[0].upper() if self._user.username else "?"

        avatar = ft.Container(
            content=ft.Text(
                initials, size=22, weight=ft.FontWeight.BOLD,
                color=_SB_BG, text_align=ft.TextAlign.CENTER,
            ),
            width=50, height=50, border_radius=25,
            bgcolor=_SB_FG, alignment=ft.Alignment.CENTER,
        )

        user_section = ft.Column(
            [
                avatar,
                ft.Text(
                    self._user.username, size=14,
                    weight=ft.FontWeight.W_600, color=_SB_FG,
                    text_align=ft.TextAlign.CENTER,
                    max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.Container(
                    content=ft.Text(
                        self._user.role_name or "Sin rol",
                        size=11, color=_SB_MUTED,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    padding=ft.Padding(left=6, top=2, right=6, bottom=2),
                    border_radius=10, bgcolor=_SB_SUBTLE,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
        )

        self._nav_eval = self._sidebar_item(
            ft.Icons.VIDEOCAM_OUTLINED, "Evaluacion", active=True,
            on_click=lambda _: self._show_nav(0),
        )
        self._nav_profile = self._sidebar_item(
            ft.Icons.PERSON_OUTLINED, "Perfil", active=False,
            on_click=lambda _: self._show_nav(1),
        )

        logout_item = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.LOGOUT, size=17, color="#FFFFFF99"),
                    ft.Text("Cerrar Sesion", size=13, color="#FFFFFF99"),
                ],
                spacing=10,
            ),
            padding=ft.Padding(left=14, top=10, right=14, bottom=10),
            border_radius=8,
            on_click=self._handle_logout,
            on_hover=lambda e: _hover_sidebar(e, logout_item),
            ink=True,
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=user_section,
                        padding=ft.Padding(left=0, top=24, right=0, bottom=16),
                    ),
                    ft.Container(height=1, bgcolor=_SB_BORDER),
                    ft.Container(
                        content=ft.Column([self._nav_eval, self._nav_profile], spacing=4),
                        padding=ft.Padding(left=0, top=12, right=0, bottom=0),
                    ),
                    ft.Container(expand=True),
                    ft.Container(height=1, bgcolor=_SB_BORDER),
                    ft.Container(
                        content=logout_item,
                        padding=ft.Padding(left=0, top=8, right=0, bottom=0),
                    ),
                ],
                spacing=0,
                expand=True,
            ),
            width=_SIDEBAR_W,
            padding=ft.Padding(left=12, top=0, right=12, bottom=16),
            bgcolor=_SB_BG,
            border_radius=ft.BorderRadius(
                top_left=0, top_right=16, bottom_left=0, bottom_right=16,
            ),
        )

    def _sidebar_item(self, icon, label, active, on_click):
        fg = _SB_FG if active else _SB_MUTED
        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(icon, size=18, color=fg),
                    ft.Text(
                        label, size=13, color=fg,
                        weight=ft.FontWeight.W_600 if active else ft.FontWeight.W_400,
                    ),
                ],
                spacing=10,
            ),
            padding=ft.Padding(left=14, top=10, right=14, bottom=10),
            border_radius=8,
            bgcolor=_SB_ACCENT if active else ft.Colors.TRANSPARENT,
            on_click=on_click,
            ink=True,
        )

    # ================================================================
    # Navigation switching
    # ================================================================

    def _show_nav(self, index: int) -> None:
        self._active_nav = index
        _update_nav(self._nav_eval, active=(index == 0))
        _update_nav(self._nav_profile, active=(index == 1))

        self._main_content.controls.clear()
        if index == 0:
            self._build_evaluation_page()
        else:
            self._build_profile_page()

        try:
            self._page.update()
        except Exception:
            pass

    # ================================================================
    # Page builders (theme-responsive colors)
    # ================================================================

    def _build_evaluation_page(self) -> None:
        sync_row = ft.Container(
            content=ft.Row(
                [
                    self._sync_catalog_btn,
                    self._catalog_spinner,
                    self._catalog_status,
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding(left=18, top=12, right=18, bottom=12),
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

        self._main_content.controls = [
            ft.Text(
                "Evaluacion", size=22,
                weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE,
            ),
            sync_row,
            self._build_pipeline_section(),
        ]

    def _build_profile_page(self) -> None:
        initials = self._user.username[0].upper() if self._user.username else "?"

        info_rows = [
            ("Usuario", self._user.username),
            ("Rol", self._user.role_name or "Sin rol"),
        ]
        detail_controls = []
        for label, value in info_rows:
            detail_controls.append(
                ft.Row(
                    [
                        ft.Text(label, size=13, color=ft.Colors.ON_SURFACE_VARIANT, width=100),
                        ft.Text(
                            value, size=13,
                            weight=ft.FontWeight.W_600, color=ft.Colors.ON_SURFACE,
                        ),
                    ],
                    spacing=12,
                )
            )

        self._main_content.controls = [
            ft.Text(
                "Perfil", size=22,
                weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE,
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Container(
                                    content=ft.Text(
                                        initials, size=28,
                                        weight=ft.FontWeight.BOLD,
                                        color=ft.Colors.ON_PRIMARY,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                    width=64, height=64, border_radius=32,
                                    bgcolor=ft.Colors.PRIMARY,
                                    alignment=ft.Alignment.CENTER,
                                ),
                                ft.Column(
                                    [
                                        ft.Text(
                                            self._user.username, size=18,
                                            weight=ft.FontWeight.BOLD,
                                            color=ft.Colors.ON_SURFACE,
                                        ),
                                        ft.Text(
                                            self._user.role_name or "Sin rol",
                                            size=13,
                                            color=ft.Colors.ON_SURFACE_VARIANT,
                                        ),
                                    ],
                                    spacing=2,
                                ),
                            ],
                            spacing=16,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
                        ft.Column(detail_controls, spacing=10),
                    ],
                    spacing=16,
                ),
                padding=24,
                border_radius=12,
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
                border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            ),
        ]

    def _build_pipeline_section(self) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.ResponsiveRow(
                        [
                            ft.Container(
                                content=ft.Row(
                                    [
                                        ft.Icon(
                                            ft.Icons.VIDEOCAM_OUTLINED,
                                            size=20, color=ft.Colors.PRIMARY,
                                        ),
                                        ft.Text(
                                            "Procesamiento de Video", size=16,
                                            weight=ft.FontWeight.W_600,
                                            color=ft.Colors.ON_SURFACE,
                                        ),
                                    ],
                                    spacing=8,
                                ),
                                col={"xs": 12, "md": 7},
                            ),
                            ft.Container(
                                content=ft.Row(
                                    [self._gpu_switch, self._gpu_status],
                                    spacing=8,
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                    alignment=ft.MainAxisAlignment.END,
                                ),
                                col={"xs": 12, "md": 5},
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self._step_indicator,
                    ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
                    self._step_content,
                ],
                spacing=14,
            ),
            padding=22,
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            expand=True,
        )

    # ================================================================
    # Step content builders
    # ================================================================

    def _show_step(self, step: int) -> None:
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
        self._video_label.value = "Ningun video seleccionado"
        self._video_label.italic = True
        self._video_label.color = ft.Colors.ON_SURFACE_VARIANT
        self._load_status.value = ""

        select_btn = ft.ElevatedButton(
            "Seleccionar Video",
            icon=ft.Icons.VIDEO_FILE_OUTLINED,
            on_click=self._handle_pick_file,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                color=ft.Colors.ON_SURFACE,
            ),
            height=40,
        )

        self._step_content.controls = [
            ft.Text(
                "Seleccione un archivo de video para iniciar el procesamiento.",
                size=13, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.ResponsiveRow([
                ft.Container(
                    content=ft.Row(
                        [select_btn, self._video_label],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        wrap=True,
                    ),
                    col={"xs": 12},
                ),
            ]),
            ft.Row(
                [self._load_spinner, self._load_status],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ]

    def _build_step_1(self) -> None:
        if not self._video_info or self._video_info.first_frame is None:
            self._step_content.controls = [
                ft.Text("Error: no se pudo cargar el video.", color=ft.Colors.ERROR),
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
            style=ft.ButtonStyle(color=ft.Colors.PRIMARY),
        )

        info = self._video_info
        info_text = ft.Text(
            f"{os.path.basename(info.path)}  —  "
            f"{info.width}x{info.height}  |  {info.fps} FPS  |  "
            f"{info.total_frames} frames"
            f"{('  |  HDR: ' + info.hdr_transfer.upper()) if info.hdr_transfer else ''}",
            size=12, color=ft.Colors.ON_SURFACE_VARIANT,
        )

        self._step_content.controls = [
            ft.Row(
                [back_btn, info_text],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                wrap=True,
            ),
            roi_canvas,
        ]

    def _build_step_2(self) -> None:
        self._process_progress.visible = True
        self._process_progress.value = 0
        self._process_status.value = "Iniciando procesamiento..."
        self._process_status.color = ft.Colors.ON_SURFACE_VARIANT
        self._process_counters.value = ""

        self._step_content.controls = [
            ft.Text(
                "Procesando video — deteccion, tracking y clasificacion...",
                size=13, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self._process_progress,
            self._process_status,
            self._process_counters,
        ]

    def _build_step_3(self) -> None:
        if not self._result:
            self._step_content.controls = [
                ft.Text("No hay resultados disponibles.", color=ft.Colors.ON_SURFACE_VARIANT),
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
            [self._analysis_spinner, self._analysis_status],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._step_content.controls = [
            results_view,
            ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(
                                    ft.Icons.CLOUD_UPLOAD_OUTLINED,
                                    size=18, color=ft.Colors.PRIMARY,
                                ),
                                ft.Text(
                                    "Sincronizacion de analisis", size=14,
                                    weight=ft.FontWeight.W_600,
                                    color=ft.Colors.ON_SURFACE,
                                ),
                            ],
                            spacing=8,
                        ),
                        sync_row,
                        self._pending_text,
                    ],
                    spacing=6,
                ),
                padding=ft.Padding(left=14, top=12, right=14, bottom=12),
                border_radius=10,
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            ),
        ]

    # ================================================================
    # GPU detection
    # ================================================================

    def _detect_gpu_availability(self) -> None:
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                self._gpu_status.value = "CUDA disponible"
                self._gpu_status.color = ft.Colors.PRIMARY
                self._gpu_switch.value = True
                self._use_gpu = True
            else:
                self._gpu_status.value = "Solo CPU"
                self._gpu_status.color = ft.Colors.ON_SURFACE_VARIANT
                self._gpu_switch.value = False
                self._use_gpu = False
        except ImportError:
            self._gpu_status.value = "onnxruntime no instalado"
            self._gpu_status.color = ft.Colors.ERROR
            self._gpu_switch.value = False
            self._gpu_switch.disabled = True
            self._use_gpu = False

    def _handle_gpu_toggle(self, e) -> None:
        self._use_gpu = e.control.value
        self._gpu_status.value = "GPU activada" if self._use_gpu else "Solo CPU"
        self._gpu_status.color = ft.Colors.PRIMARY if self._use_gpu else ft.Colors.ON_SURFACE_VARIANT
        self._page.update()

    # ================================================================
    # Handlers
    # ================================================================

    async def _handle_pick_file(self, _e) -> None:
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
            self._load_status.color = ft.Colors.ERROR
            self._page.update()
            return

        name = os.path.basename(video_path)
        self._video_label.value = name
        self._video_label.italic = False
        self._video_label.color = ft.Colors.ON_SURFACE
        self._load_spinner.visible = True
        self._load_status.value = "Cargando video..."
        self._load_status.color = ft.Colors.ON_SURFACE_VARIANT
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
        self._load_status.color = ft.Colors.PRIMARY
        self._page.update()
        self._show_step(1)

    def _on_video_load_error(self, error: str) -> None:
        self._load_spinner.visible = False
        self._load_status.value = f"Error: {error}"
        self._load_status.color = ft.Colors.ERROR
        self._page.update()

    def _on_roi_confirmed(self, roi_points: list[tuple[int, int]]) -> None:
        self._roi_points = roi_points
        self._show_step(2)
        self._start_processing()

    def _start_processing(self) -> None:
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
        self, frame: int, total: int, in_count: int, out_count: int,
    ) -> None:
        try:
            pct = frame / total if total > 0 else 0
            self._process_progress.value = pct
            self._process_status.value = f"Frame {frame}/{total} ({pct:.0%})"
            self._process_status.color = ft.Colors.ON_SURFACE
            self._process_counters.value = f"IN: {in_count}   OUT: {out_count}"
            self._page.update()
        except Exception:
            pass

    def _on_processing_done(self, result: ProcessingResult) -> None:
        self._result = result
        self._process_progress.visible = False
        self._show_step(3)
        self._auto_sync_analysis()

    def _on_processing_error(self, error: str) -> None:
        self._process_progress.visible = False
        self._process_status.value = f"Error: {error}"
        self._process_status.color = ft.Colors.ERROR
        self._page.update()

    # ── Analysis sync ─────────────────────────────────────

    def _auto_sync_analysis(self) -> None:
        try:
            pending = SyncService.get_pending_analysis_count()
        except Exception:
            pending = 0

        if pending == 0:
            self._analysis_status.value = "Sin datos pendientes de sincronizacion."
            self._analysis_status.color = ft.Colors.ON_SURFACE_VARIANT
            self._refresh_counts()
            self._page.update()
            return

        self._analysis_spinner.visible = True
        self._analysis_status.value = f"Sincronizando {pending} registro(s)..."
        self._analysis_status.color = ft.Colors.ON_SURFACE_VARIANT
        self._page.update()

        def _do_sync():
            result = self._sync_service.sync_analysis_records()
            self._page.run_thread(lambda: self._on_analysis_sync_done(result))

        threading.Thread(target=_do_sync, daemon=True).start()

    def _on_analysis_sync_done(self, result) -> None:
        self._analysis_spinner.visible = False
        if result.success:
            self._analysis_status.value = f"Sincronizacion completada — {result.detail}"
            self._analysis_status.color = ft.Colors.PRIMARY
        else:
            self._analysis_status.value = f"Pendiente: {result.detail}"
            self._analysis_status.color = ft.Colors.TERTIARY
        self._refresh_counts()
        self._page.update()

    # ── Pipeline reset ────────────────────────────────────

    def _reset_pipeline(self) -> None:
        self._video_info = None
        self._result = None
        self._show_step(0)

    # ── Catalog sync ──────────────────────────────────────

    def _handle_sync_catalogs(self, _e) -> None:
        self._sync_catalog_btn.disabled = True
        self._catalog_spinner.visible = True
        self._catalog_status.value = "Sincronizando..."
        self._catalog_status.color = ft.Colors.ON_SURFACE_VARIANT
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
            self._catalog_status.color = ft.Colors.PRIMARY
        else:
            errors_summary = "; ".join(result.errors) if result.errors else result.detail
            self._catalog_status.value = f"Error: {errors_summary}"
            self._catalog_status.color = ft.Colors.ERROR
        self._refresh_counts()
        self._page.update()

    # ── Logout ────────────────────────────────────────────

    def _handle_logout(self, _e) -> None:
        from src.services.auth_service import AuthService
        AuthService().logout()
        self._on_logout()

    # ── Helpers ───────────────────────────────────────────

    def _refresh_counts(self) -> None:
        try:
            pending = SyncService.get_pending_analysis_count()
        except Exception:
            pending = 0
        if pending > 0:
            self._pending_text.value = f"{pending} registro(s) pendientes de sincronizacion"
        else:
            self._pending_text.value = "No hay registros pendientes"


# ── Module-level helpers ─────────────────────────────────

def _update_nav(item: ft.Container, active: bool) -> None:
    fg = _SB_FG if active else _SB_MUTED
    item.bgcolor = _SB_ACCENT if active else ft.Colors.TRANSPARENT
    item.content.controls[0].color = fg
    txt = item.content.controls[1]
    txt.color = fg
    txt.weight = ft.FontWeight.W_600 if active else ft.FontWeight.W_400


def _hover_sidebar(e, container: ft.Container):
    container.bgcolor = _SB_SUBTLE if e.data == "true" else ft.Colors.TRANSPARENT
    container.update()
