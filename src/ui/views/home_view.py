"""Home view — sidebar navigation + 5-step video processing pipeline.

Thin orchestrator: owns pipeline state and wires callbacks between
child components. All heavy logic lives in the extracted components.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import flet as ft

from src.config import DISPLAY_MAX_H, DISPLAY_MAX_W
from src.model.auth import CachedUser
from src.schemas import ProcessingResult, VideoInfo
from src.services.sync_service import SyncService
from src.ui.components.pipeline_step_context import PipelineStepContext
from src.ui.components.pipeline_step_load import PipelineStepLoad
from src.ui.components.pipeline_step_process import PipelineStepProcess
from src.ui.components.profile_page import ProfilePage
from src.ui.components.roi_canvas import ROICanvas
from src.ui.components.sidebar import Sidebar
from src.ui.components.step_indicator import StepIndicator
from src.ui.components.sync_panel import AnalysisSyncStatus, CatalogSyncRow
from src.ui.views.results_view import ResultsView
from src.utils.image_utils import frame_to_base64, resize_frame_for_display

logger = logging.getLogger(__name__)


class HomeView(ft.Row):
    """Main view: green sidebar + content area with navigation."""

    def __init__(
        self,
        user: CachedUser,
        page: ft.Page,
        on_logout: Callable[[], None],
        dark_mode_btn: ft.IconButton | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._user = user
        self._page = page
        self._on_logout = on_logout

        self.spacing = 0
        self.expand = True
        self.vertical_alignment = ft.CrossAxisAlignment.STRETCH

        # ── Pipeline state (orchestrator owns) ────────────
        self._video_info: VideoInfo | None = None
        self._result: ProcessingResult | None = None
        self._roi_points: list[tuple[int, int]] = []
        self._production_unit_id: int | None = None
        self._current_step = 0

        # ── Services ──────────────────────────────────────
        sync_service = SyncService(access_token=user.access_token)
        self._file_picker = ft.FilePicker()
        page.services.append(self._file_picker)

        # ── Child components ──────────────────────────────
        self._sidebar = Sidebar(
            user=user,
            on_nav_change=self._show_nav,
            on_logout=self._handle_logout,
        )
        self._step_indicator = StepIndicator(current=0)
        self._step_context = PipelineStepContext(
            on_context_selected=self._on_context_selected,
        )
        self._step_load = PipelineStepLoad(
            page=page,
            file_picker=self._file_picker,
            on_video_loaded=self._on_video_loaded,
        )
        self._step_process = PipelineStepProcess(
            page=page,
            on_done=self._on_processing_done,
            on_error=self._on_processing_error,
        )
        self._catalog_sync = CatalogSyncRow(
            sync_service=sync_service,
            page=page,
        )
        self._analysis_sync = AnalysisSyncStatus(
            sync_service=sync_service,
            page=page,
        )

        # ── Layout containers ─────────────────────────────
        self._step_content = ft.Column(spacing=12, expand=True)
        self._main_content = ft.Column(
            spacing=20,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )

        # ── Content-area header (only above content, not sidebar)
        initials = user.username[0].upper() if user.username else "?"
        profile_avatar = ft.Container(
            content=ft.Text(
                initials,
                size=14,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.ON_PRIMARY,
                text_align=ft.TextAlign.CENTER,
            ),
            width=36,
            height=36,
            border_radius=18,
            bgcolor=ft.Colors.PRIMARY,
            alignment=ft.Alignment.CENTER,
            on_click=lambda _: self._show_nav(1),
            tooltip=user.username,
        )

        self._sidebar_collapsed = False
        self._sidebar_toggle = ft.IconButton(
            icon=ft.Icons.MENU,
            tooltip="Colapsar sidebar",
            on_click=self._toggle_sidebar,
            icon_size=20,
        )

        header_left = ft.Row(
            [self._sidebar_toggle],
            spacing=4,
        )

        header_right = ft.Row(
            [dark_mode_btn, profile_avatar] if dark_mode_btn else [profile_avatar],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        content_header = ft.Container(
            content=ft.Row(
                [header_left, header_right],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding(left=16, top=8, right=20, bottom=8),
        )

        self.controls = [
            self._sidebar,
            ft.Column(
                [
                    content_header,
                    ft.Divider(height=1, color=ft.Colors.OUTLINE),
                    ft.Container(
                        content=self._main_content,
                        expand=True,
                        padding=ft.Padding(left=28, top=12, right=28, bottom=24),
                    ),
                ],
                spacing=0,
                expand=True,
            ),
        ]

        self._show_nav(0)
        self._show_step(0)

    # ================================================================
    # Navigation
    # ================================================================

    def _show_nav(self, index: int) -> None:
        self._sidebar.set_active_nav(index)
        self._main_content.controls.clear()
        if index == 0:
            self._build_evaluation_page()
        else:
            self._main_content.controls = [ProfilePage(user=self._user)]
        try:
            self._page.update()
        except Exception:
            pass

    def _build_evaluation_page(self) -> None:
        self._main_content.controls = [
            ft.Text(
                "Evaluacion",
                size=22,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.ON_SURFACE,
            ),
            self._catalog_sync,
            self._build_pipeline_section(),
        ]

    def _build_pipeline_section(self) -> ft.Container:
        gpu_switch, gpu_status = self._step_process.gpu_controls
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
                                            size=20,
                                            color=ft.Colors.PRIMARY,
                                        ),
                                        ft.Text(
                                            "Procesamiento de Video",
                                            size=16,
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
                                    [gpu_switch, gpu_status],
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
    # Step routing
    # ================================================================

    def _show_step(self, step: int) -> None:
        self._current_step = step
        try:
            self._step_indicator.set_step(step)
        except Exception:
            pass

        self._step_content.controls.clear()

        try:
            if step == 0:
                self._step_content.controls = [self._step_context]
            elif step == 1:
                self._step_load.reset()
                self._step_content.controls = [self._step_load]
            elif step == 2:
                self._build_step_2()
            elif step == 3:
                self._step_content.controls = [self._step_process]
                self._step_process.start(self._video_info, self._roi_points)
            elif step == 4:
                self._build_step_4()
        except Exception as exc:
            logger.exception("Error building step %d", step)
            self._step_content.controls = [
                ft.Text(
                    f"Error al cargar paso {step}: {exc}",
                    color=ft.Colors.ERROR,
                    size=13,
                ),
            ]

        try:
            self._page.update()
        except Exception:
            pass

    def _build_step_2(self) -> None:
        """Step 2: ROI canvas on the first frame."""
        if not self._video_info or self._video_info.first_frame is None:
            self._step_content.controls = [
                ft.Text("Error: no se pudo cargar el video.", color=ft.Colors.ERROR),
            ]
            return

        frame = self._video_info.first_frame
        resized, scale_x, scale_y = resize_frame_for_display(
            frame,
            DISPLAY_MAX_W,
            DISPLAY_MAX_H,
        )
        display_h, display_w = resized.shape[:2]

        roi_canvas = ROICanvas(
            image_base64=frame_to_base64(resized),
            display_width=display_w,
            display_height=display_h,
            scale_x=scale_x,
            scale_y=scale_y,
            on_roi_confirmed=self._on_roi_confirmed,
        )

        back_btn = ft.TextButton(
            "Volver",
            icon=ft.Icons.ARROW_BACK,
            on_click=lambda _: self._show_step(1),
            style=ft.ButtonStyle(color=ft.Colors.PRIMARY),
        )

        info = self._video_info
        info_text = ft.Text(
            f"{os.path.basename(info.path)}  —  "
            f"{info.width}x{info.height}  |  {info.fps} FPS  |  "
            f"{info.total_frames} frames"
            f"{('  |  HDR: ' + info.hdr_transfer.upper()) if info.hdr_transfer else ''}",
            size=12,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )

        self._step_content.controls = [
            ft.Row(
                [back_btn, info_text],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                wrap=True,
            ),
            roi_canvas,
        ]

    def _build_step_4(self) -> None:
        """Step 4: results view + sync status."""
        if not self._result:
            self._step_content.controls = [
                ft.Text(
                    "No hay resultados disponibles.", color=ft.Colors.ON_SURFACE_VARIANT
                ),
            ]
            return

        results_view = ResultsView(
            result=self._result,
            crop_images=self._result.crop_images,
            page=self._page,
            on_reset=lambda _: self._reset_pipeline(),
        )

        self._step_content.controls = [
            results_view,
            ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
            self._analysis_sync,
        ]

    # ================================================================
    # Sidebar toggle
    # ================================================================

    def _toggle_sidebar(self, _) -> None:
        self._sidebar_collapsed = not self._sidebar_collapsed
        self._sidebar.visible = not self._sidebar_collapsed
        self._sidebar_toggle.icon = (
            ft.Icons.MENU_OPEN if self._sidebar_collapsed else ft.Icons.MENU
        )
        self._sidebar_toggle.tooltip = (
            "Expandir sidebar" if self._sidebar_collapsed else "Colapsar sidebar"
        )
        try:
            self._page.update()
        except Exception:
            pass

    # ================================================================
    # Orchestrator callbacks
    # ================================================================

    def _on_context_selected(self, production_unit_id: int) -> None:
        self._production_unit_id = production_unit_id
        self._show_step(1)

    def _on_video_loaded(self, info: VideoInfo) -> None:
        self._video_info = info
        self._show_step(2)

    def _on_roi_confirmed(self, roi_points: list[tuple[int, int]]) -> None:
        self._roi_points = roi_points
        self._show_step(3)

    def _on_processing_done(self, result: ProcessingResult) -> None:
        self._result = result
        self._show_step(4)
        self._analysis_sync.auto_sync()

    def _on_processing_error(self, error: str) -> None:
        pass  # Error is already displayed by PipelineStepProcess

    def _reset_pipeline(self) -> None:
        self._video_info = None
        self._result = None
        self._roi_points = []
        self._show_step(0)

    def _handle_logout(self) -> None:
        from src.services.auth_service import AuthService

        AuthService().logout()
        self._on_logout()
