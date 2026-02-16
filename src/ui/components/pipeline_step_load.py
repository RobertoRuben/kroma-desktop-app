"""Step 0 — video file selection and loading."""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

import flet as ft

from src.schemas import VideoInfo
from src.services.video_loader import VideoLoader

logger = logging.getLogger(__name__)


class PipelineStepLoad(ft.Column):
    """Step 0: select video file, show loading progress, report result."""

    def __init__(
        self,
        page: ft.Page,
        file_picker: ft.FilePicker,
        on_video_loaded: Callable[[VideoInfo], None],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._page = page
        self._file_picker = file_picker
        self._on_video_loaded = on_video_loaded

        self._video_label = ft.Text(
            "Ningun video seleccionado",
            size=13, color=ft.Colors.ON_SURFACE_VARIANT, italic=True,
        )
        self._load_spinner = ft.ProgressRing(
            width=16, height=16, visible=False, color=ft.Colors.PRIMARY,
        )
        self._load_status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE)

        self.spacing = 12
        self._build()

    def reset(self) -> None:
        """Reset to initial 'no video selected' state."""
        self._video_label.value = "Ningun video seleccionado"
        self._video_label.italic = True
        self._video_label.color = ft.Colors.ON_SURFACE_VARIANT
        self._load_status.value = ""
        self._load_spinner.visible = False

    def _build(self) -> None:
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

        self.controls = [
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
                self._page.run_thread(lambda: self._on_loaded(info))
            except Exception as exc:
                error_msg = str(exc)
                self._page.run_thread(lambda: self._on_error(error_msg))

        threading.Thread(target=_do_load, daemon=True).start()

    def _on_loaded(self, info: VideoInfo) -> None:
        self._load_spinner.visible = False
        self._load_status.value = (
            f"{info.width}x{info.height} | {info.fps} FPS | {info.total_frames} frames"
        )
        self._load_status.color = ft.Colors.PRIMARY
        self._page.update()
        self._on_video_loaded(info)

    def _on_error(self, error: str) -> None:
        self._load_spinner.visible = False
        self._load_status.value = f"Error: {error}"
        self._load_status.color = ft.Colors.ERROR
        self._page.update()
