"""Step 2 — video processing with GPU toggle and progress."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import flet as ft

from src.config import (
    CLS_MODEL_PATH,
    DEFAULT_CONFIDENCE,
    DET_MODEL_PATH,
    QUALITY_MODEL_PATH,
    TRACKER_CONFIG,
)
from src.schemas import ProcessingResult, VideoInfo

logger = logging.getLogger(__name__)


class PipelineStepProcess(ft.Column):
    """Step 2: run detection + tracking + classification, show progress."""

    def __init__(
        self,
        page: ft.Page,
        on_done: Callable[[ProcessingResult], None],
        on_error: Callable[[str], None],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._page = page
        self._on_done = on_done
        self._on_error = on_error
        self.spacing = 12

        # ── GPU state ──
        self._use_gpu = False
        self._gpu_switch = ft.Switch(
            label="GPU", value=False,
            on_change=self._handle_gpu_toggle,
            active_color=ft.Colors.PRIMARY,
        )
        self._gpu_status = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self._detect_gpu_availability()

        # ── Progress controls ──
        self._process_progress = ft.ProgressBar(
            visible=False,
            color=ft.Colors.PRIMARY,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
        )
        self._process_status = ft.Text("", size=13, color=ft.Colors.ON_SURFACE)
        self._process_counters = ft.Text(
            "", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.ON_SURFACE,
        )

    @property
    def gpu_controls(self) -> tuple[ft.Switch, ft.Text]:
        """Return GPU switch + status text for external layout placement."""
        return self._gpu_switch, self._gpu_status

    def start(self, video_info: VideoInfo, roi_points: list[tuple[int, int]]) -> None:
        """Build progress UI and start the processing worker thread."""
        self._process_progress.visible = True
        self._process_progress.value = 0
        self._process_status.value = "Iniciando procesamiento..."
        self._process_status.color = ft.Colors.ON_SURFACE_VARIANT
        self._process_counters.value = ""

        self.controls = [
            ft.Text(
                "Procesando video — deteccion, tracking y clasificacion...",
                size=13, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self._process_progress,
            self._process_status,
            self._process_counters,
        ]

        self._start_processing(video_info, roi_points)

    # ── GPU detection ──

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
        self._gpu_status.color = (
            ft.Colors.PRIMARY if self._use_gpu else ft.Colors.ON_SURFACE_VARIANT
        )
        self._page.update()

    # ── Processing thread ──

    def _start_processing(
        self, info: VideoInfo, roi_points: list[tuple[int, int]],
    ) -> None:
        use_gpu = self._use_gpu

        def _do_process():
            try:
                from src.processing.video_processor import VideoProcessor
                processor = VideoProcessor(
                    video_path=info.path,
                    model_path=DET_MODEL_PATH,
                    cls_model_path=CLS_MODEL_PATH,
                    quality_model_path=QUALITY_MODEL_PATH,
                    tracker_config=TRACKER_CONFIG,
                    roi_points=roi_points,
                    use_gpu=use_gpu,
                    confidence=DEFAULT_CONFIDENCE,
                    hdr_transfer=info.hdr_transfer,
                )
                result = processor.process(
                    progress_callback=self._on_progress,
                )
                self._page.run_thread(lambda: self._on_processing_done(result))
            except Exception as exc:
                logger.exception("Processing error")
                error_msg = str(exc)
                self._page.run_thread(
                    lambda: self._on_processing_error(error_msg)
                )

        threading.Thread(target=_do_process, daemon=True).start()

    def _on_progress(
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
        self._process_progress.visible = False
        self._on_done(result)

    def _on_processing_error(self, error: str) -> None:
        self._process_progress.visible = False
        self._process_status.value = f"Error: {error}"
        self._process_status.color = ft.Colors.ERROR
        self._page.update()
        self._on_error(error)
