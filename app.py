"""Pepper Counter — Flet desktop application entry point."""

import os
import threading

import flet as ft

from src.config import (
    CLS_MODEL_PATH,
    DEFAULT_CONFIDENCE,
    DET_MODEL_PATH,
    DISPLAY_MAX_H,
    DISPLAY_MAX_W,
    TRACKER_CONFIG,
)
from src.processing.video_processor import VideoProcessor
from src.schemas import ProcessingResult
from src.services.video_loader import VideoLoader
from src.ui.components.roi_canvas import ROICanvas
from src.ui.theme import LIGHT, build_dark_theme, build_light_theme, palette
from src.ui.views.results_view import ResultsView
from src.utils.image_utils import frame_to_base64, resize_frame_for_display


def main(page: ft.Page):
    page.title = "Pepper Counter"
    page.window.width = 1000
    page.window.height = 900
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO
    page.theme = build_light_theme()
    page.dark_theme = build_dark_theme()
    page.theme_mode = ft.ThemeMode.LIGHT

    # --- State ---
    state = {
        "video_info": None,  # VideoInfo | None
        "scale_x": 1.0,
        "scale_y": 1.0,
        "display_w": 0,
        "display_h": 0,
        "processing": False,
        "last_result": None,  # ProcessingResult | None
        "last_crop_images": None,  # list[np.ndarray] | None
    }

    # --- File Picker (Service) ---
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    # --- GPU Switch ---
    try:
        import onnxruntime as ort

        cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        cuda_available = False
    gpu_switch = ft.Switch(
        label="GPU (CUDA)" if cuda_available else "GPU (no disponible)",
        value=cuda_available,
        disabled=not cuda_available,
    )

    # --- Step 1: Video Upload ---
    video_name_text = ft.Text(
        "Ningun video seleccionado", italic=True, color=LIGHT["muted_foreground"]
    )

    async def on_select_video(e):
        result = await file_picker.pick_files(
            dialog_title="Seleccionar video",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["mp4", "avi", "mov", "mkv"],
            allow_multiple=False,
        )
        if result:
            on_file_picked(result)

    btn_select_video = ft.Button(
        "Seleccionar Video",
        icon=ft.Icons.VIDEO_FILE,
        on_click=on_select_video,
    )

    step1_card = ft.Card(
        content=ft.Container(
            content=ft.ResponsiveRow(
                [
                    ft.Text(
                        "Paso 1: Cargar Video",
                        size=18,
                        weight=ft.FontWeight.BOLD,
                        col=12,
                    ),
                    ft.Container(
                        content=ft.Row(
                            [btn_select_video, video_name_text], spacing=15, wrap=True
                        ),
                        col={"sm": 12, "md": 8},
                    ),
                    ft.Container(content=gpu_switch, col={"sm": 12, "md": 4}),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                run_spacing=10,
            ),
            padding=20,
        ),
    )

    # --- Step 2: ROI Drawing ---
    roi_container = ft.Container(visible=False)
    step2_card = ft.Card(
        content=ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Paso 2: Definir Zona de Interes (ROI)",
                        size=18,
                        weight=ft.FontWeight.BOLD,
                    ),
                    roi_container,
                ],
                spacing=10,
            ),
            padding=20,
        ),
        visible=False,
    )

    # --- Step 3: Processing ---
    progress_bar = ft.ProgressBar(value=0, expand=True)
    progress_text = ft.Text("Esperando...", size=13)
    counts_text = ft.Text("", size=13)

    step3_card = ft.Card(
        content=ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Paso 3: Procesamiento", size=18, weight=ft.FontWeight.BOLD
                    ),
                    progress_text,
                    progress_bar,
                    counts_text,
                ],
                spacing=10,
            ),
            padding=20,
        ),
        visible=False,
    )

    # --- Step 4: Results ---
    results_container = ft.Column(visible=False)
    step4_card = ft.Card(
        content=ft.Container(
            content=ft.Column(
                [
                    ft.Text("Resultados", size=18, weight=ft.FontWeight.BOLD),
                    results_container,
                ],
                spacing=10,
            ),
            padding=20,
        ),
        visible=False,
    )

    # --- Dark Mode Toggle ---
    def _toggle_dark_mode(e):
        if page.theme_mode == ft.ThemeMode.LIGHT:
            page.theme_mode = ft.ThemeMode.DARK
            dark_mode_btn.icon = ft.Icons.LIGHT_MODE
            dark_mode_btn.tooltip = "Cambiar a modo claro"
        else:
            page.theme_mode = ft.ThemeMode.LIGHT
            dark_mode_btn.icon = ft.Icons.DARK_MODE
            dark_mode_btn.tooltip = "Cambiar a modo oscuro"

        pal = palette(page.theme_mode == ft.ThemeMode.DARK)
        if state.get("video_info"):
            video_name_text.color = pal["foreground"]
        else:
            video_name_text.color = pal["muted_foreground"]

        if results_container.visible and state.get("last_result"):
            show_results(state["last_result"], state["last_crop_images"])
        page.update()

    dark_mode_btn = ft.IconButton(
        icon=ft.Icons.DARK_MODE,
        tooltip="Cambiar a modo oscuro",
        on_click=_toggle_dark_mode,
    )

    # --- Layout ---
    page.add(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("Pepper Counter", size=28, weight=ft.FontWeight.BOLD),
                        dark_mode_btn,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                step1_card,
                step2_card,
                step3_card,
                step4_card,
            ],
            spacing=15,
            expand=True,
        )
    )

    # --- Handlers ---
    def on_file_picked(files: list):
        if not files:
            return

        video_path = files[0].path
        try:
            video_info = VideoLoader.load(video_path)
        except RuntimeError as ex:
            show_snackbar(str(ex), error=True)
            return

        state["video_info"] = video_info
        pal = palette(page.theme_mode == ft.ThemeMode.DARK)
        video_name_text.value = os.path.basename(video_path)
        video_name_text.italic = False
        video_name_text.color = pal["foreground"]
        video_name_text.update()

        available_w = int((page.width or 960) - 100)
        display_max_w = max(400, min(available_w, DISPLAY_MAX_W))
        resized, sx, sy = resize_frame_for_display(
            video_info.first_frame, display_max_w, DISPLAY_MAX_H
        )
        state["scale_x"] = sx
        state["scale_y"] = sy
        state["display_w"] = resized.shape[1]
        state["display_h"] = resized.shape[0]

        frame_b64 = frame_to_base64(resized, quality=90)

        roi_canvas = ROICanvas(
            image_base64=frame_b64,
            display_width=state["display_w"],
            display_height=state["display_h"],
            scale_x=state["scale_x"],
            scale_y=state["scale_y"],
            on_roi_confirmed=on_roi_confirmed,
            palette=pal,
        )

        roi_container.content = roi_canvas
        roi_container.visible = True
        step2_card.visible = True
        step3_card.visible = False
        step4_card.visible = False
        results_container.visible = False
        results_container.controls.clear()
        page.update()

    def on_roi_confirmed(roi_points: list[tuple[int, int]]):
        if state["processing"]:
            return
        if len(roi_points) < 2:
            show_snackbar("Se necesitan al menos 2 puntos para el ROI.", error=True)
            return

        if not os.path.exists(DET_MODEL_PATH):
            show_snackbar(f"No se encuentra el modelo: {DET_MODEL_PATH}", error=True)
            return
        if not os.path.exists(CLS_MODEL_PATH):
            show_snackbar(
                f"No se encuentra el clasificador: {CLS_MODEL_PATH}", error=True
            )
            return

        state["processing"] = True
        step3_card.visible = True
        progress_bar.value = 0
        progress_text.value = "Iniciando procesamiento..."
        counts_text.value = ""
        page.update()

        threading.Thread(target=run_processing, args=(roi_points,), daemon=True).start()

    def run_processing(roi_points: list[tuple[int, int]]):
        try:
            video_info = state["video_info"]
            processor = VideoProcessor(
                video_path=video_info.path,
                model_path=DET_MODEL_PATH,
                cls_model_path=CLS_MODEL_PATH,
                tracker_config=TRACKER_CONFIG,
                roi_points=roi_points,
                use_gpu=gpu_switch.value,
                confidence=DEFAULT_CONFIDENCE,
                hdr_transfer=video_info.hdr_transfer,
            )

            def on_progress(frame_num, total_frames, in_count, out_count):
                pct = frame_num / total_frames if total_frames > 0 else 0
                progress_bar.value = pct
                progress_text.value = (
                    f"Frame {frame_num}/{total_frames} ({pct * 100:.1f}%)"
                )
                counts_text.value = (
                    f"IN: {in_count} | OUT: {out_count} | Total: {in_count + out_count}"
                )
                page.update()

            result = processor.process(progress_callback=on_progress)

            progress_bar.value = 1.0
            progress_text.value = "Procesamiento completado!"
            counts_text.value = f"IN: {result.in_count} | OUT: {result.out_count} | Total: {result.total}"
            page.update()

            show_results(result, result.crop_images)

        except Exception as ex:
            show_snackbar(f"Error en procesamiento: {ex}", error=True)
            progress_text.value = f"Error: {ex}"
            page.update()
        finally:
            state["processing"] = False

    def show_results(result: ProcessingResult, crop_images: list):
        pal = palette(page.theme_mode == ft.ThemeMode.DARK)
        state["last_result"] = result
        state["last_crop_images"] = crop_images

        results_view = ResultsView(
            result=result,
            crop_images=crop_images,
            palette=pal,
            on_reset=on_reset,
        )

        results_container.controls.clear()
        results_container.controls.append(results_view)
        results_container.visible = True
        step4_card.visible = True
        page.update()

    def on_reset(e):
        pal = palette(page.theme_mode == ft.ThemeMode.DARK)
        state["video_info"] = None
        state["processing"] = False
        state["last_result"] = None
        state["last_crop_images"] = None

        video_name_text.value = "Ningun video seleccionado"
        video_name_text.italic = True
        video_name_text.color = pal["muted_foreground"]

        roi_container.content = None
        roi_container.visible = False
        step2_card.visible = False
        step3_card.visible = False
        step4_card.visible = False
        results_container.visible = False
        results_container.controls.clear()
        page.update()

    def show_snackbar(msg: str, error: bool = False):
        pal = palette(page.theme_mode == ft.ThemeMode.DARK)
        snack = ft.SnackBar(
            content=ft.Text(msg, color="#FFFFFF"),
            bgcolor=pal["destructive"] if error else pal["primary"],
            open=True,
        )
        page.show_dialog(snack)


if __name__ == "__main__":
    ft.run(main)
