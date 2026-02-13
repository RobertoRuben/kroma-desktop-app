import os
import threading

import cv2
import flet as ft
import torch

from src.processing.video_processor import VideoProcessor, apply_rotation, detect_video_rotation
from src.ui.crop_gallery import CropGallery
from src.ui.roi_canvas import ROICanvas
from src.utils.image_utils import (
    detect_hdr_transfer,
    frame_to_base64,
    resize_frame_for_display,
    tone_map_pq_frame,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
TRACKER_CONFIG = os.path.join(BASE_DIR, "botsort.yaml")
DEFAULT_CONFIDENCE = 0.40
DISPLAY_MAX_W = 800
DISPLAY_MAX_H = 600


def main(page: ft.Page):
    page.title = "Pepper Counter"
    page.window.width = 1000
    page.window.height = 900
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO
    page.theme_mode = ft.ThemeMode.LIGHT

    # --- State ---
    state = {
        "video_path": None,
        "first_frame": None,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "display_w": 0,
        "display_h": 0,
        "processing": False,
        "hdr_transfer": None,
    }

    # --- File Picker (Service) ---
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    # --- GPU Switch ---
    cuda_available = torch.cuda.is_available()
    gpu_switch = ft.Switch(
        label="GPU (CUDA)" if cuda_available else "GPU (no disponible)",
        value=cuda_available,
        disabled=not cuda_available,
    )

    # --- Step 1: Video Upload ---
    video_name_text = ft.Text(
        "Ningun video seleccionado", italic=True, color=ft.Colors.GREY_600
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
                            [btn_select_video, video_name_text],
                            spacing=15,
                            wrap=True,
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

    # --- Layout ---
    page.add(
        ft.Column(
            [
                ft.Text("Pepper Counter", size=28, weight=ft.FontWeight.BOLD),
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
        state["video_path"] = video_path
        video_name_text.value = os.path.basename(video_path)
        video_name_text.italic = False
        video_name_text.color = ft.Colors.BLACK
        video_name_text.update()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            show_snackbar("Error al abrir el video.", error=True)
            return

        rotation = detect_video_rotation(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            show_snackbar("No se pudo leer el primer frame del video.", error=True)
            return

        frame = apply_rotation(frame, rotation)

        # Detectar HDR y tone-map para display correcto
        hdr_transfer = detect_hdr_transfer(video_path)
        state["hdr_transfer"] = hdr_transfer
        if hdr_transfer == "pq":
            frame = tone_map_pq_frame(frame)

        state["first_frame"] = frame
        resized, sx, sy = resize_frame_for_display(frame, DISPLAY_MAX_W, DISPLAY_MAX_H)
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

        if not os.path.exists(MODEL_PATH):
            show_snackbar(f"No se encuentra el modelo: {MODEL_PATH}", error=True)
            return

        state["processing"] = True
        step3_card.visible = True
        progress_bar.value = 0
        progress_text.value = "Iniciando procesamiento..."
        counts_text.value = ""
        page.update()

        threading.Thread(
            target=run_processing, args=(roi_points,), daemon=True
        ).start()

    def run_processing(roi_points: list[tuple[int, int]]):
        try:
            processor = VideoProcessor(
                video_path=state["video_path"],
                model_path=MODEL_PATH,
                tracker_config=TRACKER_CONFIG,
                roi_points=roi_points,
                use_gpu=gpu_switch.value,
                confidence=DEFAULT_CONFIDENCE,
                hdr_transfer=state["hdr_transfer"],
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

            results = processor.process(progress_callback=on_progress)

            progress_bar.value = 1.0
            progress_text.value = "Procesamiento completado!"
            counts_text.value = (
                f"IN: {results['in_count']} | OUT: {results['out_count']} "
                f"| Total: {results['total']}"
            )
            page.update()

            show_results(results)

        except Exception as ex:
            show_snackbar(f"Error en procesamiento: {ex}", error=True)
            progress_text.value = f"Error: {ex}"
            page.update()
        finally:
            state["processing"] = False

    def show_results(results: dict):
        output_path = results["output_path"]
        crops = results["crops"]
        crop_metadata = results.get("crop_metadata", [])

        summary = ft.Container(
            content=ft.Column(
                [
                    ft.ResponsiveRow(
                        [
                            ft.Container(
                                content=_count_chip(
                                    "Total Frutos",
                                    str(results["total"]),
                                    ft.Colors.GREEN_700,
                                ),
                                col={"xs": 12, "sm": 4},
                            ),
                            ft.Container(
                                content=_count_chip(
                                    "IN",
                                    str(results["in_count"]),
                                    ft.Colors.BLUE_700,
                                ),
                                col={"xs": 6, "sm": 4},
                            ),
                            ft.Container(
                                content=_count_chip(
                                    "OUT",
                                    str(results["out_count"]),
                                    ft.Colors.ORANGE_700,
                                ),
                                col={"xs": 6, "sm": 4},
                            ),
                        ],
                        spacing=10,
                        run_spacing=10,
                    ),
                    ft.Text(
                        f"Frames procesados: {results['frames_processed']}/{results['total_frames']}",
                        size=12,
                        color=ft.Colors.GREY_600,
                    ),
                    ft.Row(
                        [
                            ft.Button(
                                "Abrir Video Procesado",
                                icon=ft.Icons.PLAY_CIRCLE,
                                on_click=lambda e: os.startfile(output_path),
                            ),
                            ft.OutlinedButton(
                                "Abrir Carpeta",
                                icon=ft.Icons.FOLDER_OPEN,
                                on_click=lambda e: os.startfile(
                                    os.path.dirname(output_path)
                                ),
                            ),
                        ],
                        spacing=10,
                        wrap=True,
                    ),
                ],
                spacing=10,
            ),
            padding=10,
        )

        results_container.controls.clear()
        results_container.controls.append(summary)

        if crops:
            gallery = CropGallery(crops=crops, crop_metadata=crop_metadata)
            results_container.controls.append(gallery)
        else:
            results_container.controls.append(
                ft.Text(
                    "No se detectaron objetos cruzando la zona de interes.",
                    italic=True,
                    color=ft.Colors.GREY_600,
                )
            )

        results_container.controls.append(
            ft.Button(
                "Procesar Otro Video",
                icon=ft.Icons.REFRESH,
                on_click=on_reset,
            )
        )

        results_container.visible = True
        step4_card.visible = True
        page.update()

    def on_reset(e):
        state["video_path"] = None
        state["first_frame"] = None
        state["processing"] = False
        state["hdr_transfer"] = None

        video_name_text.value = "Ningun video seleccionado"
        video_name_text.italic = True
        video_name_text.color = ft.Colors.GREY_600

        roi_container.content = None
        roi_container.visible = False
        step2_card.visible = False
        step3_card.visible = False
        step4_card.visible = False
        results_container.visible = False
        results_container.controls.clear()

        page.update()

    def show_snackbar(msg: str, error: bool = False):
        snack = ft.SnackBar(
            content=ft.Text(msg, color=ft.Colors.WHITE),
            bgcolor=ft.Colors.RED_700 if error else ft.Colors.GREEN_700,
            open=True,
        )
        page.show_dialog(snack)


def _count_chip(label: str, value: str, color: str) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(value, size=28, weight=ft.FontWeight.BOLD, color=color),
                ft.Text(label, size=12, color=ft.Colors.GREY_700),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=2,
        ),
        padding=ft.Padding.symmetric(horizontal=20, vertical=10),
        border=ft.Border.all(1, ft.Colors.GREY_300),
        border_radius=12,
        bgcolor=ft.Colors.WHITE,
        alignment=ft.Alignment.CENTER,
    )


if __name__ == "__main__":
    ft.run(main)
