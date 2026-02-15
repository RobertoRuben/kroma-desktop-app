"""Vista de resultados del procesamiento."""

import os
import shutil

import flet as ft
import numpy as np

from src.config import ALL_RIPENESS_CLASSES, RIPENESS_COLORS
from src.schemas import ProcessingResult
from src.ui.components.crop_gallery import CropGallery


class ResultsView(ft.Column):
    """Componente que muestra los resultados del procesamiento."""

    def __init__(
        self,
        result: ProcessingResult,
        crop_images: list[np.ndarray],
        palette: dict,
        page: ft.Page,
        on_reset: callable = None,
    ):
        super().__init__()
        self._result = result
        self._crop_images = crop_images
        self._pal = palette
        self._page = page
        self._on_reset = on_reset
        self.spacing = 10

        self._save_picker = ft.FilePicker()
        page.services.append(self._save_picker)

        self._build()

    def _build(self):
        pal = self._pal
        result = self._result

        summary = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.ANALYTICS, color=pal["primary"], size=20),
                            ft.Text("Resumen", size=16, weight=ft.FontWeight.BOLD, color=pal["card_foreground"]),
                        ],
                        spacing=8,
                    ),
                    ft.Container(
                        content=ft.ResponsiveRow(
                            [
                                ft.Container(
                                    content=_count_chip(
                                        "Total Frutos", str(result.total), pal["primary"], pal,
                                    ),
                                    col={"xs": 12, "sm": 4},
                                ),
                                ft.Container(
                                    content=_count_chip(
                                        "IN", str(result.in_count), pal["sidebar_accent"], pal,
                                    ),
                                    col={"xs": 6, "sm": 4},
                                ),
                                ft.Container(
                                    content=_count_chip(
                                        "OUT", str(result.out_count), pal["accent"], pal,
                                    ),
                                    col={"xs": 6, "sm": 4},
                                ),
                            ],
                            spacing=10,
                            run_spacing=10,
                        ),
                        bgcolor=pal["secondary"],
                        border_radius=10,
                        padding=10,
                    ),
                    _ripeness_breakdown(result, pal),
                    ft.Text(
                        f"Frames procesados: {result.frames_processed}/{result.total_frames}",
                        size=12,
                        color=pal["muted_foreground"],
                    ),
                    ft.Row(
                        [
                            ft.Button(
                                "Abrir Video Procesado",
                                icon=ft.Icons.PLAY_CIRCLE,
                                on_click=lambda e: os.startfile(result.output_path),
                            ),
                            ft.ElevatedButton(
                                "Guardar Video",
                                icon=ft.Icons.SAVE_ALT,
                                on_click=self._handle_save_video,
                                style=ft.ButtonStyle(
                                    bgcolor=pal["primary"],
                                    color=pal["primary_foreground"],
                                    shape=ft.RoundedRectangleBorder(radius=8),
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

        controls = [summary, ft.Divider(color=pal["border"])]

        if self._crop_images:
            gallery = CropGallery(
                crop_images=self._crop_images,
                crop_infos=result.crops,
                palette=pal,
            )
            controls.append(gallery)
        else:
            controls.append(
                ft.Text(
                    "No se detectaron objetos cruzando la zona de interes.",
                    italic=True,
                    color=pal["muted_foreground"],
                )
            )

        if self._on_reset:
            controls.append(
                ft.Button(
                    "Procesar Otro Video",
                    icon=ft.Icons.REFRESH,
                    on_click=self._on_reset,
                )
            )

        self.controls = controls

    async def _handle_save_video(self, _e) -> None:
        """Open save-file dialog to copy the processed video."""
        source = self._result.output_path
        default_name = os.path.basename(source)
        dest = await self._save_picker.save_file(
            dialog_title="Guardar video procesado",
            file_name=default_name,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["mp4"],
        )
        if not dest:
            return
        if not dest.lower().endswith(".mp4"):
            dest += ".mp4"
        try:
            shutil.copy2(source, dest)
            snack = ft.SnackBar(
                content=ft.Text(f"Video guardado en: {dest}"),
                bgcolor=self._pal["primary"],
                open=True,
            )
            self._page.overlay.append(snack)
            self._page.update()
        except Exception as exc:
            snack = ft.SnackBar(
                content=ft.Text(f"Error al guardar: {exc}"),
                bgcolor=self._pal["destructive"],
                open=True,
            )
            self._page.overlay.append(snack)
            self._page.update()


def _count_chip(label: str, value: str, color: str, pal: dict) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(value, size=32, weight=ft.FontWeight.BOLD, color=color),
                ft.Text(label, size=12, color=pal["muted_foreground"]),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=2,
        ),
        padding=ft.Padding.symmetric(horizontal=20, vertical=10),
        border=ft.Border.all(1, pal["border"]),
        border_radius=12,
        bgcolor=pal["card"],
        alignment=ft.Alignment.CENTER,
    )


def _ripeness_breakdown(result: ProcessingResult, pal: dict) -> ft.Container:
    """Tabla visual con desglose de madurez por IN/OUT."""

    def _num_cell(value: str, bold: bool = False) -> ft.Container:
        return ft.Container(
            ft.Text(
                value,
                size=11,
                weight=ft.FontWeight.BOLD if bold else None,
                text_align=ft.TextAlign.CENTER,
                color=pal["card_foreground"],
            ),
            alignment=ft.Alignment.CENTER,
            col={"xs": 2, "sm": 2},
        )

    header = ft.ResponsiveRow(
        [
            ft.Container(
                ft.Text("Madurez", size=11, weight=ft.FontWeight.BOLD,
                        color=pal["card_foreground"]),
                col={"xs": 6, "sm": 6},
            ),
            ft.Container(
                ft.Text("IN", size=11, weight=ft.FontWeight.BOLD,
                        color=pal["sidebar_accent"], text_align=ft.TextAlign.CENTER),
                alignment=ft.Alignment.CENTER,
                col={"xs": 2, "sm": 2},
            ),
            ft.Container(
                ft.Text("OUT", size=11, weight=ft.FontWeight.BOLD,
                        color=pal["accent"], text_align=ft.TextAlign.CENTER),
                alignment=ft.Alignment.CENTER,
                col={"xs": 2, "sm": 2},
            ),
            ft.Container(
                ft.Text("Total", size=11, weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER, color=pal["card_foreground"]),
                alignment=ft.Alignment.CENTER,
                col={"xs": 2, "sm": 2},
            ),
        ],
        spacing=4,
        run_spacing=0,
    )

    rows = [header, ft.Divider(height=1, color=pal["border"])]
    for cls in ALL_RIPENESS_CLASSES:
        color_hex = RIPENESS_COLORS[cls].hex
        row = ft.ResponsiveRow(
            [
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(width=10, height=10, bgcolor=color_hex, border_radius=5),
                            ft.Text(cls.value.capitalize(), size=11, color=pal["card_foreground"]),
                        ],
                        spacing=6,
                    ),
                    col={"xs": 6, "sm": 6},
                ),
                _num_cell(str(result.in_ripeness.get(cls))),
                _num_cell(str(result.out_ripeness.get(cls))),
                _num_cell(str(result.ripeness_counts.get(cls)), bold=True),
            ],
            spacing=4,
            run_spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        rows.append(row)

    return ft.Container(
        content=ft.Column(rows, spacing=4),
        padding=10,
        border=ft.Border.all(1, pal["border"]),
        border_radius=8,
        bgcolor=pal["card"],
    )
