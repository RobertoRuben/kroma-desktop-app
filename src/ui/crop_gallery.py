import flet as ft
import numpy as np

from src.utils.image_utils import frame_to_base64


class CropGallery(ft.Column):
    """Galeria GridView de crops 224x224 extraidos del detector YOLO."""

    def __init__(
        self,
        crops: list[np.ndarray],
        crop_metadata: list[dict] | None = None,
    ):
        super().__init__()
        self.crops = crops
        self.crop_metadata = crop_metadata or []

        gallery_items = []
        for i, crop in enumerate(self.crops):
            crop_b64 = frame_to_base64(crop, quality=95)

            meta_text = ""
            if i < len(self.crop_metadata):
                meta = self.crop_metadata[i]
                meta_text = f"ID:{meta.get('track_id', '?')} F:{meta.get('frame', '?')}"

            gallery_items.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Image(
                                src=crop_b64,
                                width=112,
                                height=112,
                                fit=ft.BoxFit.CONTAIN,
                                border_radius=ft.BorderRadius.all(4),
                            ),
                            ft.Text(
                                f"#{i + 1}",
                                size=11,
                                weight=ft.FontWeight.BOLD,
                                text_align=ft.TextAlign.CENTER,
                            ),
                            ft.Text(
                                meta_text,
                                size=9,
                                color=ft.Colors.GREY_600,
                                text_align=ft.TextAlign.CENTER,
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=2,
                    ),
                    border=ft.Border.all(1, ft.Colors.GREY_400),
                    border_radius=8,
                    padding=6,
                    bgcolor=ft.Colors.GREY_100,
                )
            )

        grid = ft.GridView(
            controls=gallery_items,
            runs_count=5,
            child_aspect_ratio=0.75,
            spacing=10,
            run_spacing=10,
            expand=True,
        )

        self.controls = [
            ft.Text(
                f"Crops Extraidos ({len(self.crops)})",
                size=16,
                weight=ft.FontWeight.BOLD,
            ),
            ft.Container(
                content=grid,
                height=400,
                border=ft.Border.all(1, ft.Colors.GREY_300),
                border_radius=8,
                padding=10,
            ),
        ]
        self.spacing = 10
