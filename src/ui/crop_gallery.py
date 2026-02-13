import flet as ft
import numpy as np

from src.utils.image_utils import frame_to_base64


_RIPENESS_FLET_COLORS = {
    "brown": (ft.Colors.BROWN_400, ft.Colors.WHITE),
    "green": (ft.Colors.GREEN_600, ft.Colors.WHITE),
    "red": (ft.Colors.RED_600, ft.Colors.WHITE),
    "turning": (ft.Colors.ORANGE_600, ft.Colors.WHITE),
}

_ALL_CLASSES = ["green", "red", "turning", "brown"]


class CropGallery(ft.Column):
    """Galeria GridView de crops con filtro por madurez y eliminacion."""

    def __init__(
        self,
        crops: list[np.ndarray],
        crop_metadata: list[dict] | None = None,
        on_crops_changed: callable = None,
    ):
        super().__init__()
        self.crops = list(crops)
        self.crop_metadata = list(crop_metadata or [])
        self.on_crops_changed = on_crops_changed
        self._active_filter: str | None = None

        # Pre-encode todas las imagenes
        self._crop_b64: list[str] = [
            frame_to_base64(c, quality=95) for c in self.crops
        ]

        # --- Filter chips ---
        self._filter_chips = ft.Row(spacing=8, wrap=True)

        # --- Header con conteo ---
        self._header_text = ft.Text(size=16, weight=ft.FontWeight.BOLD)

        # --- Grid ---
        self._grid = ft.GridView(
            runs_count=5,
            child_aspect_ratio=0.65,
            spacing=10,
            run_spacing=10,
            expand=True,
        )

        self.controls = [
            self._header_text,
            self._filter_chips,
            ft.Container(
                content=self._grid,
                height=400,
                border=ft.Border.all(1, ft.Colors.GREY_300),
                border_radius=8,
                padding=10,
            ),
        ]
        self.spacing = 10
        self._rebuild()

    def _rebuild(self):
        """Reconstruye chips y grid segun filtro activo."""
        # Contar por clase
        class_counts = {c: 0 for c in _ALL_CLASSES}
        for meta in self.crop_metadata:
            r = meta.get("ripeness", "")
            if r in class_counts:
                class_counts[r] += 1

        # Filter chips
        chips = []
        all_selected = self._active_filter is None
        chips.append(
            ft.Chip(
                label=ft.Text(f"Todos ({len(self.crops)})"),
                selected=all_selected,
                on_select=lambda e: self._set_filter(None),
            )
        )
        for cls_name in _ALL_CLASSES:
            count = class_counts[cls_name]
            bg, _ = _RIPENESS_FLET_COLORS.get(cls_name, (ft.Colors.GREY_400, ft.Colors.WHITE))
            selected = self._active_filter == cls_name
            chips.append(
                ft.Chip(
                    label=ft.Text(f"{cls_name.capitalize()} ({count})"),
                    selected=selected,
                    selected_color=bg,
                    on_select=lambda e, c=cls_name: self._set_filter(c),
                )
            )
        self._filter_chips.controls = chips

        # Filtrar items
        visible_indices = []
        for i in range(len(self.crops)):
            meta = self.crop_metadata[i] if i < len(self.crop_metadata) else {}
            ripeness = meta.get("ripeness", "")
            if self._active_filter is None or ripeness == self._active_filter:
                visible_indices.append(i)

        # Header
        if self._active_filter:
            self._header_text.value = (
                f"Crops Extraidos — {self._active_filter.capitalize()} "
                f"({len(visible_indices)}/{len(self.crops)})"
            )
        else:
            self._header_text.value = f"Crops Extraidos ({len(self.crops)})"

        # Grid items
        items = []
        for i in visible_indices:
            items.append(self._build_card(i))
        self._grid.controls = items

    def _build_card(self, idx: int) -> ft.Container:
        meta = self.crop_metadata[idx] if idx < len(self.crop_metadata) else {}
        ripeness = meta.get("ripeness", "")
        ripeness_conf = meta.get("ripeness_conf", 0)

        bg_color, text_color = _RIPENESS_FLET_COLORS.get(
            ripeness, (ft.Colors.GREY_400, ft.Colors.WHITE)
        )
        ripeness_badge = ft.Container(
            content=ft.Text(
                f"{ripeness.capitalize()} {ripeness_conf:.0%}" if ripeness else "",
                size=10,
                weight=ft.FontWeight.BOLD,
                color=text_color,
                text_align=ft.TextAlign.CENTER,
            ),
            bgcolor=bg_color,
            border_radius=10,
            padding=ft.Padding.symmetric(horizontal=8, vertical=2),
        ) if ripeness else ft.Container()

        delete_btn = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_size=14,
            icon_color=ft.Colors.RED_400,
            tooltip="Eliminar",
            on_click=lambda e, i=idx: self._delete_crop(i),
            style=ft.ButtonStyle(padding=0),
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Stack(
                        [
                            ft.Image(
                                src=self._crop_b64[idx],
                                width=112,
                                height=112,
                                fit=ft.BoxFit.CONTAIN,
                                border_radius=ft.BorderRadius.all(4),
                            ),
                            ft.Container(
                                content=delete_btn,
                                alignment=ft.Alignment.TOP_RIGHT,
                            ),
                        ],
                        width=112,
                        height=112,
                    ),
                    ft.Text(
                        f"#{idx + 1}",
                        size=11,
                        weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ripeness_badge,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
            ),
            border=ft.Border.all(1, ft.Colors.GREY_400),
            border_radius=8,
            padding=6,
            bgcolor=ft.Colors.GREY_100,
        )

    def _set_filter(self, cls_name: str | None):
        self._active_filter = cls_name
        self._rebuild()
        self.update()

    def _delete_crop(self, idx: int):
        if idx < 0 or idx >= len(self.crops):
            return
        self.crops.pop(idx)
        self.crop_metadata.pop(idx)
        self._crop_b64.pop(idx)
        self._rebuild()
        self.update()
        if self.on_crops_changed:
            self.on_crops_changed(self.crops, self.crop_metadata)
