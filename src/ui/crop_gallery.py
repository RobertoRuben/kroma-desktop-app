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


_DIRECTION_FLET_COLORS = {
    "in": (ft.Colors.BLUE_700, ft.Colors.WHITE),
    "out": (ft.Colors.ORANGE_700, ft.Colors.WHITE),
}


class CropGallery(ft.Column):
    """Galeria GridView de crops con filtro por madurez, direccion y eliminacion."""

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
        self._direction_filter: str | None = None  # "in", "out" o None (todos)

        # Pre-encode todas las imagenes
        self._crop_b64: list[str] = [frame_to_base64(c, quality=95) for c in self.crops]

        # --- Direction filter chips ---
        self._direction_chips = ft.Row(spacing=8, wrap=True)

        # --- Ripeness filter chips ---
        self._filter_chips = ft.Row(spacing=8, wrap=True)

        # --- Header con conteo ---
        self._header_text = ft.Text(size=16, weight=ft.FontWeight.BOLD)

        # --- Grid: max_extent para auto-calcular columnas segun ancho ---
        self._grid = ft.GridView(
            max_extent=150,
            child_aspect_ratio=0.65,
            spacing=10,
            run_spacing=10,
            expand=True,
        )

        self.controls = [
            self._header_text,
            ft.Row([self._direction_chips], scroll=ft.ScrollMode.AUTO),
            ft.Row([self._filter_chips], scroll=ft.ScrollMode.AUTO),
            ft.Container(
                content=self._grid,
                expand=True,
                height=420,
                border=ft.Border.all(1, ft.Colors.GREY_300),
                border_radius=8,
                padding=10,
            ),
        ]
        self.spacing = 10
        self._rebuild()

    def _rebuild(self):
        """Reconstruye chips y grid segun filtros activos."""
        # Contar por clase y direccion
        class_counts = {c: 0 for c in _ALL_CLASSES}
        dir_counts = {"in": 0, "out": 0}
        for meta in self.crop_metadata:
            r = meta.get("ripeness", "")
            d = meta.get("direction", "")
            if r in class_counts:
                class_counts[r] += 1
            if d in dir_counts:
                dir_counts[d] += 1

        # Direction chips
        dir_chips = []
        all_dir_selected = self._direction_filter is None
        dir_chips.append(
            ft.Chip(
                label=ft.Text(f"Ambos ({len(self.crops)})"),
                selected=all_dir_selected,
                on_select=lambda e: self._set_direction_filter(None),
            )
        )
        for d_name, d_label in [("in", "IN"), ("out", "OUT")]:
            count = dir_counts[d_name]
            bg, _ = _DIRECTION_FLET_COLORS.get(
                d_name, (ft.Colors.GREY_400, ft.Colors.WHITE)
            )
            selected = self._direction_filter == d_name
            dir_chips.append(
                ft.Chip(
                    label=ft.Text(f"{d_label} ({count})"),
                    selected=selected,
                    selected_color=bg,
                    on_select=lambda e, c=d_name: self._set_direction_filter(c),
                )
            )
        self._direction_chips.controls = [
            ft.Text("Direccion:", size=12, weight=ft.FontWeight.BOLD)
        ] + dir_chips

        # Ripeness filter chips
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
            bg, _ = _RIPENESS_FLET_COLORS.get(
                cls_name, (ft.Colors.GREY_400, ft.Colors.WHITE)
            )
            selected = self._active_filter == cls_name
            chips.append(
                ft.Chip(
                    label=ft.Text(f"{cls_name.capitalize()} ({count})"),
                    selected=selected,
                    selected_color=bg,
                    on_select=lambda e, c=cls_name: self._set_filter(c),
                )
            )
        self._filter_chips.controls = [
            ft.Text("Madurez:", size=12, weight=ft.FontWeight.BOLD)
        ] + chips

        # Filtrar items (por direccion Y madurez)
        visible_indices = []
        for i in range(len(self.crops)):
            meta = self.crop_metadata[i] if i < len(self.crop_metadata) else {}
            ripeness = meta.get("ripeness", "")
            direction = meta.get("direction", "")
            if (
                self._direction_filter is not None
                and direction != self._direction_filter
            ):
                continue
            if self._active_filter is not None and ripeness != self._active_filter:
                continue
            visible_indices.append(i)

        # Header
        filter_parts = []
        if self._direction_filter:
            filter_parts.append(self._direction_filter.upper())
        if self._active_filter:
            filter_parts.append(self._active_filter.capitalize())
        if filter_parts:
            self._header_text.value = (
                f"Crops Extraidos — {' / '.join(filter_parts)} "
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
        direction = meta.get("direction", "")

        bg_color, text_color = _RIPENESS_FLET_COLORS.get(
            ripeness, (ft.Colors.GREY_400, ft.Colors.WHITE)
        )
        ripeness_badge = (
            ft.Container(
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
            )
            if ripeness
            else ft.Container()
        )

        # Badge de direccion
        dir_bg, dir_tc = _DIRECTION_FLET_COLORS.get(
            direction, (ft.Colors.GREY_400, ft.Colors.WHITE)
        )
        direction_badge = (
            ft.Container(
                content=ft.Text(
                    direction.upper() if direction else "",
                    size=9,
                    weight=ft.FontWeight.BOLD,
                    color=dir_tc,
                    text_align=ft.TextAlign.CENTER,
                ),
                bgcolor=dir_bg,
                border_radius=10,
                padding=ft.Padding.symmetric(horizontal=6, vertical=1),
            )
            if direction and direction != "unknown"
            else ft.Container()
        )

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
                                fit=ft.BoxFit.CONTAIN,
                                border_radius=ft.BorderRadius.all(4),
                                expand=True,
                            ),
                            ft.Container(
                                content=delete_btn,
                                alignment=ft.Alignment.TOP_RIGHT,
                            ),
                        ],
                        expand=True,
                    ),
                    ft.Row(
                        [
                            ft.Text(
                                f"#{idx + 1}",
                                size=11,
                                weight=ft.FontWeight.BOLD,
                            ),
                            direction_badge,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=4,
                    ),
                    ripeness_badge,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
                expand=True,
            ),
            border=ft.Border.all(1, ft.Colors.GREY_400),
            border_radius=8,
            padding=6,
            bgcolor=ft.Colors.GREY_100,
            expand=True,
        )

    def _set_filter(self, cls_name: str | None):
        self._active_filter = cls_name
        self._rebuild()
        self.update()

    def _set_direction_filter(self, direction: str | None):
        self._direction_filter = direction
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
