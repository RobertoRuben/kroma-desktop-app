"""Galeria de crops con filtrado por madurez/direccion y paginacion."""

import flet as ft
import numpy as np

from src.config import ALL_RIPENESS_CLASSES, RIPENESS_COLORS
from src.enums import Direction, RipenessClass
from src.schemas import CropInfo
from src.ui.theme import LIGHT
from src.utils.image_utils import frame_to_base64

_PAGE_SIZE = 20


def _direction_colors(pal: dict) -> dict[str, tuple[str, str]]:
    return {
        Direction.IN: (pal["sidebar_accent"], pal["primary_foreground"]),
        Direction.OUT: (pal["accent"], pal["accent_foreground"]),
    }


class CropGallery(ft.Column):
    """Galeria GridView de crops con filtro por madurez, direccion, eliminacion y paginacion."""

    def __init__(
        self,
        crop_images: list[np.ndarray],
        crop_infos: list[CropInfo],
        on_crops_changed: callable = None,
        palette: dict | None = None,
    ):
        super().__init__()
        self._pal = palette or LIGHT
        self._crop_images = list(crop_images)
        self._crop_infos = list(crop_infos)
        self.on_crops_changed = on_crops_changed
        self._active_filter: RipenessClass | None = None
        self._direction_filter: Direction | None = None
        self._current_page: int = 0

        # Pre-encode todas las imagenes
        self._crop_b64: list[str] = [frame_to_base64(c, quality=95) for c in self._crop_images]

        self._direction_chips = ft.Row(spacing=8, wrap=True)
        self._filter_chips = ft.Row(spacing=8, wrap=True)
        self._header_text = ft.Text(
            size=16, weight=ft.FontWeight.BOLD, color=self._pal["card_foreground"]
        )
        self._grid = ft.GridView(
            max_extent=150, child_aspect_ratio=0.65, spacing=10, run_spacing=10, expand=True,
        )
        self._page_info = ft.Text(size=12, color=self._pal["muted_foreground"])
        self._btn_prev = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT,
            on_click=lambda e: self._change_page(-1),
            disabled=True,
            icon_color=self._pal["primary"],
        )
        self._btn_next = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT,
            on_click=lambda e: self._change_page(1),
            disabled=True,
            icon_color=self._pal["primary"],
        )
        self._pagination_row = ft.Row(
            [self._btn_prev, self._page_info, self._btn_next],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=4,
        )
        self._grid_container = ft.Container(
            content=self._grid,
            expand=True,
            height=420,
            border=ft.Border.all(1, self._pal["border"]),
            border_radius=8,
            padding=10,
            bgcolor=self._pal["card"],
        )

        self.controls = [
            self._header_text,
            ft.Row([self._direction_chips], scroll=ft.ScrollMode.AUTO),
            ft.Row([self._filter_chips], scroll=ft.ScrollMode.AUTO),
            self._grid_container,
            self._pagination_row,
        ]
        self.spacing = 10
        self._rebuild()

    def _rebuild(self):
        pal = self._pal
        dir_colors = _direction_colors(pal)
        total_crops = len(self._crop_images)

        # Contar por clase y direccion
        class_counts: dict[RipenessClass, int] = {c: 0 for c in ALL_RIPENESS_CLASSES}
        dir_counts: dict[Direction, int] = {Direction.IN: 0, Direction.OUT: 0}
        for info in self._crop_infos:
            if info.ripeness in class_counts:
                class_counts[info.ripeness] += 1
            if info.direction in dir_counts:
                dir_counts[info.direction] += 1

        # Direction chips
        dir_chips = []
        dir_chips.append(
            ft.Chip(
                label=ft.Text(f"Ambos ({total_crops})"),
                selected=self._direction_filter is None,
                on_select=lambda e: self._set_direction_filter(None),
            )
        )
        for d, d_label in [(Direction.IN, "IN"), (Direction.OUT, "OUT")]:
            count = dir_counts[d]
            bg, _ = dir_colors.get(d, (pal["muted"], pal["foreground"]))
            dir_chips.append(
                ft.Chip(
                    label=ft.Text(f"{d_label} ({count})"),
                    selected=self._direction_filter == d,
                    selected_color=bg,
                    on_select=lambda e, c=d: self._set_direction_filter(c),
                )
            )
        self._direction_chips.controls = [
            ft.Text("Direccion:", size=12, weight=ft.FontWeight.BOLD, color=pal["card_foreground"])
        ] + dir_chips

        # Ripeness filter chips
        chips = []
        chips.append(
            ft.Chip(
                label=ft.Text(f"Todos ({total_crops})"),
                selected=self._active_filter is None,
                on_select=lambda e: self._set_filter(None),
            )
        )
        for cls in ALL_RIPENESS_CLASSES:
            count = class_counts[cls]
            bg = RIPENESS_COLORS[cls].hex
            chips.append(
                ft.Chip(
                    label=ft.Text(f"{cls.value.capitalize()} ({count})"),
                    selected=self._active_filter == cls,
                    selected_color=bg,
                    on_select=lambda e, c=cls: self._set_filter(c),
                )
            )
        self._filter_chips.controls = [
            ft.Text("Madurez:", size=12, weight=ft.FontWeight.BOLD, color=pal["card_foreground"])
        ] + chips

        # Filtrar items
        self._visible_indices = []
        for i, info in enumerate(self._crop_infos):
            if self._direction_filter is not None and info.direction != self._direction_filter:
                continue
            if self._active_filter is not None and info.ripeness != self._active_filter:
                continue
            self._visible_indices.append(i)

        # Header
        filter_parts = []
        if self._direction_filter:
            filter_parts.append(self._direction_filter.value.upper())
        if self._active_filter:
            filter_parts.append(self._active_filter.value.capitalize())
        if filter_parts:
            self._header_text.value = (
                f"Crops Extraidos — {' / '.join(filter_parts)} "
                f"({len(self._visible_indices)}/{total_crops})"
            )
        else:
            self._header_text.value = f"Crops Extraidos ({total_crops})"

        # Paginacion
        total_visible = len(self._visible_indices)
        total_pages = max(1, (total_visible + _PAGE_SIZE - 1) // _PAGE_SIZE)
        if self._current_page >= total_pages:
            self._current_page = max(0, total_pages - 1)
        start = self._current_page * _PAGE_SIZE
        end = min(start + _PAGE_SIZE, total_visible)
        page_indices = self._visible_indices[start:end]

        self._page_info.value = f"Pagina {self._current_page + 1} de {total_pages}"
        self._btn_prev.disabled = self._current_page <= 0
        self._btn_next.disabled = self._current_page >= total_pages - 1

        items = [self._build_card(i) for i in page_indices]
        self._grid.controls = items

    def _build_card(self, idx: int) -> ft.Container:
        pal = self._pal
        dir_colors = _direction_colors(pal)
        info = self._crop_infos[idx]

        colors = RIPENESS_COLORS.get(info.ripeness)
        bg_color = colors.hex if colors else pal["muted"]
        text_color = colors.text_color if colors else pal["foreground"]

        ripeness_badge = ft.Container(
            content=ft.Text(
                f"{info.ripeness.value.capitalize()} {info.ripeness_conf:.0%}",
                size=10, weight=ft.FontWeight.BOLD, color=text_color,
                text_align=ft.TextAlign.CENTER,
            ),
            bgcolor=bg_color,
            border_radius=10,
            padding=ft.Padding.symmetric(horizontal=8, vertical=2),
        )

        dir_bg, dir_tc = dir_colors.get(
            info.direction, (pal["muted"], pal["foreground"])
        )
        direction_badge = (
            ft.Container(
                content=ft.Text(
                    info.direction.value.upper(), size=9, weight=ft.FontWeight.BOLD,
                    color=dir_tc, text_align=ft.TextAlign.CENTER,
                ),
                bgcolor=dir_bg, border_radius=10,
                padding=ft.Padding.symmetric(horizontal=6, vertical=1),
            )
            if info.direction != Direction.UNKNOWN
            else ft.Container()
        )

        delete_btn = ft.IconButton(
            icon=ft.Icons.CLOSE, icon_size=14, icon_color=pal["destructive"],
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
                                src=self._crop_b64[idx], fit=ft.BoxFit.CONTAIN,
                                border_radius=ft.BorderRadius.all(4), expand=True,
                            ),
                            ft.Container(content=delete_btn, alignment=ft.Alignment.TOP_RIGHT),
                        ],
                        expand=True,
                    ),
                    ft.Row(
                        [
                            ft.Text(f"#{idx + 1}", size=11, weight=ft.FontWeight.BOLD, color=pal["card_foreground"]),
                            direction_badge,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER, spacing=4,
                    ),
                    ripeness_badge,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2, expand=True,
            ),
            border=ft.Border.all(1, pal["border"]),
            border_radius=10, padding=6, bgcolor=pal["secondary"], expand=True,
            shadow=ft.BoxShadow(blur_radius=4, color="#10000000"),
        )

    def _set_filter(self, cls: RipenessClass | None):
        self._active_filter = cls
        self._current_page = 0
        self._rebuild()
        self.update()

    def _set_direction_filter(self, direction: Direction | None):
        self._direction_filter = direction
        self._current_page = 0
        self._rebuild()
        self.update()

    def _change_page(self, delta: int):
        self._current_page += delta
        self._rebuild()
        self.update()

    def _delete_crop(self, idx: int):
        if idx < 0 or idx >= len(self._crop_images):
            return
        self._crop_images.pop(idx)
        self._crop_infos.pop(idx)
        self._crop_b64.pop(idx)
        self._rebuild()
        self.update()
        if self.on_crops_changed:
            self.on_crops_changed(self._crop_images, self._crop_infos)
