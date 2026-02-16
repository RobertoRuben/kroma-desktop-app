"""Green sidebar — logo, nav items, logout."""

from __future__ import annotations

import os
from typing import Callable

import flet as ft

from src.model.auth import CachedUser
from src.ui.theme import LIGHT

# Sidebar palette (always green, theme-independent)
_SB_BG = LIGHT["sidebar_bg"]  # #427B61
_SB_FG = LIGHT["sidebar_fg"]  # #FFFFFF
_SB_ACCENT = LIGHT["sidebar_accent"]  # #4B9B77
_SB_BORDER = LIGHT["sidebar_border"]  # #36634F
_SB_MUTED = "#FFFFFFB3"
_SB_SUBTLE = "#FFFFFF1A"

_SIDEBAR_W = 220

# Logo path (resolved relative to project root)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_LOGO_PATH = os.path.join(_PROJECT_ROOT, "public", "logo-fruit-flow.png")


class Sidebar(ft.Container):
    """Green sidebar with logo, navigation items, and logout."""

    def __init__(
        self,
        user: CachedUser,
        on_nav_change: Callable[[int], None],
        on_logout: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._on_nav_change = on_nav_change
        self._on_logout = on_logout

        logo_section = ft.Row(
            [
                ft.Image(
                    src=_LOGO_PATH,
                    width=44,
                    height=44,
                    fit=ft.BoxFit.CONTAIN,
                ),
                ft.Column(
                    [
                        ft.Text(
                            "KROMA",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=_SB_FG,
                        ),
                        ft.Text(
                            "Vision Defined",
                            size=11,
                            color=_SB_MUTED,
                        ),
                    ],
                    spacing=0,
                ),
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._nav_items = [
            _build_nav_item(
                ft.Icons.VIDEOCAM_OUTLINED,
                "Evaluacion",
                active=True,
                on_click=lambda _: self._on_nav_change(0),
            ),
        ]

        logout_item = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.LOGOUT, size=17, color="#FFFFFF99"),
                    ft.Text("Cerrar Sesion", size=13, color="#FFFFFF99"),
                ],
                spacing=10,
            ),
            padding=ft.Padding(left=14, top=10, right=14, bottom=10),
            border_radius=8,
            on_click=lambda _: self._on_logout(),
            on_hover=lambda e: _hover_sidebar(e, logout_item),
            ink=True,
        )

        self.content = ft.Column(
            [
                ft.Container(
                    content=logo_section,
                    padding=ft.Padding(left=0, top=24, right=0, bottom=16),
                ),
                ft.Container(height=1, bgcolor=_SB_BORDER),
                ft.Container(
                    content=ft.Column(self._nav_items, spacing=4),
                    padding=ft.Padding(left=0, top=12, right=0, bottom=0),
                ),
                ft.Container(expand=True),
                ft.Container(height=1, bgcolor=_SB_BORDER),
                ft.Container(
                    content=logout_item,
                    padding=ft.Padding(left=0, top=8, right=0, bottom=0),
                ),
            ],
            spacing=0,
            expand=True,
        )
        self.width = _SIDEBAR_W
        self.padding = ft.Padding(left=12, top=0, right=12, bottom=16)
        self.bgcolor = _SB_BG
        self.border_radius = ft.BorderRadius(
            top_left=0,
            top_right=16,
            bottom_left=0,
            bottom_right=16,
        )

    def set_active_nav(self, index: int) -> None:
        """Highlight the nav item at *index*, dim the others."""
        for i, item in enumerate(self._nav_items):
            _update_nav(item, active=(i == index))


# ── Module helpers ─────────────────────────────────────────


def _build_nav_item(icon, label, active, on_click) -> ft.Container:
    fg = _SB_FG if active else _SB_MUTED
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(icon, size=18, color=fg),
                ft.Text(
                    label,
                    size=13,
                    color=fg,
                    weight=ft.FontWeight.W_600 if active else ft.FontWeight.W_400,
                ),
            ],
            spacing=10,
        ),
        padding=ft.Padding(left=14, top=10, right=14, bottom=10),
        border_radius=8,
        bgcolor=_SB_ACCENT if active else ft.Colors.TRANSPARENT,
        on_click=on_click,
        ink=True,
    )


def _update_nav(item: ft.Container, active: bool) -> None:
    fg = _SB_FG if active else _SB_MUTED
    item.bgcolor = _SB_ACCENT if active else ft.Colors.TRANSPARENT
    item.content.controls[0].color = fg
    txt = item.content.controls[1]
    txt.color = fg
    txt.weight = ft.FontWeight.W_600 if active else ft.FontWeight.W_400


def _hover_sidebar(e, container: ft.Container) -> None:
    container.bgcolor = _SB_SUBTLE if e.data == "true" else ft.Colors.TRANSPARENT
    container.update()
