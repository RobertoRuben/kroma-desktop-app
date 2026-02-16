"""Profile page — user info card."""

from __future__ import annotations

import flet as ft

from src.model.auth import CachedUser


class ProfilePage(ft.Column):
    """Displays user profile information."""

    def __init__(self, user: CachedUser, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spacing = 20
        self._build(user)

    def _build(self, user: CachedUser) -> None:
        initials = user.username[0].upper() if user.username else "?"

        info_rows = [
            ("Usuario", user.username),
            ("Rol", user.role_name or "Sin rol"),
        ]
        detail_controls = []
        for label, value in info_rows:
            detail_controls.append(
                ft.Row(
                    [
                        ft.Text(label, size=13, color=ft.Colors.ON_SURFACE_VARIANT, width=100),
                        ft.Text(
                            value, size=13,
                            weight=ft.FontWeight.W_600, color=ft.Colors.ON_SURFACE,
                        ),
                    ],
                    spacing=12,
                )
            )

        self.controls = [
            ft.Text(
                "Perfil", size=22,
                weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE,
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Container(
                                    content=ft.Text(
                                        initials, size=28,
                                        weight=ft.FontWeight.BOLD,
                                        color=ft.Colors.ON_PRIMARY,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                    width=64, height=64, border_radius=32,
                                    bgcolor=ft.Colors.PRIMARY,
                                    alignment=ft.Alignment.CENTER,
                                ),
                                ft.Column(
                                    [
                                        ft.Text(
                                            user.username, size=18,
                                            weight=ft.FontWeight.BOLD,
                                            color=ft.Colors.ON_SURFACE,
                                        ),
                                        ft.Text(
                                            user.role_name or "Sin rol",
                                            size=13,
                                            color=ft.Colors.ON_SURFACE_VARIANT,
                                        ),
                                    ],
                                    spacing=2,
                                ),
                            ],
                            spacing=16,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
                        ft.Column(detail_controls, spacing=10),
                    ],
                    spacing=16,
                ),
                padding=24,
                border_radius=12,
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
                border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            ),
        ]
