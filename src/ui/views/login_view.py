"""Login view — first screen shown to the user."""

from __future__ import annotations

import threading
from typing import Callable

import flet as ft

from src.services.auth_service import AuthResult, AuthService


class LoginView(ft.Column):
    """A centred login card with username / password fields.

    Parameters
    ----------
    on_login_success:
        Callback invoked with the ``AuthResult`` when the user logs in
        successfully (either online or from local cache).
    page:
        The Flet page reference (needed for theme-aware styling and updates).
    """

    def __init__(
        self,
        on_login_success: Callable[[AuthResult], None],
        page: ft.Page,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._page = page
        self._on_login_success = on_login_success
        self._auth = AuthService()

        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.alignment = ft.MainAxisAlignment.CENTER
        self.expand = True

        # ── form controls ───────────────────────────────────
        self._username = ft.TextField(
            label="Usuario",
            prefix_icon=ft.Icons.PERSON,
            border_radius=10,
            width=340,
            autofocus=True,
            on_submit=self._handle_login,
        )
        self._password = ft.TextField(
            label="Contrasena",
            prefix_icon=ft.Icons.LOCK,
            password=True,
            can_reveal_password=True,
            border_radius=10,
            width=340,
            on_submit=self._handle_login,
        )
        self._error_text = ft.Text("", color=ft.Colors.RED_400, size=13, visible=False)
        self._login_btn = ft.ElevatedButton(
            "Iniciar Sesion",
            icon=ft.Icons.LOGIN,
            width=340,
            height=45,
            on_click=self._handle_login,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.PRIMARY,
                color=ft.Colors.ON_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=10),
            ),
        )
        self._spinner = ft.ProgressRing(width=24, height=24, visible=False)
        self._offline_text = ft.Text(
            "",
            size=12,
            italic=True,
            text_align=ft.TextAlign.CENTER,
            visible=False,
        )

        # ── layout ──────────────────────────────────────────
        self.controls = [
            ft.Container(expand=True),  # top spacer
            ft.Container(
                content=ft.Column(
                    [
                        ft.Image(
                            src="/public/logo-fruit-flow.png",
                            width=80,
                            height=80,
                            fit=ft.BoxFit.CONTAIN,
                        ),
                        ft.Text(
                            "Kroma Desktop",
                            size=28,
                            weight=ft.FontWeight.BOLD,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Text(
                            "Inicie sesion para continuar",
                            size=14,
                            text_align=ft.TextAlign.CENTER,
                            color=ft.Colors.ON_SURFACE_VARIANT,
                        ),
                        ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
                        self._username,
                        self._password,
                        self._error_text,
                        ft.Container(height=5),
                        ft.Row(
                            [self._login_btn, self._spinner],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        self._offline_text,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                ),
                padding=40,
                border_radius=16,
                bgcolor=ft.Colors.SURFACE,
                shadow=ft.BoxShadow(
                    blur_radius=24,
                    spread_radius=2,
                    color=ft.Colors.with_opacity(0.08, ft.Colors.BLACK),
                ),
                width=420,
            ),
            ft.Container(expand=True),  # bottom spacer
        ]

        # ── Try offline login automatically ─────────────────
        self._try_offline_auto()

    # ── handlers ────────────────────────────────────────────

    def _try_offline_auto(self) -> None:
        """If a cached user exists, offer instant offline login."""
        result = self._auth.login_offline()
        if result.success and result.user:
            self._offline_text.value = f"Sesion guardada de: {result.user.username} — presione Enter o el boton para continuar."
            self._offline_text.visible = True
            self._username.value = result.user.username
            self._username.disabled = True
            self._password.label = "Contrasena (opcional si offline)"

    def _handle_login(self, e) -> None:
        """Attempt online login; fall back to cached user."""
        self._error_text.visible = False
        self._spinner.visible = True
        self._login_btn.disabled = True
        self._page.update()

        username = self._username.value.strip()
        password = self._password.value.strip()

        def _do_login():
            # If password was given, try online first
            if password:
                result = self._auth.login_online(username, password)
                if result.success:
                    self._page.run_thread(lambda: self._finish_login(result))
                    return
                # If online fails with connectivity, try offline
                if "conexion" in (result.error or "").lower():
                    offline = self._auth.login_offline()
                    if offline.success:
                        self._page.run_thread(lambda: self._finish_login(offline))
                        return
                self._page.run_thread(
                    lambda: self._show_error(result.error or "Error desconocido")
                )
            else:
                # No password — try offline only
                offline = self._auth.login_offline()
                if offline.success:
                    self._page.run_thread(lambda: self._finish_login(offline))
                else:
                    self._page.run_thread(
                        lambda: self._show_error(
                            offline.error or "Ingrese su contrasena."
                        )
                    )

        threading.Thread(target=_do_login, daemon=True).start()

    def _finish_login(self, result: AuthResult) -> None:
        self._spinner.visible = False
        self._login_btn.disabled = False
        self._page.update()
        self._on_login_success(result)

    def _show_error(self, msg: str) -> None:
        self._error_text.value = msg
        self._error_text.visible = True
        self._spinner.visible = False
        self._login_btn.disabled = False
        self._page.update()
