"""Kroma Desktop — Flet application entry point.

Startup flow
------------
1. Initialize SQLite database (create tables if needed).
2. Check for a cached user session.
   - If found → show HomeView directly (offline-first).
   - If not  → show LoginView.
3. After login → show HomeView with sync controls + processing pipeline.
"""

import logging

import flet as ft

from src.db import create_db_and_tables
from src.services.auth_service import AuthResult, AuthService
from src.ui.theme import build_dark_theme, build_light_theme
from src.ui.views.home_view import HomeView
from src.ui.views.login_view import LoginView

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(page: ft.Page) -> None:
    # ── window & theme setup ────────────────────────────────
    page.title = "Kroma Desktop"
    page.window.width = 1000
    page.window.height = 800
    page.padding = 0
    page.theme = build_light_theme()
    page.dark_theme = build_dark_theme()
    page.theme_mode = ft.ThemeMode.LIGHT

    # ── Database initialisation ─────────────────────────────
    logger.info("Initializing database...")
    create_db_and_tables()
    logger.info("Database ready.")

    # ── Dark mode toggle ────────────────────────────────────
    def _toggle_dark_mode(e):
        if page.theme_mode == ft.ThemeMode.LIGHT:
            page.theme_mode = ft.ThemeMode.DARK
            dark_mode_btn.icon = ft.Icons.LIGHT_MODE
            dark_mode_btn.tooltip = "Cambiar a modo claro"
        else:
            page.theme_mode = ft.ThemeMode.LIGHT
            dark_mode_btn.icon = ft.Icons.DARK_MODE
            dark_mode_btn.tooltip = "Cambiar a modo oscuro"
        page.update()

    dark_mode_btn = ft.IconButton(
        icon=ft.Icons.DARK_MODE,
        tooltip="Cambiar a modo oscuro",
        on_click=_toggle_dark_mode,
    )

    # ── Navigation helper ───────────────────────────────────
    content_area = ft.Column(expand=True)

    def _show_login():
        """Display the login view."""
        content_area.controls.clear()
        login_view = LoginView(
            on_login_success=_on_login_success,
            page=page,
        )
        content_area.controls.append(login_view)
        page.update()

    def _show_home(user):
        """Display the home view with sync controls."""
        content_area.controls.clear()
        home_view = HomeView(
            user=user,
            page=page,
            on_logout=_on_logout,
            dark_mode_btn=dark_mode_btn,
        )
        content_area.controls.append(home_view)
        page.update()

    def _on_login_success(result: AuthResult):
        """Callback from LoginView on successful authentication."""
        if result.success and result.user:
            logger.info(
                "Login successful for user=%s (offline=%s)",
                result.user.username,
                result.offline,
            )
            _show_home(result.user)

    def _on_logout():
        """Callback from HomeView when user logs out."""
        logger.info("User logged out.")
        _show_login()

    # ── Page layout ─────────────────────────────────────────
    page.add(content_area)

    # ── Auto-login attempt ──────────────────────────────────
    auth = AuthService()
    cached = auth.get_cached_user()
    if cached:
        logger.info("Found cached session for user=%s", cached.username)
        _show_home(cached)
    else:
        _show_login()


if __name__ == "__main__":
    ft.run(main, assets_dir=".")
