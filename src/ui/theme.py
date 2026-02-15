"""Tema personalizado Pepper Counter — paleta verde agrícola con modo claro y oscuro."""

import flet as ft

# ──────────────────────────────────────────────
# Paleta de colores — tokens semánticos
# ──────────────────────────────────────────────

# Modo claro
LIGHT = {
    "background": "#F6FAF7",
    "foreground": "#23332B",
    "card": "#FFFFFF",
    "card_foreground": "#23332B",
    "primary": "#4B8C6B",
    "primary_foreground": "#FFFFFF",
    "primary_hover": "#3E9C6B",
    "primary_light": "#F2FCF9",
    "secondary": "#F0F6F3",
    "secondary_foreground": "#23332B",
    "muted": "#E3ECE7",
    "muted_foreground": "#7CA18C",
    "accent": "#C18A3A",
    "accent_foreground": "#FFFFFF",
    "destructive": "#E04A4A",
    "destructive_foreground": "#FFFFFF",
    "border": "#E3ECE7",
    "input": "#E3ECE7",
    "ring": "#4B8C6B",
    "sidebar_bg": "#4B8C6B",
    "sidebar_fg": "#FFFFFF",
    "sidebar_accent": "#3E9C6B",
    "sidebar_border": "#3A5C4A",
    "step_completed": "#4B8C6B",
    "step_pending": "#E3ECE7",
}

# Modo oscuro
DARK = {
    "background": "#0E1A17",
    "foreground": "#F6FAF7",
    "card": "#1A2B23",
    "card_foreground": "#F6FAF7",
    "primary": "#4B8C6B",
    "primary_foreground": "#FFFFFF",
    "primary_hover": "#3E9C6B",
    "primary_light": "#1A2B23",
    "secondary": "#23332B",
    "secondary_foreground": "#F6FAF7",
    "muted": "#232E29",
    "muted_foreground": "#A3C2B2",
    "accent": "#C18A3A",
    "accent_foreground": "#FFFFFF",
    "destructive": "#E04A4A",
    "destructive_foreground": "#FFFFFF",
    "border": "#232E29",
    "input": "#232E29",
    "ring": "#4B8C6B",
    "sidebar_bg": "#4B8C6B",
    "sidebar_fg": "#FFFFFF",
    "sidebar_accent": "#3E9C6B",
    "sidebar_border": "#2A3A2A",
    "step_completed": "#4B8C6B",
    "step_pending": "#232E29",
}


# ──────────────────────────────────────────────
# Mapeo a ft.ColorScheme de Material 3
# ──────────────────────────────────────────────
def _build_scheme(p: dict) -> ft.ColorScheme:
    return ft.ColorScheme(
        primary=p["primary"],
        on_primary=p["primary_foreground"],
        primary_container=p["primary_light"],
        on_primary_container=p["primary"],
        secondary=p["secondary"],
        on_secondary=p["secondary_foreground"],
        secondary_container=p["secondary"],
        on_secondary_container=p["secondary_foreground"],
        tertiary=p["accent"],
        on_tertiary=p["accent_foreground"],
        tertiary_container=p["accent"],
        on_tertiary_container=p["accent_foreground"],
        error=p["destructive"],
        on_error=p["destructive_foreground"],
        surface=p["background"],
        on_surface=p["foreground"],
        on_surface_variant=p["muted_foreground"],
        surface_container=p["muted"],
        surface_container_low=p["secondary"],
        surface_container_lowest=p["card"],
        surface_container_high=p["muted"],
        surface_container_highest=p["border"],
        outline=p["border"],
        outline_variant=p["muted"],
    )


def build_light_theme() -> ft.Theme:
    return ft.Theme(
        color_scheme=_build_scheme(LIGHT),
        scaffold_bgcolor=LIGHT["background"],
        card_bgcolor=LIGHT["card"],
        divider_color=LIGHT["border"],
    )


def build_dark_theme() -> ft.Theme:
    return ft.Theme(
        color_scheme=_build_scheme(DARK),
        scaffold_bgcolor=DARK["background"],
        card_bgcolor=DARK["card"],
        divider_color=DARK["border"],
    )


# ──────────────────────────────────────────────
# Helpers para referencia rápida desde la UI
# ──────────────────────────────────────────────
def palette(dark: bool = False) -> dict:
    """Devuelve el diccionario de tokens activo."""
    return DARK if dark else LIGHT
