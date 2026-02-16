"""Tema personalizado Pepper Counter — paleta verde agrícola con modo claro y oscuro."""

import flet as ft

# ──────────────────────────────────────────────
# Paleta de colores — tokens semánticos
# ──────────────────────────────────────────────

# Modo claro — HSL spec convertido a hex
LIGHT = {
    "background": "#F7F8F7",  # HSL(120, 2%, 97%)
    "foreground": "#222A27",  # HSL(153, 10%, 15%)
    "card": "#FFFFFF",  # HSL(0, 0%, 100%)
    "card_foreground": "#222A27",  # HSL(153, 10%, 15%)
    "primary": "#427B61",  # HSL(153, 30%, 37%)
    "primary_foreground": "#FFFFFF",  # HSL(0, 0%, 100%)
    "primary_hover": "#4B9B77",  # HSL(153, 35%, 45%)
    "primary_light": "#DFECE6",  # HSL(153, 25%, 90%)
    "secondary": "#DDDFDD",  # HSL(120, 2%, 87%)
    "secondary_foreground": "#222A27",  # HSL(153, 10%, 15%)
    "muted": "#D6DCDB",  # HSL(169, 8%, 85%)
    "muted_foreground": "#6A7C79",  # HSL(169, 8%, 45%)
    "accent": "#FFFFFF",  # Blanco (reemplaza amarillo)
    "accent_foreground": "#427B61",  # Primary green para contraste
    "destructive": "#B63A3C",  # HSL(359, 52%, 47%)
    "destructive_foreground": "#FFFFFF",  # HSL(0, 0%, 100%)
    "border": "#D6DCDB",  # HSL(169, 8%, 85%)
    "input": "#D6DCDB",  # HSL(169, 8%, 85%)
    "ring": "#427B61",  # HSL(153, 30%, 37%)
    "sidebar_bg": "#427B61",  # HSL(153, 30%, 37%)
    "sidebar_fg": "#FFFFFF",  # HSL(0, 0%, 100%)
    "sidebar_accent": "#4B9B77",  # HSL(153, 35%, 45%)
    "sidebar_border": "#36634F",  # HSL(153, 30%, 30%)
}

# Modo oscuro — colores invertidos, misma paleta tematica
DARK = {
    "background": "#101412",  # HSL(153, 10%, 7%)
    "foreground": "#F2F3F2",  # HSL(120, 2%, 95%)
    "card": "#171C1A",  # HSL(153, 10%, 10%)
    "card_foreground": "#F2F3F2",  # HSL(120, 2%, 95%)
    "primary": "#427B61",  # HSL(153, 30%, 37%) — mismo
    "primary_foreground": "#FFFFFF",
    "primary_hover": "#4B9B77",  # HSL(153, 35%, 45%) — mismo
    "primary_light": "#305041",  # HSL(153, 25%, 25%)
    "secondary": "#222A27",  # HSL(153, 10%, 15%)
    "secondary_foreground": "#F2F3F2",
    "muted": "#2F3736",  # HSL(169, 8%, 20%)
    "muted_foreground": "#91A19E",  # HSL(169, 8%, 60%)
    "accent": "#FFFFFF",  # Blanco (reemplaza amarillo)
    "accent_foreground": "#427B61",  # Primary green para contraste
    "destructive": "#B63A3C",  # HSL(359, 52%, 47%) — mismo
    "destructive_foreground": "#FFFFFF",
    "border": "#2F3736",  # HSL(169, 8%, 20%)
    "input": "#2F3736",  # HSL(169, 8%, 20%)
    "ring": "#427B61",  # HSL(153, 30%, 37%) — mismo
    "sidebar_bg": "#427B61",  # HSL(153, 30%, 37%) — mismo
    "sidebar_fg": "#FFFFFF",
    "sidebar_accent": "#4B9B77",  # HSL(153, 35%, 45%) — mismo
    "sidebar_border": "#36634F",  # HSL(153, 30%, 30%)
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
