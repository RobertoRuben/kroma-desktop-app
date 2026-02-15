"""Stepper visual horizontal — indica paso actual del flujo."""

import flet as ft

_STEPS = [
    (ft.Icons.VIDEO_FILE, "Cargar Video"),
    (ft.Icons.CROP, "Zona de Interes"),
    (ft.Icons.PLAY_CIRCLE, "Procesamiento"),
    (ft.Icons.ASSESSMENT, "Resultados"),
]


class StepIndicator(ft.Row):
    """Barra horizontal con 4 pasos: completado / actual / pendiente."""

    def __init__(self, palette: dict, current: int = 0):
        super().__init__()
        self._pal = palette
        self._current = current
        self.alignment = ft.MainAxisAlignment.CENTER
        self.vertical_alignment = ft.CrossAxisAlignment.CENTER
        self.spacing = 0
        self._build_controls()

    def set_step(self, step: int) -> None:
        self._current = step
        self._build_controls()
        self.update()

    def set_palette(self, palette: dict) -> None:
        self._pal = palette
        self._build_controls()

    def _build_controls(self) -> None:
        pal = self._pal
        items: list[ft.Control] = []

        for i, (icon, label) in enumerate(_STEPS):
            if i < self._current:
                # Completado
                circle_bg = pal["step_completed"]
                circle_icon = ft.Icons.CHECK
                icon_color = pal["primary_foreground"]
                label_color = pal["foreground"]
                label_weight = None
            elif i == self._current:
                # Actual
                circle_bg = "transparent"
                circle_icon = icon
                icon_color = pal["primary"]
                label_color = pal["primary"]
                label_weight = ft.FontWeight.BOLD
            else:
                # Pendiente
                circle_bg = pal["step_pending"]
                circle_icon = icon
                icon_color = pal["muted_foreground"]
                label_color = pal["muted_foreground"]
                label_weight = None

            border = ft.Border.all(2, pal["primary"] if i <= self._current else pal["muted"])

            circle = ft.Container(
                content=ft.Icon(circle_icon, size=18, color=icon_color),
                width=36,
                height=36,
                border_radius=18,
                bgcolor=circle_bg,
                border=border,
                alignment=ft.Alignment.CENTER,
            )

            step_col = ft.Column(
                [
                    circle,
                    ft.Text(label, size=10, color=label_color, weight=label_weight, text_align=ft.TextAlign.CENTER),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
                width=100,
            )

            items.append(step_col)

            # Linea conectora (no despues del ultimo)
            if i < len(_STEPS) - 1:
                line_color = pal["primary"] if i < self._current else pal["muted"]
                items.append(
                    ft.Container(
                        height=2,
                        width=40,
                        bgcolor=line_color,
                        margin=ft.Margin(0, 0, 0, 20),
                    )
                )

        self.controls = items
