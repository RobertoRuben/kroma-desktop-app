import flet as ft
import flet.canvas as cv

ROI_MODE_LINE = "line"
ROI_MODE_POLYGON = "polygon"


class ROICanvas(ft.Column):
    """Canvas interactivo para dibujar ROI (linea o poligono) sobre el primer frame."""

    def __init__(
        self,
        image_base64: str,
        display_width: int,
        display_height: int,
        scale_x: float,
        scale_y: float,
        on_roi_confirmed: callable,
    ):
        super().__init__()
        self.image_base64 = image_base64
        self.display_width = display_width
        self.display_height = display_height
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.on_roi_confirmed = on_roi_confirmed

        self.points: list[tuple[float, float]] = []
        self.is_closed = False
        self.mode = ROI_MODE_LINE

        # Paints
        self._stroke_paint = ft.Paint(
            color=ft.Colors.GREEN_400,
            stroke_width=2,
            style=ft.PaintingStyle.STROKE,
        )
        self._fill_paint = ft.Paint(
            color="#4000E676",
            style=ft.PaintingStyle.FILL,
        )
        self._line_band_paint = ft.Paint(
            color="#4000E676",
            stroke_width=30,
            style=ft.PaintingStyle.STROKE,
            stroke_cap=ft.StrokeCap.ROUND,
        )

        # Canvas
        self.canvas = cv.Canvas(
            shapes=[],
            width=self.display_width,
            height=self.display_height,
        )

        # Mode selector
        self.mode_radio = ft.RadioGroup(
            value=ROI_MODE_LINE,
            on_change=self._on_mode_change,
            content=ft.Row(
                [
                    ft.Radio(value=ROI_MODE_LINE, label="Linea (2 puntos)"),
                    ft.Radio(value=ROI_MODE_POLYGON, label="Poligono (3+ puntos)"),
                ],
                spacing=20,
            ),
        )

        self.points_text = ft.Text(
            "Puntos ROI: (click para agregar vertices)", size=12
        )

        self.btn_clear = ft.OutlinedButton(
            "Limpiar ROI",
            icon=ft.Icons.DELETE_OUTLINE,
            on_click=self._on_clear,
            disabled=True,
        )
        self.btn_close = ft.OutlinedButton(
            "Cerrar Poligono",
            icon=ft.Icons.CHECK_CIRCLE_OUTLINE,
            on_click=self._on_close_polygon,
            visible=False,
            disabled=True,
        )
        self.btn_confirm = ft.Button(
            "Confirmar ROI y Procesar",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_confirm,
            disabled=True,
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.GREEN_700,
        )

        gesture = ft.GestureDetector(
            content=self.canvas,
            on_tap_down=self._on_tap_down,
            mouse_cursor=ft.MouseCursor.PRECISE,
        )

        self.controls = [
            self.mode_radio,
            ft.Text(
                "Click sobre la imagen para definir la zona de interes.",
                size=13,
                italic=True,
                color=ft.Colors.GREY_600,
            ),
            ft.Container(
                content=ft.Stack(
                    [
                        ft.Image(
                            src=self.image_base64,
                            width=self.display_width,
                            height=self.display_height,
                            fit=ft.BoxFit.FILL,
                        ),
                        gesture,
                    ]
                ),
                border=ft.Border.all(2, ft.Colors.BLUE_400),
                border_radius=8,
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
                width=self.display_width,
                height=self.display_height,
            ),
            self.points_text,
            ft.Row(
                [self.btn_clear, self.btn_close, self.btn_confirm],
                spacing=10,
                wrap=True,
            ),
        ]
        self.spacing = 10

    def _on_mode_change(self, e):
        self.mode = e.control.value
        self.points.clear()
        self.is_closed = False
        self.btn_close.visible = self.mode == ROI_MODE_POLYGON
        self._redraw()
        self._update_buttons()
        self.btn_close.update()

    def _snap_point(self, x: float, y: float) -> tuple[float, float]:
        """Ajusta el punto para alinearlo horizontal/vertical si esta cerca (15 grados)."""
        if not self.points:
            return x, y
        prev_x, prev_y = self.points[-1]
        dx = abs(x - prev_x)
        dy = abs(y - prev_y)
        # Si la diferencia en un eje es muy pequena respecto al otro, snap
        if dx + dy == 0:
            return x, y
        if dy < dx * 0.27:  # ~15 grados -> snap horizontal
            return x, prev_y
        if dx < dy * 0.27:  # ~15 grados -> snap vertical
            return prev_x, y
        return x, y

    def _on_tap_down(self, e: ft.TapEvent):
        if self.is_closed:
            return

        x, y = e.local_position.x, e.local_position.y

        # Snap a horizontal/vertical si el angulo es cercano
        if self.points:
            x, y = self._snap_point(x, y)

        if self.mode == ROI_MODE_LINE:
            if len(self.points) >= 2:
                return
            self.points.append((x, y))
            if len(self.points) == 2:
                self.is_closed = True
        else:
            # Polygon mode
            if len(self.points) >= 3:
                first_x, first_y = self.points[0]
                dist = ((x - first_x) ** 2 + (y - first_y) ** 2) ** 0.5
                if dist < 15:
                    self.is_closed = True
                    self._redraw()
                    self._update_buttons()
                    return
            self.points.append((x, y))

        self._redraw()
        self._update_buttons()

    def _on_clear(self, e):
        self.points.clear()
        self.is_closed = False
        self._redraw()
        self._update_buttons()

    def _on_close_polygon(self, e):
        if self.mode == ROI_MODE_POLYGON and len(self.points) >= 3:
            self.is_closed = True
            self._redraw()
            self._update_buttons()

    def _on_confirm(self, e):
        if not self.is_closed:
            return
        min_points = 2 if self.mode == ROI_MODE_LINE else 3
        if len(self.points) >= min_points:
            roi_scaled = self.get_roi_points_scaled()
            self.on_roi_confirmed(roi_scaled)

    def _update_buttons(self):
        has_points = len(self.points) > 0
        min_points = 2 if self.mode == ROI_MODE_LINE else 3
        can_close = (
            self.mode == ROI_MODE_POLYGON
            and len(self.points) >= 3
            and not self.is_closed
        )

        self.btn_clear.disabled = not has_points
        self.btn_close.disabled = not can_close
        self.btn_confirm.disabled = not self.is_closed

        self.btn_clear.update()
        self.btn_close.update()
        self.btn_confirm.update()

        coords_str = ", ".join(f"({x:.0f},{y:.0f})" for x, y in self.points)
        mode_label = "Linea" if self.mode == ROI_MODE_LINE else "Poligono"
        status = " [LISTO]" if self.is_closed else ""
        self.points_text.value = (
            f"{mode_label} ({len(self.points)} pts){status}: {coords_str}"
        )
        self.points_text.update()

    def _redraw(self):
        shapes = []

        if self.mode == ROI_MODE_POLYGON:
            # Fill polygon (even while drawing, from 3+ points)
            if len(self.points) >= 3:
                path_elements = [
                    cv.Path.MoveTo(self.points[0][0], self.points[0][1])
                ]
                for px, py in self.points[1:]:
                    path_elements.append(cv.Path.LineTo(px, py))
                path_elements.append(cv.Path.Close())
                shapes.append(
                    cv.Path(elements=path_elements, paint=self._fill_paint)
                )

            # Draw border lines
            for i in range(len(self.points) - 1):
                shapes.append(
                    cv.Line(
                        self.points[i][0],
                        self.points[i][1],
                        self.points[i + 1][0],
                        self.points[i + 1][1],
                        paint=self._stroke_paint,
                    )
                )

            # Close border
            if self.is_closed and len(self.points) >= 3:
                shapes.append(
                    cv.Line(
                        self.points[-1][0],
                        self.points[-1][1],
                        self.points[0][0],
                        self.points[0][1],
                        paint=self._stroke_paint,
                    )
                )
        else:
            # Line mode: semi-transparent band + thin stroke
            if len(self.points) == 2:
                shapes.append(
                    cv.Line(
                        self.points[0][0],
                        self.points[0][1],
                        self.points[1][0],
                        self.points[1][1],
                        paint=self._line_band_paint,
                    )
                )
                shapes.append(
                    cv.Line(
                        self.points[0][0],
                        self.points[0][1],
                        self.points[1][0],
                        self.points[1][1],
                        paint=self._stroke_paint,
                    )
                )

        self.canvas.shapes = shapes
        self.canvas.update()

    def get_roi_points_scaled(self) -> list[tuple[int, int]]:
        """Retorna puntos ROI en coordenadas originales del video."""
        return [
            (int(x * self.scale_x), int(y * self.scale_y)) for x, y in self.points
        ]
