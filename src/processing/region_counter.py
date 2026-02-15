"""Contador de objetos que cruzan una region (linea o poligono)."""

import numpy as np
from shapely.geometry import LineString, Point

from src.enums import Direction


class RegionCounter:
    """Contador de objetos que cruzan una region (linea o poligono)."""

    def __init__(self, region: list[tuple[int, int]]):
        self.region = region
        self.in_count = 0
        self.out_count = 0
        self.counted_ids: set[int] = set()
        self._crossing_direction: dict[int, Direction] = {}
        self._prev_centroids: dict[int, tuple[float, float]] = {}

        if len(region) == 2:
            self._is_line = True
            self._line_p1 = region[0]
            self._line_p2 = region[1]
            dx = abs(region[1][0] - region[0][0])
            dy = abs(region[1][1] - region[0][1])
            self._line_is_vertical = dy > dx
            self._line = None
            self._convex_hull = None
        else:
            pts = list(region) + [region[0]]
            self._line = LineString(pts)
            self._is_line = False
            self._line_is_vertical = False
            self._convex_hull = self._line.convex_hull

    def get_direction(self, track_id: int) -> Direction:
        """Retorna la direccion de cruce de un track."""
        return self._crossing_direction.get(track_id, Direction.UNKNOWN)

    @staticmethod
    def _segments_intersect(
        p1: tuple, p2: tuple, p3: tuple, p4: tuple
    ) -> bool:
        """Test si segmento (p1->p2) intersecta segmento (p3->p4)."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def update(self, track_ids: list[int], boxes: np.ndarray) -> set[int]:
        """Actualiza conteo. Retorna set de IDs que acaban de cruzar."""
        new_crossings = set()
        for i, tid in enumerate(track_ids):
            if tid in self.counted_ids:
                continue
            cx = (boxes[i, 0] + boxes[i, 2]) / 2
            cy = (boxes[i, 1] + boxes[i, 3]) / 2

            if tid in self._prev_centroids:
                px, py = self._prev_centroids[tid]

                if self._is_line:
                    crossed = self._segments_intersect(
                        (px, py), (cx, cy), self._line_p1, self._line_p2
                    )
                else:
                    movement = LineString([(px, py), (cx, cy)])
                    crossed = self._line.intersects(movement)

                if crossed:
                    self.counted_ids.add(tid)
                    new_crossings.add(tid)
                    if self._is_line:
                        if self._line_is_vertical:
                            if cx > px:
                                self.in_count += 1
                                self._crossing_direction[tid] = Direction.IN
                            else:
                                self.out_count += 1
                                self._crossing_direction[tid] = Direction.OUT
                        else:
                            if cy > py:
                                self.in_count += 1
                                self._crossing_direction[tid] = Direction.IN
                            else:
                                self.out_count += 1
                                self._crossing_direction[tid] = Direction.OUT
                    else:
                        prev_inside = Point(px, py).within(self._convex_hull)
                        if not prev_inside:
                            self.in_count += 1
                            self._crossing_direction[tid] = Direction.IN
                        else:
                            self.out_count += 1
                            self._crossing_direction[tid] = Direction.OUT

            self._prev_centroids[tid] = (cx, cy)

        return new_crossings
