"""Servicio de anotacion de video: bboxes, leyenda, overlay ROI."""

import cv2
import numpy as np

from src.config import RIPENESS_BGR
from src.enums import RipenessClass
from src.schemas import RipnessCounts
from src.processing.region_counter import RegionCounter


class VideoAnnotator:
    """Dibuja anotaciones sobre frames de video."""

    def __init__(self, roi_points: list[tuple[int, int]]):
        self._roi_points = roi_points
        self._roi_overlay: np.ndarray | None = None
        self._roi_mask: np.ndarray | None = None
        self._legend_cache: tuple[np.ndarray, int, int, int, int] | None = None
        self._legend_counts_hash: tuple | None = None

    def build_roi_overlay(self, h: int, w: int) -> None:
        """Pre-computa el overlay ROI para aplicar rapido sobre cada frame."""
        pts = np.array(self._roi_points, dtype=np.int32)
        self._roi_mask = np.zeros((h, w), dtype=np.uint8)
        if len(self._roi_points) == 2:
            cv2.line(
                self._roi_mask,
                self._roi_points[0],
                self._roi_points[1],
                255,
                thickness=30,
            )
        else:
            cv2.fillPoly(self._roi_mask, [pts], 255)
        self._roi_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        self._roi_overlay[self._roi_mask > 0] = (0, 230, 118)

    def draw_roi(self, frame: np.ndarray) -> None:
        """Dibuja ROI overlay in-place sobre el frame."""
        if self._roi_mask is None:
            return
        mask = self._roi_mask > 0
        frame[mask] = cv2.addWeighted(
            frame[mask].reshape(-1, 3),
            0.75,
            self._roi_overlay[mask].reshape(-1, 3),
            0.25,
            0,
        ).reshape(-1, 3)
        pts = np.array(self._roi_points, dtype=np.int32)
        if len(self._roi_points) == 2:
            cv2.line(
                frame,
                self._roi_points[0],
                self._roi_points[1],
                (0, 230, 118),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.polylines(frame, [pts], True, (0, 230, 118), 2, cv2.LINE_AA)

    def annotate_frame(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_ids: list[int],
        track_ripeness: dict[int, tuple[str, float]],
    ) -> None:
        """Dibuja bboxes, track IDs y madurez sobre el frame (in-place)."""
        h, w = frame.shape[:2]

        for i, tid in enumerate(track_ids):
            x1, y1, x2, y2 = map(int, boxes[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            if tid in track_ripeness:
                label, conf = track_ripeness[tid]
                color = RIPENESS_BGR.get(label, (200, 200, 200))
            else:
                label, conf, color = "?", 0.0, (200, 200, 200)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"#{tid} {label} {conf:.0%}" if label != "?" else f"#{tid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, th_line = 0.45, 1
            (tw, th), baseline = cv2.getTextSize(text, font, fs, th_line)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(
                frame, (x1, ty - th - 4), (x1 + tw + 6, ty + baseline), color, -1
            )
            cv2.putText(
                frame, text, (x1 + 3, ty - 2), font, fs, (255, 255, 255), th_line, cv2.LINE_AA,
            )

    def draw_legend(
        self,
        frame: np.ndarray,
        region_counter: RegionCounter,
        ripeness_counts: RipnessCounts,
        in_ripeness: RipnessCounts,
        out_ripeness: RipnessCounts,
    ) -> None:
        """Dibuja leyenda con IN/OUT desglosado por madurez. Usa cache."""
        counts_hash = (
            region_counter.in_count,
            region_counter.out_count,
            tuple(in_ripeness.model_dump().values()),
            tuple(out_ripeness.model_dump().values()),
        )
        if counts_hash == self._legend_counts_hash and self._legend_cache is not None:
            legend_img, y0, x0, bh, bw = self._legend_cache
            roi = frame[y0 : y0 + bh, x0 : x0 + bw]
            cv2.addWeighted(roi, 0.3, legend_img, 0.7, 0, dst=roi)
            cv2.rectangle(
                frame, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), 1, cv2.LINE_AA
            )
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, thickness = 0.45, 1
        line_h = 22
        pad = 10

        lines: list[tuple[str, tuple[int, int, int], int, bool, tuple[int, int, int] | None]] = []

        lines.append(
            (f"IN: {region_counter.in_count}", (230, 180, 0), 0, False, None)
        )
        for cls in RipenessClass:
            count = in_ripeness.get(cls)
            if count > 0:
                color_bgr = RIPENESS_BGR[cls.value]
                lines.append(
                    (f"{cls.value.capitalize()}: {count}", (220, 220, 220), 12, True, color_bgr)
                )

        lines.append(("", (0, 0, 0), 0, False, None))

        lines.append(
            (f"OUT: {region_counter.out_count}", (0, 140, 255), 0, False, None)
        )
        for cls in RipenessClass:
            count = out_ripeness.get(cls)
            if count > 0:
                color_bgr = RIPENESS_BGR[cls.value]
                lines.append(
                    (f"{cls.value.capitalize()}: {count}", (220, 220, 220), 12, True, color_bgr)
                )

        lines.append(("", (0, 0, 0), 0, False, None))

        total = ripeness_counts.total()
        lines.append((f"Total: {total}", (255, 255, 255), 0, False, None))
        for cls in RipenessClass:
            count = ripeness_counts.get(cls)
            if count > 0:
                color_bgr = RIPENESS_BGR[cls.value]
                lines.append(
                    (f"{cls.value.capitalize()}: {count}", (220, 220, 220), 12, True, color_bgr)
                )

        max_tw = 0
        for text, _, indent, has_dot, _ in lines:
            if not text:
                continue
            (tw, _), _ = cv2.getTextSize(text, font, fs, thickness)
            max_tw = max(max_tw, tw + indent + (14 if has_dot else 0))

        separator_h = 8
        num_separators = sum(1 for t, _, _, _, _ in lines if not t)
        num_content = len(lines) - num_separators
        box_w = max_tw + pad * 2 + 8
        box_h = num_content * line_h + num_separators * separator_h + pad * 2
        h_frame, w_frame = frame.shape[:2]
        x0 = w_frame - box_w - 10
        y0 = 10

        x0 = max(0, x0)
        box_w = min(box_w, w_frame - x0)
        box_h = min(box_h, h_frame - y0)

        legend_img = np.full((box_h, box_w, 3), 30, dtype=np.uint8)
        cv2.rectangle(legend_img, (0, 0), (box_w - 1, box_h - 1), (80, 80, 80), 1, cv2.LINE_AA)

        y_cursor = pad
        for text, color, indent, has_dot, dot_color in lines:
            if not text:
                sep_y = y_cursor + separator_h // 2
                cv2.line(legend_img, (pad, sep_y), (box_w - pad, sep_y), (100, 100, 100), 1, cv2.LINE_AA)
                y_cursor += separator_h
                continue

            ty = y_cursor + line_h - 6
            tx = pad + indent
            if has_dot and dot_color:
                cv2.circle(legend_img, (tx + 5, ty - 4), 5, dot_color, -1, cv2.LINE_AA)
                tx += 14
            cv2.putText(legend_img, text, (tx, ty), font, fs, color, thickness, cv2.LINE_AA)
            y_cursor += line_h

        self._legend_cache = (legend_img, y0, x0, box_h, box_w)
        self._legend_counts_hash = counts_hash

        roi = frame[y0 : y0 + box_h, x0 : x0 + box_w]
        cv2.addWeighted(roi, 0.3, legend_img, 0.7, 0, dst=roi)
        cv2.rectangle(
            frame, (x0, y0), (x0 + box_w, y0 + box_h), (80, 80, 80), 1, cv2.LINE_AA
        )
