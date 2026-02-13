import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from ultralytics import solutions

from src.utils.image_utils import resize_with_padding, tone_map_pq_frame


def detect_video_rotation(video_path: str) -> int:
    """Detecta rotacion del metadata del video (Samsung, iPhone, etc)."""
    cap = cv2.VideoCapture(video_path)
    rotation = 0
    if cap.isOpened():
        meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        if meta in (90, 180, 270):
            rotation = int(meta)
    cap.release()
    return rotation


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Aplica rotacion al frame."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class VideoProcessor:
    """Procesa video con YOLO ObjectCounter y extrae crops de objetos que cruzan la ROI."""

    def __init__(
        self,
        video_path: str,
        model_path: str,
        tracker_config: str,
        roi_points: list[tuple[int, int]],
        use_gpu: bool,
        confidence: float = 0.40,
        hdr_transfer: str | None = None,
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.roi_points = roi_points
        self.device = "0" if use_gpu else "cpu"
        self.confidence = confidence
        self.hdr_transfer = hdr_transfer

        self.crops: list[np.ndarray] = []
        self.crop_metadata: list[dict[str, Any]] = []
        self._previously_counted_ids: set[int] = set()

        # Overlay de ROI pre-calculado (se inicializa en process())
        self._roi_overlay: np.ndarray | None = None
        self._roi_mask: np.ndarray | None = None

    def process(
        self, progress_callback: Callable[[int, int, int, int], None] | None = None
    ) -> dict[str, Any]:
        """Procesa el video (sincronico, ejecutar en thread separado)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")

        # Detectar rotacion automaticamente
        rotation = detect_video_rotation(self.video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Leer primer frame para obtener dimensiones reales (post-rotacion)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("El video parece estar vacio o danado.")
        first_frame = apply_rotation(first_frame, rotation)
        h, w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Pre-calcular overlay de ROI sombreada
        self._build_roi_overlay(h, w)

        counter = solutions.ObjectCounter(
            model=self.model_path,
            region=self.roi_points,
            show=False,
            classes=[0],
            conf=self.confidence,
            device=self.device,
            tracker=self.tracker_config,
            line_width=2,
            show_in=True,
            show_out=True,
        )

        output_path = self._get_output_path()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(
                f"No se pudo inicializar el escritor de video: {output_path}"
            )

        frame_num = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                # Aplicar rotacion si es necesario
                frame = apply_rotation(frame, rotation)

                # Tone-map HDR -> SDR si es necesario
                if self.hdr_transfer == "pq":
                    frame = tone_map_pq_frame(frame)

                # Guardar copia LIMPIA antes de pasar al counter
                # (counter modifica el frame in-place con anotaciones)
                frame_clean = frame.copy()

                results = counter(frame)
                # Reemplazar ROI default (rectangulo purpura) con sombreado verde
                annotated_frame = self._draw_roi_on_frame(results.plot_im)

                current_counted_ids = set(counter.counted_ids)
                new_crossings = current_counted_ids - self._previously_counted_ids

                if new_crossings:
                    # Extraer crops del frame LIMPIO (sin anotaciones)
                    self._extract_crops(
                        frame_clean, counter, new_crossings, frame_num
                    )

                self._previously_counted_ids = current_counted_ids

                writer.write(annotated_frame)

                if frame_num % 10 == 0 and progress_callback:
                    progress_callback(
                        frame_num, total_frames, counter.in_count, counter.out_count
                    )
        finally:
            cap.release()
            writer.release()

        # Eliminar crops duplicados (misma bbox, distinto track ID)
        self._deduplicate_crops()

        return {
            "output_path": output_path,
            "crops": self.crops,
            "crop_metadata": self.crop_metadata,
            "in_count": counter.in_count,
            "out_count": counter.out_count,
            "total": counter.in_count + counter.out_count,
            "frames_processed": frame_num,
            "total_frames": total_frames,
        }

    def _build_roi_overlay(self, h: int, w: int) -> None:
        """Pre-calcula overlay y mascara para dibujar la ROI sombreada."""
        pts = np.array(self.roi_points, dtype=np.int32)
        self._roi_mask = np.zeros((h, w), dtype=np.uint8)

        if len(self.roi_points) == 2:
            # Linea: dibujar banda gruesa
            cv2.line(self._roi_mask, self.roi_points[0], self.roi_points[1],
                     255, thickness=30)
        else:
            # Poligono: rellenar area
            cv2.fillPoly(self._roi_mask, [pts], 255)

        # Overlay verde semi-transparente
        self._roi_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        self._roi_overlay[self._roi_mask > 0] = (0, 230, 118)  # BGR verde

    def _draw_roi_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja la ROI sombreada sobre el frame (reemplaza el rectangulo default)."""
        if self._roi_mask is None:
            return frame
        alpha = 0.25
        result = frame.copy()
        mask = self._roi_mask > 0
        result[mask] = cv2.addWeighted(
            frame[mask].reshape(-1, 3), 1 - alpha,
            self._roi_overlay[mask].reshape(-1, 3), alpha,
            0,
        ).reshape(-1, 3)
        # Borde del ROI
        pts = np.array(self.roi_points, dtype=np.int32)
        if len(self.roi_points) == 2:
            cv2.line(result, self.roi_points[0], self.roi_points[1],
                     (0, 230, 118), 2, cv2.LINE_AA)
        else:
            cv2.polylines(result, [pts], isClosed=True,
                          color=(0, 230, 118), thickness=2, lineType=cv2.LINE_AA)
        return result

    @staticmethod
    def _bbox_iou(b1: tuple, b2: tuple) -> float:
        """Calcula IoU entre dos bboxes (x1,y1,x2,y2)."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (area1 + area2 - inter)

    def _deduplicate_crops(self) -> None:
        """Elimina crops duplicados por solapamiento de bbox (IoU > 0.5)."""
        if len(self.crops) <= 1:
            return
        keep = [True] * len(self.crops)
        for i in range(len(self.crop_metadata)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(self.crop_metadata)):
                if not keep[j]:
                    continue
                iou = self._bbox_iou(
                    self.crop_metadata[i]["bbox"],
                    self.crop_metadata[j]["bbox"],
                )
                if iou > 0.5:
                    keep[j] = False
        self.crops = [c for c, k in zip(self.crops, keep) if k]
        self.crop_metadata = [m for m, k in zip(self.crop_metadata, keep) if k]

    def _extract_crops(
        self,
        frame: np.ndarray,
        counter: Any,
        new_ids: set[int],
        frame_num: int,
    ) -> None:
        """Extrae crops 224x224 del frame limpio (sin anotaciones)."""
        if not hasattr(counter, "boxes") or counter.boxes is None or len(counter.boxes) == 0:
            return
        if not hasattr(counter, "track_ids") or not counter.track_ids:
            return

        boxes = (
            counter.boxes.cpu().numpy()
            if hasattr(counter.boxes, "cpu")
            else np.array(counter.boxes)
        )
        track_ids = counter.track_ids

        h, w = frame.shape[:2]

        for i, track_id in enumerate(track_ids):
            if int(track_id) in new_ids and i < len(boxes):
                x1, y1, x2, y2 = map(int, boxes[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2].copy()
                crop_padded = resize_with_padding(crop, 224, 224)

                self.crops.append(crop_padded)
                self.crop_metadata.append(
                    {
                        "track_id": int(track_id),
                        "frame": frame_num,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

    def _get_output_path(self) -> str:
        """Genera ruta del video de salida."""
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        video_name = Path(self.video_path).stem
        return os.path.join(video_dir, f"{video_name}_counted.mp4")
