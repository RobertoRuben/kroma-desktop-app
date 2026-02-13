import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from shapely.geometry import LineString, Point
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

from src.utils.image_utils import resize_with_padding, tone_map_pq_frame

# Colores BGR por clase de madurez (para anotación en video)
_RIPENESS_COLORS = {
    "brown": (42, 42, 165),
    "green": (0, 180, 0),
    "red": (0, 0, 220),
    "turning": (0, 165, 255),
}


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


class _Detections:
    """Wrapper para detecciones compatible con BOTSORT.update() que espera .conf/.xyxy/.xywh/.cls."""

    def __init__(self, dets: np.ndarray):
        """dets: (N, 6) con [x1, y1, x2, y2, conf, cls]."""
        self._dets = dets

    def __len__(self):
        return len(self._dets)

    def __getitem__(self, idx):
        return _Detections(self._dets[idx] if self._dets.ndim > 1 else self._dets[idx].reshape(-1, 6))

    @property
    def conf(self):
        return self._dets[:, 4] if len(self._dets) > 0 else np.array([])

    @property
    def xyxy(self):
        return self._dets[:, :4] if len(self._dets) > 0 else np.empty((0, 4))

    @property
    def xywh(self):
        if len(self._dets) == 0:
            return np.empty((0, 4))
        xyxy = self._dets[:, :4]
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        return np.stack([cx, cy, w, h], axis=1)

    @property
    def cls(self):
        return self._dets[:, 5] if len(self._dets) > 0 else np.array([])


class OnnxDetector:
    """Detector ONNX end2end (YOLO26) con denormalizacion correcta."""

    def __init__(self, model_path: str, use_gpu: bool, confidence: float = 0.40):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # Suppress warnings
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        inp_shape = self.session.get_inputs()[0].shape
        self.input_size = (inp_shape[2], inp_shape[3])  # (H, W) = (640, 640)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detecta objetos. Retorna array (N, 6) con [x1, y1, x2, y2, conf, cls] en coords del frame."""
        h_orig, w_orig = frame.shape[:2]
        inp_h, inp_w = self.input_size

        # Preprocess: letterbox resize a inp_size
        scale = min(inp_w / w_orig, inp_h / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        pad_x = (inp_w - new_w) // 2
        pad_y = (inp_h - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((inp_h, inp_w, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # BGR -> RGB, HWC -> CHW, normalize
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        # Inferencia
        output = self.session.run(None, {self.input_name: blob})[0]  # (1, 300, 6)
        preds = output[0]  # (300, 6): [x1, y1, x2, y2, conf, cls] normalized

        # Filtrar por confianza
        mask = preds[:, 4] > self.confidence
        dets = preds[mask]

        if len(dets) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Denormalizar: [0,1] -> coordenadas del canvas (640x640)
        dets[:, [0, 2]] *= inp_w
        dets[:, [1, 3]] *= inp_h

        # Quitar letterbox padding y escalar a coordenadas originales
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale

        # Clip a limites del frame
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, w_orig)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, h_orig)

        # Filtrar boxes invalidas (w=0 o h=0)
        valid = (dets[:, 2] - dets[:, 0] > 1) & (dets[:, 3] - dets[:, 1] > 1)
        return dets[valid].astype(np.float32)


class RegionCounter:
    """Contador de objetos que cruzan una region (linea o poligono)."""

    def __init__(self, region: list[tuple[int, int]]):
        self.region = region
        self.in_count = 0
        self.out_count = 0
        self.counted_ids: set[int] = set()
        self._prev_centroids: dict[int, tuple[float, float]] = {}

        # Para linea (2 pts): usar LineString para deteccion de cruce
        if len(region) == 2:
            self._line = LineString(region)
            self._is_line = True
        else:
            # Poligono: crear LineString del perimetro cerrado
            pts = list(region) + [region[0]]
            self._line = LineString(pts)
            self._is_line = False

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
                movement = LineString([(px, py), (cx, cy)])
                if self._line.intersects(movement):
                    self.counted_ids.add(tid)
                    new_crossings.add(tid)
                    # Determinar direccion IN/OUT
                    if self._is_line:
                        # Para linea horizontal: arriba->abajo = IN, abajo->arriba = OUT
                        if cy > py:
                            self.in_count += 1
                        else:
                            self.out_count += 1
                    else:
                        # Para poligono: entrando = IN, saliendo = OUT
                        prev_inside = Point(px, py).within(
                            LineString(list(self.region) + [self.region[0]]).convex_hull
                        )
                        if not prev_inside:
                            self.in_count += 1
                        else:
                            self.out_count += 1

            self._prev_centroids[tid] = (cx, cy)

        return new_crossings


class VideoProcessor:
    """Procesa video con ONNX detector + BoT-SORT tracker + clasificacion de madurez."""

    def __init__(
        self,
        video_path: str,
        model_path: str,
        cls_model_path: str,
        tracker_config: str,
        roi_points: list[tuple[int, int]],
        use_gpu: bool,
        confidence: float = 0.40,
        hdr_transfer: str | None = None,
    ):
        self.video_path = video_path
        self.roi_points = roi_points
        self.hdr_transfer = hdr_transfer

        # Detector ONNX con denormalizacion correcta
        self.detector = OnnxDetector(model_path, use_gpu, confidence)

        # Clasificador de madurez
        self.cls_model = YOLO(cls_model_path, task="classify")

        # Tracker BoT-SORT
        with open(tracker_config, "r") as f:
            tracker_args = yaml.safe_load(f)
        tracker_args["gmc_method"] = tracker_args.get("gmc_method", "sparseOptFlow")
        self.tracker = BOTSORT(args=type("Args", (), tracker_args)(), frame_rate=30)

        # Region counter
        self.region_counter = RegionCounter(roi_points)

        self.crops: list[np.ndarray] = []
        self.crop_metadata: list[dict[str, Any]] = []
        self._track_ripeness: dict[int, tuple[str, float]] = {}
        self._ripeness_counts: dict[str, int] = {
            "green": 0, "red": 0, "brown": 0, "turning": 0,
        }

        # Overlay ROI
        self._roi_overlay: np.ndarray | None = None
        self._roi_mask: np.ndarray | None = None

    def process(
        self, progress_callback: Callable[[int, int, int, int], None] | None = None
    ) -> dict[str, Any]:
        """Procesa el video completo."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")

        rotation = detect_video_rotation(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("El video parece estar vacio o danado.")
        first_frame = apply_rotation(first_frame, rotation)
        h, w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._build_roi_overlay(h, w)

        output_path = self._get_output_path()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"No se pudo crear video de salida: {output_path}")

        frame_num = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                frame = apply_rotation(frame, rotation)

                if self.hdr_transfer == "pq":
                    frame = tone_map_pq_frame(frame)

                frame_clean = frame.copy()

                # Detectar
                dets = self.detector.detect(frame)

                # Tracking
                track_ids, tracked_boxes = self._run_tracker(dets, frame)

                # Contar cruces
                new_crossings = set()
                if len(track_ids) > 0:
                    new_crossings = self.region_counter.update(track_ids, tracked_boxes)

                # Anotar frame
                annotated = self._annotate_frame(
                    frame, tracked_boxes, track_ids, frame_clean
                )
                annotated = self._draw_roi_on_frame(annotated)

                # Extraer crops de nuevos cruces
                if new_crossings:
                    self._extract_crops(
                        frame_clean, tracked_boxes, track_ids,
                        new_crossings, frame_num,
                    )

                writer.write(annotated)

                if frame_num % 10 == 0 and progress_callback:
                    progress_callback(
                        frame_num, total_frames,
                        self.region_counter.in_count,
                        self.region_counter.out_count,
                    )
        finally:
            cap.release()
            writer.release()

        self._deduplicate_crops()

        return {
            "output_path": output_path,
            "crops": self.crops,
            "crop_metadata": self.crop_metadata,
            "in_count": self.region_counter.in_count,
            "out_count": self.region_counter.out_count,
            "total": self.region_counter.in_count + self.region_counter.out_count,
            "ripeness_counts": dict(self._ripeness_counts),
            "frames_processed": frame_num,
            "total_frames": total_frames,
        }

    def _run_tracker(
        self, dets: np.ndarray, frame: np.ndarray
    ) -> tuple[list[int], np.ndarray]:
        """Ejecuta BoT-SORT sobre las detecciones. Retorna (track_ids, boxes xyxy)."""
        det_obj = _Detections(dets)
        tracks = self.tracker.update(det_obj, frame)

        # tracks: (N, 8) = [x1, y1, x2, y2, track_id, score, cls, idx]
        if len(tracks) == 0:
            return [], np.empty((0, 4))

        track_ids = [int(t[4]) for t in tracks]
        boxes_xyxy = tracks[:, :4]
        return track_ids, boxes_xyxy

    def _classify_crop(self, crop: np.ndarray) -> tuple[str, float]:
        """Clasifica un crop con el modelo de madurez."""
        results = self.cls_model(crop, verbose=False)
        probs = results[0].probs
        label = results[0].names[probs.top1]
        conf = float(probs.top1conf)
        return label, conf

    def _annotate_frame(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_ids: list[int],
        frame_clean: np.ndarray,
    ) -> np.ndarray:
        """Dibuja bboxes, track IDs y madurez sobre el frame."""
        annotated = frame.copy()
        h, w = frame_clean.shape[:2]

        for i, tid in enumerate(track_ids):
            x1, y1, x2, y2 = map(int, boxes[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Clasificar (con cache)
            if tid not in self._track_ripeness:
                crop = frame_clean[y1:y2, x1:x2]
                if crop.size > 0:
                    label, conf = self._classify_crop(crop)
                    self._track_ripeness[tid] = (label, conf)

            if tid in self._track_ripeness:
                label, conf = self._track_ripeness[tid]
                color = _RIPENESS_COLORS.get(label, (200, 200, 200))
            else:
                label, conf, color = "?", 0.0, (200, 200, 200)

            # Bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Etiqueta: ID + madurez
            text = f"#{tid} {label} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, th_line = 0.45, 1
            (tw, th), baseline = cv2.getTextSize(text, font, fs, th_line)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + tw + 6, ty + baseline), color, -1)
            cv2.putText(annotated, text, (x1 + 3, ty - 2), font, fs, (255, 255, 255), th_line, cv2.LINE_AA)

        # Leyenda unificada (esquina superior derecha)
        self._draw_ripeness_legend(annotated)

        return annotated

    def _draw_ripeness_legend(self, frame: np.ndarray) -> None:
        """Dibuja leyenda unificada con IN/OUT y conteo por madurez."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, thickness = 0.5, 1
        line_h = 24
        pad = 10
        total = sum(self._ripeness_counts.values())

        lines: list[tuple[str, int, tuple[int, int, int], bool]] = [
            ("IN", self.region_counter.in_count, (230, 180, 0), False),
            ("OUT", self.region_counter.out_count, (0, 140, 255), False),
            ("Total", total, (255, 255, 255), False),
        ]
        for label in ("green", "red", "turning", "brown"):
            count = self._ripeness_counts[label]
            color_bgr = _RIPENESS_COLORS[label]
            lines.append((label.capitalize(), count, color_bgr, True))

        # Calcular ancho maximo del texto (con espacio para circulito)
        max_tw = 0
        for label_text, count, _, has_dot in lines:
            text = f"{label_text}: {count}"
            (tw, _), _ = cv2.getTextSize(text, font, fs, thickness)
            max_tw = max(max_tw, tw + (16 if has_dot else 0))

        # Separador entre IN/OUT/Total y las clases de madurez
        separator_after = 2  # despues de "Total" (indice 2)
        separator_h = 10

        box_w = max_tw + pad * 2 + 4
        box_h = len(lines) * line_h + pad * 2 + separator_h
        h_frame, w_frame = frame.shape[:2]
        x0 = w_frame - box_w - 10
        y0 = 10

        # Fondo oscuro semi-transparente
        overlay = frame[y0 : y0 + box_h, x0 : x0 + box_w].copy()
        dark = np.full_like(overlay, (30, 30, 30))
        blended = cv2.addWeighted(overlay, 0.3, dark, 0.7, 0)
        frame[y0 : y0 + box_h, x0 : x0 + box_w] = blended
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (80, 80, 80), 1, cv2.LINE_AA)

        # Dibujar cada linea
        y_offset = 0
        for i, (label_text, count, color, has_dot) in enumerate(lines):
            if i == separator_after + 1:
                # Linea separadora
                sep_y = y0 + pad + i * line_h + y_offset
                cv2.line(frame, (x0 + pad, sep_y), (x0 + box_w - pad, sep_y), (100, 100, 100), 1, cv2.LINE_AA)
                y_offset += separator_h

            ty = y0 + pad + (i + 1) * line_h - 6 + y_offset
            tx = x0 + pad
            if has_dot:
                cv2.circle(frame, (tx + 5, ty - 4), 5, color, -1, cv2.LINE_AA)
                tx += 16
            text = f"{label_text}: {count}"
            cv2.putText(frame, text, (tx, ty), font, fs, color if not has_dot else (255, 255, 255), thickness, cv2.LINE_AA)

    def _extract_crops(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_ids: list[int],
        new_ids: set[int],
        frame_num: int,
    ) -> None:
        """Extrae crops 224x224 del frame limpio."""
        h, w = frame.shape[:2]
        for i, tid in enumerate(track_ids):
            if tid not in new_ids:
                continue
            x1, y1, x2, y2 = map(int, boxes[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            crop_padded = resize_with_padding(crop, 224, 224)

            if tid in self._track_ripeness:
                label, conf = self._track_ripeness[tid]
            else:
                label, conf = self._classify_crop(crop)
                self._track_ripeness[tid] = (label, conf)

            self.crops.append(crop_padded)
            self.crop_metadata.append({
                "track_id": tid,
                "frame": frame_num,
                "bbox": (x1, y1, x2, y2),
                "ripeness": label,
                "ripeness_conf": conf,
            })
            if label in self._ripeness_counts:
                self._ripeness_counts[label] += 1

    # --- ROI overlay (same as before) ---

    def _build_roi_overlay(self, h: int, w: int) -> None:
        pts = np.array(self.roi_points, dtype=np.int32)
        self._roi_mask = np.zeros((h, w), dtype=np.uint8)
        if len(self.roi_points) == 2:
            cv2.line(self._roi_mask, self.roi_points[0], self.roi_points[1], 255, thickness=30)
        else:
            cv2.fillPoly(self._roi_mask, [pts], 255)
        self._roi_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        self._roi_overlay[self._roi_mask > 0] = (0, 230, 118)

    def _draw_roi_on_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._roi_mask is None:
            return frame
        result = frame.copy()
        mask = self._roi_mask > 0
        result[mask] = cv2.addWeighted(
            frame[mask].reshape(-1, 3), 0.75,
            self._roi_overlay[mask].reshape(-1, 3), 0.25, 0,
        ).reshape(-1, 3)
        pts = np.array(self.roi_points, dtype=np.int32)
        if len(self.roi_points) == 2:
            cv2.line(result, self.roi_points[0], self.roi_points[1], (0, 230, 118), 2, cv2.LINE_AA)
        else:
            cv2.polylines(result, [pts], True, (0, 230, 118), 2, cv2.LINE_AA)
        return result

    @staticmethod
    def _bbox_iou(b1: tuple, b2: tuple) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter)

    def _deduplicate_crops(self) -> None:
        """Elimina crops duplicados del mismo track_id (seguridad extra)."""
        if len(self.crops) <= 1:
            return
        seen_ids: set[int] = set()
        keep = []
        for i, meta in enumerate(self.crop_metadata):
            tid = meta.get("track_id")
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            keep.append(i)
        self.crops = [self.crops[i] for i in keep]
        self.crop_metadata = [self.crop_metadata[i] for i in keep]

    def _get_output_path(self) -> str:
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        video_name = Path(self.video_path).stem
        return os.path.join(video_dir, f"{video_name}_counted.mp4")
