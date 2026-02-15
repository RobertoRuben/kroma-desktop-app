"""Procesador de video: deteccion + tracking + clasificacion + conteo."""

import os
import queue
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml
from ultralytics.trackers.bot_sort import BOTSORT

from src.config import CROP_SIZE
from src.enums import Direction, RipenessClass
from src.schemas import CropInfo, ProcessingResult, RipnessCounts
from src.processing.detections import Detections
from src.processing.onnx_classifier import OnnxClassifier
from src.processing.onnx_detector import OnnxDetector
from src.processing.region_counter import RegionCounter
from src.services.video_annotator import VideoAnnotator
from src.services.video_loader import VideoLoader
from src.utils.image_utils import resize_with_padding, tone_map_hdr_frame


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
        output_dir: str | None = None,
    ):
        self.video_path = video_path
        self.roi_points = roi_points
        self.hdr_transfer = hdr_transfer
        self.output_dir = output_dir

        self.detector = OnnxDetector(model_path, use_gpu, confidence)
        self.classifier = OnnxClassifier(cls_model_path, use_gpu)

        with open(tracker_config, "r") as f:
            tracker_args = yaml.safe_load(f)
        tracker_args["gmc_method"] = tracker_args.get("gmc_method", "sparseOptFlow")
        self.tracker = BOTSORT(args=type("Args", (), tracker_args)(), frame_rate=30)

        self.region_counter = RegionCounter(roi_points)
        self.annotator = VideoAnnotator(roi_points)

        self._crops: list[np.ndarray] = []
        self._crop_infos: list[CropInfo] = []
        self._track_ripeness: dict[int, tuple[RipenessClass, float]] = {}
        self._ripeness_counts = RipnessCounts.empty()
        self._in_ripeness = RipnessCounts.empty()
        self._out_ripeness = RipnessCounts.empty()

    def process(
        self, progress_callback: Callable[[int, int, int, int], None] | None = None
    ) -> ProcessingResult:
        """Procesa el video completo con I/O threading para mayor velocidad."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")

        rotation = VideoLoader.detect_rotation(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("El video parece estar vacio o danado.")
        first_frame = VideoLoader.apply_rotation(first_frame, rotation)
        h, w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.annotator.build_roi_overlay(h, w)

        output_path = self._get_output_path()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"No se pudo crear video de salida: {output_path}")

        # --- Producer-Consumer I/O threading ---
        read_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
        write_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
        reader_error: list[Exception] = []

        def reader_thread():
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        read_q.put(None)
                        return
                    read_q.put(frame)
            except Exception as ex:
                reader_error.append(ex)
                read_q.put(None)

        def writer_thread():
            while True:
                frame = write_q.get()
                if frame is None:
                    return
                writer.write(frame)

        t_reader = threading.Thread(target=reader_thread, daemon=True)
        t_writer = threading.Thread(target=writer_thread, daemon=True)
        t_reader.start()
        t_writer.start()

        frame_num = 0

        try:
            while True:
                frame = read_q.get()
                if frame is None:
                    break

                frame_num += 1
                frame = VideoLoader.apply_rotation(frame, rotation)

                if self.hdr_transfer:
                    frame = tone_map_hdr_frame(frame, self.hdr_transfer)

                frame_clean = frame.copy()

                dets = self.detector.detect(frame)
                track_ids, tracked_boxes = self._run_tracker(dets, frame)

                if len(track_ids) > 0:
                    self._classify_tracks(frame_clean, tracked_boxes, track_ids)
                    new_crossings = self.region_counter.update(track_ids, tracked_boxes)
                    if new_crossings:
                        self._extract_crops(
                            frame_clean, tracked_boxes, track_ids, new_crossings, frame_num,
                        )

                # Track ripeness as str dict for annotator compatibility
                track_rip_str = {
                    tid: (cls.value, conf) for tid, (cls, conf) in self._track_ripeness.items()
                }
                self.annotator.annotate_frame(frame, tracked_boxes, track_ids, track_rip_str)
                self.annotator.draw_legend(
                    frame, self.region_counter,
                    self._ripeness_counts, self._in_ripeness, self._out_ripeness,
                )
                self.annotator.draw_roi(frame)

                write_q.put(frame)

                if frame_num % 10 == 0 and progress_callback:
                    progress_callback(
                        frame_num, total_frames,
                        self.region_counter.in_count, self.region_counter.out_count,
                    )
        finally:
            write_q.put(None)
            t_reader.join(timeout=5)
            t_writer.join(timeout=5)
            cap.release()
            writer.release()

        self._deduplicate_crops()

        return ProcessingResult(
            output_path=output_path,
            in_count=self.region_counter.in_count,
            out_count=self.region_counter.out_count,
            ripeness_counts=self._ripeness_counts,
            in_ripeness=self._in_ripeness,
            out_ripeness=self._out_ripeness,
            frames_processed=frame_num,
            total_frames=total_frames,
            crops=self._crop_infos,
            crop_images=self._crops,
        )

    def _run_tracker(
        self, dets: np.ndarray, frame: np.ndarray
    ) -> tuple[list[int], np.ndarray]:
        """Ejecuta BoT-SORT sobre las detecciones."""
        det_obj = Detections(dets)
        tracks = self.tracker.update(det_obj, frame)

        if len(tracks) == 0:
            return [], np.empty((0, 4))

        track_ids = [int(t[4]) for t in tracks]
        boxes_xyxy = tracks[:, :4]
        return track_ids, boxes_xyxy

    def _extract_crops(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_ids: list[int],
        new_ids: set[int],
        frame_num: int,
    ) -> None:
        """Extrae crops del frame limpio y clasifica."""
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
            crop_padded = resize_with_padding(crop, CROP_SIZE, CROP_SIZE)

            if tid in self._track_ripeness:
                ripeness_cls, conf = self._track_ripeness[tid]
            else:
                ripeness_cls, conf = self.classifier.classify(crop)
                self._track_ripeness[tid] = (ripeness_cls, conf)

            direction = self.region_counter.get_direction(tid)

            crop_info = CropInfo(
                track_id=tid,
                frame_number=frame_num,
                bbox=(x1, y1, x2, y2),
                ripeness=ripeness_cls,
                ripeness_conf=conf,
                direction=direction,
            )

            self._crops.append(crop_padded)
            self._crop_infos.append(crop_info)
            self._ripeness_counts = self._ripeness_counts.increment(ripeness_cls)

            if direction == Direction.IN:
                self._in_ripeness = self._in_ripeness.increment(ripeness_cls)
            elif direction == Direction.OUT:
                self._out_ripeness = self._out_ripeness.increment(ripeness_cls)

    def _deduplicate_crops(self) -> None:
        """Elimina crops duplicados del mismo track_id."""
        if len(self._crops) <= 1:
            return
        seen_ids: set[int] = set()
        keep = []
        for i, info in enumerate(self._crop_infos):
            if info.track_id in seen_ids:
                continue
            seen_ids.add(info.track_id)
            keep.append(i)
        self._crops = [self._crops[i] for i in keep]
        self._crop_infos = [self._crop_infos[i] for i in keep]

    def _get_output_path(self) -> str:
        out_dir = self.output_dir or os.path.dirname(os.path.abspath(self.video_path))
        video_name = Path(self.video_path).stem
        return os.path.join(out_dir, f"{video_name}_counted.mp4")
