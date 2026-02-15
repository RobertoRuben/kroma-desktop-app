"""Servicio de carga de video: apertura, rotacion, HDR, primer frame."""

import cv2
import numpy as np

from src.schemas import VideoInfo
from src.utils.image_utils import detect_hdr_transfer, tone_map_hdr_frame


class VideoLoader:
    """Carga y prepara un video para procesamiento o display."""

    @staticmethod
    def detect_rotation(video_path: str) -> int:
        """Detecta rotacion del metadata del video (Samsung, iPhone, etc)."""
        cap = cv2.VideoCapture(video_path)
        rotation = 0
        if cap.isOpened():
            meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            if meta in (90, 180, 270):
                rotation = int(meta)
        cap.release()
        return rotation

    @staticmethod
    def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
        """Aplica rotacion al frame."""
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @classmethod
    def load(cls, video_path: str) -> VideoInfo:
        """Carga un video y retorna su informacion completa.

        Raises:
            RuntimeError: Si el video no se puede abrir o leer.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        rotation = cls.detect_rotation(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("No se pudo leer el primer frame del video.")

        frame = cls.apply_rotation(frame, rotation)

        hdr_transfer = detect_hdr_transfer(video_path)
        if hdr_transfer:
            frame = tone_map_hdr_frame(frame, hdr_transfer)

        h, w = frame.shape[:2]

        return VideoInfo(
            path=video_path,
            rotation=rotation,
            hdr_transfer=hdr_transfer,
            width=w,
            height=h,
            fps=fps,
            total_frames=total_frames,
            first_frame=frame,
        )
