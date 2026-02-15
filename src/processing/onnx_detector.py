"""Detector ONNX end2end (YOLO) con denormalizacion correcta y buffers pre-alocados."""

import cv2
import numpy as np
import onnxruntime as ort


class OnnxDetector:
    """Detector ONNX end2end (YOLO26) con denormalizacion correcta y buffers pre-alocados."""

    def __init__(self, model_path: str, use_gpu: bool, confidence: float = 0.40):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        inp_shape = self.session.get_inputs()[0].shape
        self.input_size = (inp_shape[2], inp_shape[3])  # (H, W) = (640, 640)
        self.confidence = confidence

        # Pre-alocar buffers reutilizables
        inp_h, inp_w = self.input_size
        self._canvas = np.full((inp_h, inp_w, 3), 114, dtype=np.uint8)
        self._blob = np.zeros((1, 3, inp_h, inp_w), dtype=np.float32)
        self._cached_dims: tuple[int, int] | None = None
        self._scale = 0.0
        self._new_w = 0
        self._new_h = 0
        self._pad_x = 0
        self._pad_y = 0

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detecta objetos. Retorna array (N, 6) con [x1, y1, x2, y2, conf, cls] en coords del frame."""
        h_orig, w_orig = frame.shape[:2]
        inp_h, inp_w = self.input_size

        dims = (h_orig, w_orig)
        if dims != self._cached_dims:
            self._cached_dims = dims
            self._scale = min(inp_w / w_orig, inp_h / h_orig)
            self._new_w = int(w_orig * self._scale)
            self._new_h = int(h_orig * self._scale)
            self._pad_x = (inp_w - self._new_w) // 2
            self._pad_y = (inp_h - self._new_h) // 2

        scale = self._scale
        new_w, new_h = self._new_w, self._new_h
        pad_x, pad_y = self._pad_x, self._pad_y

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self._canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        np.divide(
            self._canvas[:, :, ::-1].transpose(2, 0, 1),
            255.0,
            out=self._blob[0],
            casting="unsafe",
        )

        output = self.session.run(None, {self.input_name: self._blob})[0]
        preds = output[0]

        mask = preds[:, 4] > self.confidence
        dets = preds[mask].copy()

        if len(dets) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Denormalizar: [0,1] -> coordenadas del canvas
        dets[:, [0, 2]] *= inp_w
        dets[:, [1, 3]] *= inp_h

        # Quitar letterbox padding y escalar a coordenadas originales
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale

        # Clip a limites del frame
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, w_orig)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, h_orig)

        # Filtrar boxes invalidas
        valid = (dets[:, 2] - dets[:, 0] > 1) & (dets[:, 3] - dets[:, 1] > 1)
        return dets[valid].astype(np.float32)
