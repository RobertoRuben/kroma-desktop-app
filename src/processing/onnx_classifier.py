"""Clasificador ONNX generico para pimientos (madurez, calidad, etc.)."""

from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort


class OnnxClassifier:
    """Clasificador ONNX directo, sin overhead de Ultralytics."""

    def __init__(self, model_path: str, use_gpu: bool, index_map: dict[int, Enum]):
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
        self.input_size = (inp_shape[2], inp_shape[3])
        self._blob = np.zeros(
            (1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32
        )
        self._index_map = index_map

    def classify(self, crop: np.ndarray) -> tuple[Enum, float]:
        """Clasifica un crop. Retorna (clase, confidence).

        Replica el preprocessing de YOLO classify: Resize(shortest=224) + CenterCrop(224).
        """
        target = self.input_size[0]
        ch, cw = crop.shape[:2]

        if ch < cw:
            new_h = target
            new_w = int(cw * target / ch)
        else:
            new_w = target
            new_h = int(ch * target / cw)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        y_off = (new_h - target) // 2
        x_off = (new_w - target) // 2
        cropped = resized[y_off : y_off + target, x_off : x_off + target]

        np.divide(
            cropped[:, :, ::-1].transpose(2, 0, 1),
            255.0,
            out=self._blob[0],
            casting="unsafe",
        )
        output = self.session.run(None, {self.input_name: self._blob})[0]
        probs = output[0]
        top_idx = int(np.argmax(probs))
        return self._index_map[top_idx], float(probs[top_idx])
