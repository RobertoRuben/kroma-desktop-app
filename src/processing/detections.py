"""Wrapper de detecciones compatible con BOTSORT.update()."""

import numpy as np


class Detections:
    """Wrapper para detecciones compatible con BOTSORT.update() que espera .conf/.xyxy/.xywh/.cls."""

    def __init__(self, dets: np.ndarray):
        """dets: (N, 6) con [x1, y1, x2, y2, conf, cls]."""
        self._dets = dets

    def __len__(self):
        return len(self._dets)

    def __getitem__(self, idx):
        return Detections(
            self._dets[idx] if self._dets.ndim > 1 else self._dets[idx].reshape(-1, 6)
        )

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
