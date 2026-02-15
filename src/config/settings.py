"""Paths y constantes globales del proyecto."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DET_MODEL_PATH = os.path.join(BASE_DIR, "src", "weights", "pepper_det.onnx")
CLS_MODEL_PATH = os.path.join(BASE_DIR, "src", "weights", "pepper_ripeness_cls_v1.onnx")
TRACKER_CONFIG = os.path.join(BASE_DIR, "botsort.yaml")

DEFAULT_CONFIDENCE = 0.40
CROP_SIZE = 224
DISPLAY_MAX_W = 800
DISPLAY_MAX_H = 600
