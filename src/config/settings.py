"""Paths y constantes globales del proyecto."""

import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DET_MODEL_PATH = os.path.join(BASE_DIR, "src", "weights", "pepper_det.onnx")
CLS_MODEL_PATH = os.path.join(BASE_DIR, "src", "weights", "pepper_ripeness_cls_v1.onnx")
TRACKER_CONFIG = os.path.join(BASE_DIR, "botsort.yaml")

DEFAULT_CONFIDENCE = 0.40
CROP_SIZE = 224
DISPLAY_MAX_W = 800
DISPLAY_MAX_H = 600

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://kroma-api.automaginex-ai.lat/api/v1")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
