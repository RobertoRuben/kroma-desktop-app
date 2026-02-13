import base64
import subprocess
import sys

import cv2
import numpy as np


# --- HDR (PQ / ST.2084) to SDR tone mapping ---

def detect_hdr_transfer(video_path: str) -> str | None:
    """Detect HDR transfer function. Returns 'pq', 'hlg', or None.

    Tries ffprobe first; falls back to frame-level heuristic analysis.
    """
    # 1. Try ffprobe (most reliable)
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=color_transfer",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=10,
            creationflags=flags,
        )
        transfer = result.stdout.strip().lower()
        if "smpte2084" in transfer:
            return "pq"
        if "arib-std-b67" in transfer:
            return "hlg"
        # ffprobe ran successfully but no HDR detected
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # 2. Heuristic: analyze first frame pixel distribution
    #    PQ-encoded frames read as uint8 have: max < ~235, mean > 125,
    #    and nearly zero pixels in the top ~10% of the range.
    return _detect_hdr_heuristic(video_path)


def _detect_hdr_heuristic(video_path: str) -> str | None:
    """Heuristic PQ detection from frame statistics."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    max_val = frame.max()
    mean_val = frame.mean()
    # Count pixels in the top 12% of range (230-255)
    top_fraction = np.count_nonzero(frame > 230) / frame.size

    # PQ content: max typically < 235, mean > 125, almost no pixels above 230
    if max_val < 240 and mean_val > 125 and top_fraction < 0.005:
        return "pq"
    return None


def _build_pq_inverse_lut() -> np.ndarray:
    """Precompute LUT: PQ uint8 value -> linear luminance in nits [0, 10000]."""
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    lut = np.zeros(256, dtype=np.float32)
    for i in range(1, 256):
        V = i / 255.0
        Vp = V ** (1.0 / m2)
        num = max(Vp - c1, 0.0)
        den = c2 - c3 * Vp
        if den > 1e-10:
            lut[i] = (num / den) ** (1.0 / m1) * 10000.0
    return lut


_PQ_LUT = _build_pq_inverse_lut()

# BT.2020 -> BT.709 color matrix (linear RGB)
_M_2020_TO_709 = np.array([
    [ 1.6605, -0.5877, -0.0728],
    [-0.1246,  1.1329, -0.0083],
    [-0.0182, -0.1006,  1.1187],
], dtype=np.float32)


def tone_map_pq_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a PQ (ST.2084) / BT.2020 frame to SDR BT.709.

    Input:  BGR uint8 as read by OpenCV from an HDR10 video.
    Output: BGR uint8 suitable for SDR display and encoding.
    """
    h, w = frame.shape[:2]

    # 1. PQ -> linear nits via LUT  (BGR -> RGB for color matrix)
    rgb = frame[:, :, ::-1]
    linear = _PQ_LUT[rgb]  # (H, W, 3) float32, nits

    # 2. BT.2020 -> BT.709 gamut conversion (linear space)
    flat = linear.reshape(-1, 3)
    flat_709 = flat @ _M_2020_TO_709.T
    linear_709 = np.maximum(flat_709.reshape(h, w, 3), 0.0)

    # 3. Tone mapping — Reinhard extended (Lmax ≈ 2000 nits)
    #    Normalize so SDR reference white (203 nits, ITU-R BT.2408) = 1.0
    x = linear_709 / 203.0
    max_white = 10.0  # 10× reference = ~2030 nits before full white
    mapped = x * (1.0 + x / (max_white * max_white)) / (1.0 + x)
    mapped = np.clip(mapped, 0.0, 1.0)

    # 4. Gamma 2.2 (BT.709 / sRGB approximation)
    sdr = np.power(mapped, 1.0 / 2.2)

    # 5. Back to BGR uint8
    bgr_out = np.clip(sdr[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    return bgr_out


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Convierte un frame OpenCV (BGR) a data URI base64 JPEG para Flet Image.src."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def resize_with_padding(
    image: np.ndarray, target_w: int = 224, target_h: int = 224
) -> np.ndarray:
    """Resize manteniendo aspect ratio y padding negro centrado a target_w x target_h."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def resize_frame_for_display(
    frame: np.ndarray, max_width: int = 800, max_height: int = 600
) -> tuple[np.ndarray, float, float]:
    """Resize frame para display en UI. Retorna (frame_resized, scale_x, scale_y).

    scale_x/scale_y convierten coordenadas de display a coordenadas originales:
        original_x = display_x * scale_x
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale < 1.0:
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = frame.copy()

    scale_x = w / new_w
    scale_y = h / new_h
    return resized, scale_x, scale_y
