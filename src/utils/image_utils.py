import base64
import subprocess
import sys

import cv2
import numpy as np


# --- HDR (PQ / ST.2084 & HLG) to SDR tone mapping ---

def detect_hdr_transfer(video_path: str) -> str | None:
    """Detect HDR transfer function. Returns 'pq', 'hlg', or None.

    Tries ffprobe first, then pymediainfo, then frame-level heuristic.
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

    # 2. Try pymediainfo (reliable, no ffprobe dependency)
    detected = _detect_hdr_pymediainfo(video_path)
    if detected is not None:
        return detected

    # 3. Heuristic: analyze first frame pixel distribution
    return _detect_hdr_heuristic(video_path)


def _detect_hdr_pymediainfo(video_path: str) -> str | None:
    """Detect HDR transfer via pymediainfo metadata."""
    try:
        from pymediainfo import MediaInfo

        media_info = MediaInfo.parse(video_path)
        for track in media_info.tracks:
            if track.track_type == "Video":
                transfer = (track.transfer_characteristics or "").lower()
                if "pq" in transfer or "smpte 2084" in transfer or "2084" in transfer:
                    return "pq"
                if "hlg" in transfer or "arib" in transfer or "b67" in transfer:
                    return "hlg"
                return None
    except (ImportError, Exception):
        pass
    return None


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


# ============================================================
# Pre-computed LUTs (built once at import time)
# ============================================================

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


def _build_hlg_inverse_lut() -> np.ndarray:
    """Precompute LUT: HLG uint8 value -> linear scene luminance [0, 1].

    ARIB STD-B67 inverse OETF.
    """
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * np.log(4.0 * a)

    lut = np.zeros(256, dtype=np.float32)
    for i in range(1, 256):
        E = i / 255.0
        if E <= 0.5:
            lut[i] = (E * E) / 3.0
        else:
            lut[i] = (np.exp((E - c) / a) + b) / 12.0
    return lut


def _build_nits_to_sdr_lut(n_entries: int = 65536) -> np.ndarray:
    """Precompute LUT: linear nits -> SDR uint8 (Reinhard + gamma 2.2).

    This replaces the per-pixel Reinhard + np.power which is the main bottleneck.
    """
    max_nits = 10000.0
    max_white_sq = 100.0  # (10.0)^2

    lut = np.zeros(n_entries, dtype=np.uint8)
    for i in range(1, n_entries):
        nits = i * max_nits / (n_entries - 1)
        x = nits / 203.0  # SDR reference white = 203 nits (ITU-R BT.2408)
        mapped = x * (1.0 + x / max_white_sq) / (1.0 + x)
        mapped = min(mapped, 1.0)
        sdr = mapped ** (1.0 / 2.2)
        lut[i] = int(min(sdr * 255.0, 255))
    return lut


_PQ_LUT = _build_pq_inverse_lut()
_HLG_LUT = _build_hlg_inverse_lut()
_NITS_TO_SDR_LUT = _build_nits_to_sdr_lut()
_NITS_LUT_SIZE = len(_NITS_TO_SDR_LUT)
_NITS_LUT_SCALE = np.float32((_NITS_LUT_SIZE - 1) / 10000.0)

# BT.2020 -> BT.709 color matrix (linear RGB)
_M_2020_TO_709 = np.array([
    [ 1.6605, -0.5877, -0.0728],
    [-0.1246,  1.1329, -0.0083],
    [-0.0182, -0.1006,  1.1187],
], dtype=np.float32)


# ============================================================
# Tone mapping functions
# ============================================================

def tone_map_pq_frame_fast(frame: np.ndarray) -> np.ndarray:
    """Convert PQ (ST.2084) / BT.2020 frame to SDR BT.709.

    Uses pre-computed LUTs + BLAS matrix multiply for speed.
    Correct cross-channel BT.2020→BT.709 gamut conversion.

    Input:  BGR uint8 as read by OpenCV from an HDR10 video.
    Output: BGR uint8 suitable for SDR display and encoding.
    """
    h, w = frame.shape[:2]

    # 1. PQ → linear nits via LUT  (BGR → RGB for color matrix)
    linear = _PQ_LUT[frame[:, :, ::-1]]  # (H, W, 3) float32, nits

    # 2. BT.2020 → BT.709 gamut conversion (BLAS matmul, ~11ms for 1080p)
    flat_709 = linear.reshape(-1, 3) @ _M_2020_TO_709.T
    np.maximum(flat_709, 0.0, out=flat_709)

    # 3. Nits → SDR uint8 via LUT (replaces Reinhard + gamma: 96ms → ~33ms)
    indices = np.clip(
        (flat_709 * _NITS_LUT_SCALE).astype(np.int32), 0, _NITS_LUT_SIZE - 1
    )
    sdr_rgb = _NITS_TO_SDR_LUT[indices].reshape(h, w, 3)

    # 4. RGB → BGR
    return sdr_rgb[:, :, ::-1].copy()


def tone_map_hlg_frame_fast(frame: np.ndarray) -> np.ndarray:
    """Convert HLG (ARIB STD-B67) / BT.2020 frame to SDR BT.709.

    Input:  BGR uint8 as read by OpenCV from an HLG video (iPhone, etc).
    Output: BGR uint8 suitable for SDR display and encoding.
    """
    h, w = frame.shape[:2]

    # 1. HLG → linear scene light via LUT  (BGR → RGB)
    linear = _HLG_LUT[frame[:, :, ::-1]]  # (H, W, 3) float32 [0, ~1]

    # 2. OOTF: scene → display (system gamma ≈ 1.2 for 1000-nit display)
    #    L = 0.2627*R + 0.6780*G + 0.0593*B  (BT.2020 luminance)
    lum = 0.2627 * linear[:, :, 0] + 0.6780 * linear[:, :, 1] + 0.0593 * linear[:, :, 2]
    # Apply OOTF: display = scene * (lum ^ (gamma-1))
    ootf_factor = np.power(np.maximum(lum, 1e-10), 0.2)  # gamma=1.2, so exp=0.2
    display = linear * ootf_factor[:, :, np.newaxis]

    # 3. BT.2020 → BT.709 gamut conversion
    flat_709 = display.reshape(-1, 3) @ _M_2020_TO_709.T
    np.maximum(flat_709, 0.0, out=flat_709)
    linear_709 = flat_709.reshape(h, w, 3)

    # 4. Gamma 2.2 (BT.709/sRGB) — HLG values are already in ~[0, 1] range
    np.clip(linear_709, 0.0, 1.0, out=linear_709)
    sdr = np.power(linear_709, np.float32(1.0 / 2.2))

    # 5. Back to BGR uint8
    return np.clip(sdr[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)


def tone_map_hdr_frame(frame: np.ndarray, transfer: str) -> np.ndarray:
    """Unified HDR → SDR tone mapping for any detected transfer function.

    Args:
        frame: BGR uint8 as read by OpenCV.
        transfer: 'pq' or 'hlg'.
    Returns:
        BGR uint8 suitable for SDR display.
    """
    if transfer == "pq":
        return tone_map_pq_frame_fast(frame)
    elif transfer == "hlg":
        return tone_map_hlg_frame_fast(frame)
    return frame


# ============================================================
# Utility functions
# ============================================================

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
