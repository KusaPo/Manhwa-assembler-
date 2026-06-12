"""
Image Processing Module
========================
Loads panel images and applies Ken Burns effects (zoom + pan)
with the manhwa-style BLURRED BACKGROUND FILL technique.

For portrait/tall panels: blurred enlarged version fills the background,
sharp original sits centered on top.

For landscape panels: standard fit with optional blur fill on the sides.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("image_processor")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Ken Burns effect types
EFFECT_TYPES = ["zoom_in", "zoom_out", "pan_left", "pan_right", "static"]
EFFECT_WEIGHTS = [35, 30, 15, 15, 5]  # Heavily favor zoom in/out


def load_images(images_dir: Path) -> List[Path]:
    """Load and sort all image files from the directory."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    
    if not images:
        logger.warning(f"No images found in {images_dir}")
    
    return images


def fit_image_to_canvas(
    image: np.ndarray,
    target_width: int = 1920,
    target_height: int = 1080,
    blur_background: bool = True,
    blur_radius: int = 51,
    blur_dim: float = 0.7,
    foreground_scale: float = 0.95,
) -> np.ndarray:
    """
    Fit image to canvas using the BLURRED BACKGROUND FILL technique.
    
    Method (manhwa-recap style):
      1. Take original image, scale to OVERFILL the canvas (cover mode)
      2. Apply heavy Gaussian blur to that overfilled version
      3. Optionally darken it slightly so foreground pops
      4. Scale original to FIT the canvas (contain mode, preserving aspect ratio)
      5. Center foreground over blurred background
    """
    h, w = image.shape[:2]
    target_aspect = target_width / target_height
    img_aspect = w / h

    if not blur_background:
        return _simple_fit(image, target_width, target_height)

    # === LAYER 1: Blurred background (cover mode) ===
    if img_aspect > target_aspect:
        bg_h = target_height
        bg_w = int(target_height * img_aspect)
    else:
        bg_w = target_width
        bg_h = int(target_width / img_aspect)

    # Scale bg up a bit more so blur doesn't reveal edges
    scale_boost = 1.15
    bg_w = int(bg_w * scale_boost)
    bg_h = int(bg_h * scale_boost)

    background = cv2.resize(image, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)

    # Crop center to canvas size
    x_offset = (bg_w - target_width) // 2
    y_offset = (bg_h - target_height) // 2
    background = background[y_offset:y_offset + target_height, x_offset:x_offset + target_width]

    # Apply Gaussian blur (kernel must be odd)
    if blur_radius % 2 == 0:
        blur_radius += 1
    background = cv2.GaussianBlur(background, (blur_radius, blur_radius), 0)

    # Darken background so foreground pops
    if blur_dim < 1.0:
        background = (background.astype(np.float32) * blur_dim).clip(0, 255).astype(np.uint8)

    # === LAYER 2: Sharp foreground (contain mode) ===
    fg_target_h = int(target_height * foreground_scale)
    fg_target_w = int(target_width * foreground_scale)

    if img_aspect > (fg_target_w / fg_target_h):
        new_w = fg_target_w
        new_h = int(fg_target_w / img_aspect)
    else:
        new_h = fg_target_h
        new_w = int(fg_target_h * img_aspect)

    foreground = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # === Composite: paste foreground centered on background ===
    canvas = background.copy()
    y_paste = (target_height - new_h) // 2
    x_paste = (target_width - new_w) // 2
    canvas[y_paste:y_paste + new_h, x_paste:x_paste + new_w] = foreground

    return canvas


def _simple_fit(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Fallback: black letterbox/pillarbox (no blur)."""
    h, w = image.shape[:2]
    target_aspect = target_width / target_height
    img_aspect = w / h

    if img_aspect > target_aspect:
        new_w = target_width
        new_h = int(target_width / img_aspect)
    else:
        new_h = target_height
        new_w = int(target_height * img_aspect)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def get_ken_burns_frame_function(
    image: np.ndarray,
    duration_seconds: float,
    effect_type: str = "random",
    zoom_intensity: float = 0.15,
):
    """
    Return a function that generates a frame at time t.
    Memory-efficient frame generator for MoviePy's VideoClip(make_frame=...).
    """
    if effect_type == "random":
        effect_type = random.choices(EFFECT_TYPES, weights=EFFECT_WEIGHTS, k=1)[0]

    def make_frame(t: float) -> np.ndarray:
        progress = min(1.0, t / max(0.01, duration_seconds))
        frame_bgr = _generate_frame(image, progress, effect_type, zoom_intensity)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return make_frame, effect_type


def _generate_frame(
    image: np.ndarray,
    progress: float,
    effect_type: str,
    zoom_intensity: float,
) -> np.ndarray:
    """Generate a single frame at progress point 0.0 to 1.0."""
    if effect_type == "static":
        return image.copy()

    elif effect_type == "zoom_in":
        scale = 1.0 + (progress * zoom_intensity)
        return _zoom_image(image, scale, center_offset=(0.0, -0.03))

    elif effect_type == "zoom_out":
        scale = (1.0 + zoom_intensity) - (progress * zoom_intensity)
        return _zoom_image(image, scale, center_offset=(0.0, 0.03))

    elif effect_type == "pan_left":
        scale = 1.0 + (zoom_intensity * 0.5)
        x_offset = 0.05 * (1.0 - 2.0 * progress)
        return _zoom_image(image, scale, center_offset=(x_offset, 0.0))

    elif effect_type == "pan_right":
        scale = 1.0 + (zoom_intensity * 0.5)
        x_offset = -0.05 * (1.0 - 2.0 * progress)
        return _zoom_image(image, scale, center_offset=(x_offset, 0.0))

    else:
        return image.copy()


def _zoom_image(
    image: np.ndarray,
    scale: float,
    center_offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Zoom into an image by scale factor with optional pan offset."""
    h, w = image.shape[:2]
    
    new_w = int(w / scale)
    new_h = int(h / scale)
    
    offset_x = int(center_offset[0] * w)
    offset_y = int(center_offset[1] * h)
    
    center_x = w // 2 + offset_x
    center_y = h // 2 + offset_y
    
    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    
    if x2 - x1 < new_w:
        x1 = max(0, x2 - new_w)
    if y2 - y1 < new_h:
        y1 = max(0, y2 - new_h)
    
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)


def apply_ken_burns_effect(*args, **kwargs):
    """Deprecated - kept for backward compatibility."""
    pass
