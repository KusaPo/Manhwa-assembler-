"""
Vision Engine — vertical strip slicing and blur-fill panel export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from core.config import RecapConfig, ProjectPaths
from core.models import PanelAsset
from core.utils import IMAGE_EXTENSIONS

logger = logging.getLogger("engines.vision")

EFFECT_TYPES = ["zoom_in", "zoom_out", "pan_left", "pan_right", "static"]


def fit_image_to_canvas(
    image: np.ndarray,
    target_width: int = 1920,
    target_height: int = 1080,
    blur_background: bool = True,
    blur_radius: int = 51,
    blur_dim: float = 0.7,
    foreground_scale: float = 0.95,
) -> np.ndarray:
    h, w = image.shape[:2]
    target_aspect = target_width / target_height
    img_aspect = w / h

    if not blur_background:
        return _simple_fit(image, target_width, target_height)

    if img_aspect > target_aspect:
        bg_h = target_height
        bg_w = int(target_height * img_aspect)
    else:
        bg_w = target_width
        bg_h = int(target_width / img_aspect)

    scale_boost = 1.15
    bg_w = int(bg_w * scale_boost)
    bg_h = int(bg_h * scale_boost)
    background = cv2.resize(image, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    x_offset = (bg_w - target_width) // 2
    y_offset = (bg_h - target_height) // 2
    background = background[y_offset:y_offset + target_height, x_offset:x_offset + target_width]

    if blur_radius % 2 == 0:
        blur_radius += 1
    background = cv2.GaussianBlur(background, (blur_radius, blur_radius), 0)
    if blur_dim < 1.0:
        background = (background.astype(np.float32) * blur_dim).clip(0, 255).astype(np.uint8)

    fg_target_h = int(target_height * foreground_scale)
    fg_target_w = int(target_width * foreground_scale)
    if img_aspect > (fg_target_w / fg_target_h):
        new_w = fg_target_w
        new_h = int(fg_target_w / img_aspect)
    else:
        new_h = fg_target_h
        new_w = int(fg_target_h * img_aspect)

    foreground = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
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


def detect_panel_boundaries(
    image: np.ndarray,
    threshold: int = 240,
    min_panel_height: int = 80,
    max_gutter_height: int = 40,
) -> List[Tuple[int, int]]:
    """
    Pass 2 panel slicer.

    Two separator signals are OR-ed:
    - Bright rows : row mean > threshold  (white gutter)
    - Dark rows   : ≥85 % of pixels below value 25  (black border line)

    Wide clusters (> max_gutter_height) are only treated as gutters if
    ≥70 % of their pixels are white — catching blank transition areas and
    mixed bubble+white-band separators, while blocking speech bubbles that
    sit entirely on dark coloured art backgrounds.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    row_means = gray.mean(axis=1)
    dark_frac = (gray < 25).sum(axis=1) / w
    is_separator = (row_means > threshold) | (dark_frac > 0.85)

    boundaries = [0]
    in_sep = False
    sep_start = 0

    for y in range(h):
        if is_separator[y] and not in_sep:
            in_sep = True
            sep_start = y
        elif not is_separator[y] and in_sep:
            in_sep = False
            sep_height = y - sep_start
            sep_mid = (sep_start + y) // 2

            if sep_height <= max_gutter_height:
                # Thin separator — always a gutter
                is_gutter = True
            else:
                # Wide separator: only a gutter if rows are near-pure-white.
                # Blank space / narration boxes are ≥90 % white across full width.
                # Speech bubbles on coloured art backgrounds have much lower
                # white coverage even when they pass the row-mean threshold.
                white_frac = float((gray[sep_start:y] > threshold).mean())
                is_gutter = white_frac > 0.70

            if is_gutter and sep_mid - boundaries[-1] >= min_panel_height:
                boundaries.append(sep_mid)
    boundaries.append(h)

    panels: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        y1, y2 = boundaries[i], boundaries[i + 1]
        if y2 - y1 >= min_panel_height:
            panels.append((y1, y2))

    return panels if len(panels) > 1 else [(0, h)]


def get_ken_burns_frame_function(
    image: np.ndarray,
    duration_seconds: float,
    effect_type: str = "random",
    zoom_intensity: float = 0.15,
):
    import random

    weights = [35, 30, 15, 15, 5]
    if effect_type == "random":
        effect_type = random.choices(EFFECT_TYPES, weights=weights, k=1)[0]

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
    if effect_type == "static":
        return image.copy()
    if effect_type == "zoom_in":
        scale = 1.0 + (progress * zoom_intensity)
        return _zoom_image(image, scale, center_offset=(0.0, -0.03))
    if effect_type == "zoom_out":
        scale = (1.0 + zoom_intensity) - (progress * zoom_intensity)
        return _zoom_image(image, scale, center_offset=(0.0, 0.03))
    if effect_type == "pan_left":
        scale = 1.0 + (zoom_intensity * 0.5)
        x_offset = 0.05 * (1.0 - 2.0 * progress)
        return _zoom_image(image, scale, center_offset=(x_offset, 0.0))
    if effect_type == "pan_right":
        scale = 1.0 + (zoom_intensity * 0.5)
        x_offset = -0.05 * (1.0 - 2.0 * progress)
        return _zoom_image(image, scale, center_offset=(x_offset, 0.0))
    return image.copy()


def _zoom_image(
    image: np.ndarray,
    scale: float,
    center_offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
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


def is_junk_panel(
    crop: np.ndarray,
    min_content_ratio: float = 0.05,
    white_threshold: int = 240,
) -> bool:
    """
    Return True if the crop should be discarded. Three checks:

    1. Too short  (<80 px) — thin slivers and bubble-bottom cuts.
    2. Too wide relative to height (>3×) — horizontal text-only title cards
       like "ASSASSIN," or "THE LEGENDARY"; webtoon panels are always
       taller than wide so a landscape ratio means a bad cut.
    3. Mostly blank — standalone speech-bubble panels with little art.
    """
    h, w = crop.shape[:2]
    if h < 80:
        return True
    if w / h > 3.0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    non_white = int((gray < white_threshold).sum())
    return (non_white / gray.size) < min_content_ratio


def crop_to_content(
    image: np.ndarray,
    white_threshold: int = 240,
    min_content_frac: float = 0.10,
    min_bubble_width_frac: float = 0.15,
    min_bubble_height: int = 15,
    margin_px: int = 4,
) -> np.ndarray:
    """
    Two-step crop to remove cross-panel speech bubbles and white margins.

    Step 1 — edge-bubble removal:
        Find large white blobs that touch the top or bottom edge of the
        crop. These are always speech bubbles that bled across the panel
        boundary cut. They are removed entirely.

    Step 2 — row tightening:
        Within the remaining image, drop rows where fewer than
        min_content_frac of pixels are non-white. This catches title
        boxes, partial borders, and blank margins left after Step 1.

    Returns the original image unchanged if the result would be < 30 px tall.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Step 1: remove edge-touching white blobs (speech bubbles) ---
    _, binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)

    top_cut = 0
    bottom_cut = h
    min_blob_w = int(w * min_bubble_width_frac)
    edge_tol = 15  # pixels — handles panels with black border before the bubble starts

    for i in range(1, num_labels):
        blob_w = stats[i, cv2.CC_STAT_WIDTH]
        blob_h = stats[i, cv2.CC_STAT_HEIGHT]
        if blob_w < min_blob_w or blob_h < min_bubble_height:
            continue
        y_top = stats[i, cv2.CC_STAT_TOP]
        y_bot = y_top + blob_h
        if y_top <= edge_tol:
            top_cut = max(top_cut, y_bot)
        if y_bot >= h - edge_tol:
            bottom_cut = min(bottom_cut, y_top)

    if top_cut >= bottom_cut or (bottom_cut - top_cut) < 30:
        return image

    # --- Step 2: tighten to content rows within the surviving region ---
    sub_gray = gray[top_cut:bottom_cut, :]
    row_has_content = (sub_gray < white_threshold).mean(axis=1) >= min_content_frac
    content_rows = np.where(row_has_content)[0]

    if len(content_rows) == 0:
        return image

    r1 = top_cut + max(0, int(content_rows[0]) - margin_px)
    r2 = top_cut + min(bottom_cut - top_cut, int(content_rows[-1]) + 1 + margin_px)

    if (r2 - r1) < 30:
        return image

    return image[r1:r2, :]


class ImageProcessor:
    def __init__(self, config: RecapConfig, paths: ProjectPaths) -> None:
        self.config = config
        self.paths = paths

    def process_strips(self) -> List[PanelAsset]:
        strips_dir = self.paths.raw_strips_dir
        if not strips_dir.exists():
            raise FileNotFoundError(f"Raw strips directory not found: {strips_dir}")

        strip_files = sorted(
            p for p in strips_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not strip_files:
            raise FileNotFoundError(f"No images in {strips_dir}")

        self.paths.panels_dir.mkdir(parents=True, exist_ok=True)
        for old in self.paths.panels_dir.glob("*-panel.webp"):
            old.unlink()

        panels: List[PanelAsset] = []
        order = 1

        for strip_path in strip_files:
            image = cv2.imread(str(strip_path))
            if image is None:
                logger.warning(f"  Could not read {strip_path.name}, skipping")
                continue

            # Pass 1: slice into panels
            bounds = detect_panel_boundaries(
                image,
                threshold=self.config.strip_slice_threshold,
                max_gutter_height=self.config.strip_max_gutter_height,
            )
            logger.info(f"  [Pass 1] {strip_path.name}: {len(bounds)} panel(s)")

            skipped = 0
            for y1, y2 in bounds:
                crop = image[y1:y2, :]

                # Pass 2: tight-crop to art content (trims white bubble margins)
                if self.config.content_crop:
                    crop = crop_to_content(crop)

                # Discard standalone bubble panels or near-blank slivers
                if is_junk_panel(crop, min_content_ratio=self.config.panel_min_content_ratio):
                    skipped += 1
                    logger.debug(f"    Skipping blank/junk panel slice (y={y1}-{y2})")
                    continue

                fitted = fit_image_to_canvas(
                    crop,
                    target_width=self.config.video_width,
                    target_height=self.config.video_height,
                    blur_background=self.config.blur_background,
                    blur_radius=self.config.blur_radius,
                    blur_dim=self.config.blur_dim,
                    foreground_scale=self.config.foreground_scale,
                )
                out_name = f"{order:03d}-panel.webp"
                out_path = self.paths.panels_dir / out_name
                cv2.imwrite(str(out_path), fitted)
                panels.append(
                    PanelAsset(index=order, path=out_path, source_strip=strip_path.name)
                )
                order += 1
            if skipped:
                logger.info(f"    Discarded {skipped} junk panel(s) from {strip_path.name}")

        logger.info(f"  Exported {len(panels)} panels to {self.paths.panels_dir}")
        return panels

    def load_panel_paths(self) -> List[Path]:
        if not self.paths.panels_dir.exists():
            return []
        return sorted(
            p for p in self.paths.panels_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
