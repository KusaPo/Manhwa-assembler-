"""
Pass 1 — Speech bubble detection and inpainting.

EasyOCR locates text regions; we flood-fill from each text centre into
the surrounding light area (the bubble interior) to get the full bubble
shape, dilate to capture the border, then use TELEA inpainting to
replace it with surrounding art.

For dark or coloured bubbles the flood-fill is skipped and a simple
dilated bounding rectangle is used instead.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("engines.vision.bubbles")

BBox = Tuple[int, int, int, int]  # x, y, w, h

_reader_cache: Dict[Tuple, object] = {}


def _get_reader(langs: Tuple[str, ...], use_gpu: bool):
    key = (langs, use_gpu)
    if key not in _reader_cache:
        try:
            import easyocr  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "easyocr is required for bubble removal. "
                "Run: pip install easyocr"
            )
        logger.info(
            f"  Loading EasyOCR (langs={list(langs)}, gpu={use_gpu})"
            " — first run downloads model weights (~100 MB)"
        )
        _reader_cache[key] = easyocr.Reader(list(langs), gpu=use_gpu)
    return _reader_cache[key]


def detect_text_regions(
    image: np.ndarray,
    langs: List[str],
    use_gpu: bool = False,
    confidence_threshold: float = 0.3,
) -> List[BBox]:
    """Return (x, y, w, h) axis-aligned bboxes for every text region found."""
    reader = _get_reader(tuple(langs), use_gpu)
    results = reader.readtext(image, detail=1, paragraph=False)

    bboxes: List[BBox] = []
    for quad, _text, conf in results:
        if conf < confidence_threshold:
            continue
        xs = [int(p[0]) for p in quad]
        ys = [int(p[1]) for p in quad]
        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(image.shape[1], max(xs))
        y2 = min(image.shape[0], max(ys))
        if x2 > x1 and y2 > y1:
            bboxes.append((x1, y1, x2 - x1, y2 - y1))

    return bboxes


def _build_bubble_mask(
    image: np.ndarray,
    text_regions: List[BBox],
    dilation_px: int,
    white_threshold: int = 210,
) -> np.ndarray:
    """
    For each text region:
    - If the centre pixel is light (bubble interior), flood-fill the
      connected light region to capture the whole bubble shape.
    - Otherwise fall back to an expanded rectangle.
    Then dilate the combined mask to cover bubble outlines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    combined = np.zeros((h, w), dtype=np.uint8)

    light_binary = np.where(gray >= white_threshold, np.uint8(255), np.uint8(0))

    for x, y, bw, bh in text_regions:
        cx = min(max(x + bw // 2, 0), w - 1)
        cy = min(max(y + bh // 2, 0), h - 1)

        if gray[cy, cx] >= white_threshold:
            ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
            cv2.floodFill(
                light_binary.copy(), ff_mask, (cx, cy), 255,
                loDiff=0, upDiff=0, flags=flags,
            )
            region = ff_mask[1:-1, 1:-1]
        else:
            region = np.zeros((h, w), dtype=np.uint8)
            rx1 = max(0, x - dilation_px)
            ry1 = max(0, y - dilation_px)
            rx2 = min(w, x + bw + dilation_px)
            ry2 = min(h, y + bh + dilation_px)
            region[ry1:ry2, rx1:rx2] = 255

        combined = cv2.bitwise_or(combined, region)

    if dilation_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        combined = cv2.dilate(combined, k, iterations=1)

    return combined


def remove_bubbles(
    image: np.ndarray,
    langs: List[str],
    use_gpu: bool = False,
    confidence_threshold: float = 0.3,
    dilation_px: int = 20,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    Full Pass 1: detect text → build bubble mask → inpaint.
    Returns the cleaned image (same shape and dtype as input).
    """
    regions = detect_text_regions(
        image,
        langs=langs,
        use_gpu=use_gpu,
        confidence_threshold=confidence_threshold,
    )
    if not regions:
        logger.debug("    No text regions detected — strip unchanged")
        return image

    logger.info(f"    Detected {len(regions)} text region(s) — inpainting")
    mask = _build_bubble_mask(image, regions, dilation_px=dilation_px)
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
