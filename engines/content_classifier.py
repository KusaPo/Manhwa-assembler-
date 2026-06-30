"""
Content-aware panel classifier — v2.

Suggests Ken Burns motions by analyzing each panel image for:
  - Content bounds (the non-blurred center vertical strip)
  - Skin/character tone clustering (where are the characters?)
  - Edge density per region (action vs calm)
  - Spatial distribution of content (centered face vs spread group)

Categories mapped to motions in ffmpeg_renderer.MOTIONS / video_assembler effects:
  face_closeup    -> zoom_in_center  (slow zoom intensifies emotion)
  group_scene     -> pan_down        (lets viewer scan across characters)
  action_impact   -> zoom_in_slow    (builds intensity to the punch)
  wide_establish  -> drift_diagonal  (gentle drift, scene-setting)
  detail_narrow   -> zoom_in_center  (focus on detail)
  static_quiet    -> static          (let the beat breathe)

Optional: if lbpcascade_animeface.xml is present, uses it for face detection.
Download (one-time, ~750KB MIT):
  wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("content_classifier")

CATEGORY_TO_MOTION = {
    "face_closeup":   "zoom_in_center",
    "group_scene":    "pan_down",
    "action_impact":  "zoom_in_slow",
    "wide_establish": "drift_diagonal",
    "detail_narrow":  "zoom_in_center",
    "static_quiet":   "static",
}

# Manhwa skin tones: pale pink to tan in HSV. H 0-25 + 160-180 (red wraparound)
SKIN_HSV_RANGES = [
    (np.array([0, 30, 100]),   np.array([25, 180, 255])),
    (np.array([160, 30, 100]), np.array([180, 180, 255])),
]


# ---------------------------------------------------------------------------
# Image analysis primitives
# ---------------------------------------------------------------------------

def detect_content_bounds(img: np.ndarray) -> Tuple[int, int]:
    """Find horizontal bounds of the non-blurred content strip via edge projection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    col_sums = edges.sum(axis=0)
    if col_sums.max() == 0:
        return 0, img.shape[1]
    threshold = col_sums.max() * 0.3
    content_cols = np.where(col_sums > threshold)[0]
    if len(content_cols) == 0:
        return 0, img.shape[1]
    return int(content_cols.min()), int(content_cols.max())


def skin_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of skin-tone pixels with morphological cleanup."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in SKIN_HSV_RANGES:
        mask |= cv2.inRange(hsv, low, high)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_skin_blobs(
    mask: np.ndarray, min_area_frac: float = 0.005
) -> List[Tuple[int, int, int, int, int]]:
    """Find connected skin-tone regions. Returns [(x, y, w, h, area), ...]."""
    h, w = mask.shape
    min_area = int(min_area_frac * h * w)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area >= min_area:
            blobs.append((int(x), int(y), int(bw), int(bh), int(area)))
    return blobs


def edge_density(img: np.ndarray, content_bounds: Tuple[int, int]) -> float:
    """Edge density inside the content strip. High = busy/action."""
    x0, x1 = content_bounds
    if x1 - x0 < 10:
        return 0.0
    content = img[:, x0:x1]
    gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean() / 255.0)


def detect_anime_faces(
    img: np.ndarray, cascade: Optional[cv2.CascadeClassifier]
) -> List[Tuple[int, int, int, int]]:
    """Anime face detection — only runs if cascade is loaded."""
    if cascade is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return [tuple(map(int, f)) for f in faces]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_panel(
    img: np.ndarray,
    anime_cascade: Optional[cv2.CascadeClassifier] = None,
) -> Tuple[str, Dict, float]:
    """Return (category, signals_dict, confidence)."""
    h, w = img.shape[:2]
    cb = detect_content_bounds(img)
    content_w = cb[1] - cb[0]
    content_aspect = content_w / h if h > 0 else 0

    # Restrict skin analysis to content strip to avoid blurred sides inflating counts
    content_img = img[:, cb[0]:cb[1]] if content_w > 50 else img
    content_area = content_img.shape[0] * content_img.shape[1]

    smask = skin_mask(content_img)
    skin_frac = float(smask.mean() / 255.0)
    blobs = find_skin_blobs(smask)
    largest_blob_frac = (max(b[4] for b in blobs) / content_area) if blobs else 0.0
    n_blobs = len(blobs)

    largest_blob_y_frac = 0.5
    if blobs:
        largest = max(blobs, key=lambda b: b[4])
        cy = largest[1] + largest[3] / 2
        largest_blob_y_frac = cy / content_img.shape[0]

    faces = detect_anime_faces(img, anime_cascade)
    n_faces = len(faces)
    largest_face_frac = 0.0
    if faces:
        largest = max(faces, key=lambda f: f[2] * f[3])
        largest_face_frac = (largest[2] * largest[3]) / (h * w)

    ed = edge_density(img, cb)

    signals = {
        "skin_frac":             round(skin_frac, 4),
        "largest_skin_blob_frac": round(largest_blob_frac, 4),
        "largest_blob_y_frac":   round(largest_blob_y_frac, 3),
        "n_skin_blobs":          n_blobs,
        "n_faces":               n_faces,
        "largest_face_frac":     round(largest_face_frac, 4),
        "content_aspect":        round(content_aspect, 3),
        "edge_density":          round(ed, 4),
    }

    # Priority-ordered classification rules
    if n_faces == 1 and largest_face_frac > 0.03:
        return "face_closeup", signals, min(1.0, 0.9 + largest_face_frac * 2)
    if n_faces >= 2:
        return "group_scene", signals, min(1.0, 0.8 + 0.05 * n_faces)

    if largest_blob_frac >= 0.25:
        return "face_closeup", signals, min(0.9, 0.7 + largest_blob_frac * 0.5)

    if ed > 0.12:
        return "action_impact", signals, min(1.0, 0.55 + (ed - 0.12) * 3)

    # Tuning: raise from 4 to 5 to reduce false group_scene on face panels
    if n_blobs >= 5:
        return "group_scene", signals, 0.65

    if largest_blob_frac >= 0.10 and n_blobs <= 3:
        return "face_closeup", signals, 0.65

    if n_blobs >= 2 and skin_frac > 0.08:
        return "group_scene", signals, 0.55

    if content_aspect < 0.30 and ed < 0.08:
        return "detail_narrow", signals, 0.6

    if ed < 0.05 and skin_frac < 0.02:
        return "static_quiet", signals, 0.65

    return "wide_establish", signals, 0.5


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def classify_all_panels(
    panels_dir: Path,
    output_json: Path,
    cascade_path: Optional[Path] = None,
) -> Dict:
    """Run classifier on every panel and write motion_map.json."""
    anime_cascade = None
    if cascade_path and cascade_path.exists():
        anime_cascade = cv2.CascadeClassifier(str(cascade_path))
        if anime_cascade.empty():
            logger.warning(f"  Failed to load cascade: {cascade_path}")
            anime_cascade = None
        else:
            logger.info(f"  Using anime face cascade: {cascade_path.name}")
    else:
        logger.info("  No anime cascade — using heuristics only")

    panel_files = sorted(panels_dir.glob("*-panel.webp")) + \
                  sorted(panels_dir.glob("*-panel.png"))
    if not panel_files:
        raise FileNotFoundError(f"No *-panel.{{webp,png}} files in {panels_dir}")

    motion_map = {}
    category_counts: Dict[str, int] = {}

    for pf in panel_files:
        try:
            panel_num = int(pf.stem.split("-")[0])
        except (ValueError, IndexError):
            continue

        img = cv2.imread(str(pf))
        if img is None:
            continue

        category, signals, confidence = classify_panel(img, anime_cascade)
        motion = CATEGORY_TO_MOTION[category]

        motion_map[str(panel_num)] = {
            "motion":     motion,
            "category":   category,
            "confidence": round(confidence, 2),
            "signals":    signals,
        }
        category_counts[category] = category_counts.get(category, 0) + 1

    total = len(motion_map)
    print(f"\nClassified {total} panels:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {cat:18s} -> {CATEGORY_TO_MOTION[cat]:18s} : {count:3d} ({pct:.1f}%)")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(motion_map, f, indent=2)
    print(f"\nMotion map written: {output_json}")
    print("Edit this file to override any panel's motion.")

    return motion_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Content-aware panel classifier v2")
    p.add_argument("--panels",  type=Path, required=True)
    p.add_argument("--output",  type=Path, required=True)
    p.add_argument("--cascade", type=Path, default=None,
                   help="Optional path to lbpcascade_animeface.xml")
    args = p.parse_args()
    classify_all_panels(args.panels, args.output, args.cascade)
