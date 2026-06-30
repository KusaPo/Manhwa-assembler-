"""
Visual review tool — render a contact sheet showing every panel labeled
with its motion classification. Lets you spot mis-classifications and
edit motion_map.json directly to fix them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CATEGORY_COLORS = {
    "face_closeup":   (52, 152, 219),
    "group_scene":    (46, 204, 113),
    "action_impact":  (231, 76, 60),
    "wide_establish": (155, 89, 182),
    "detail_narrow":  (241, 196, 15),
    "static_quiet":   (149, 165, 166),
}


def build_review_sheet(
    panels_dir: Path,
    motion_map_path: Path,
    output_path: Path,
    thumb_w: int = 320,
    cols: int = 6,
) -> Path:
    motion_map = json.loads(motion_map_path.read_text(encoding="utf-8"))
    panel_files = sorted(panels_dir.glob("*-panel.webp")) + \
                  sorted(panels_dir.glob("*-panel.png"))
    if not panel_files:
        raise FileNotFoundError(f"No panels in {panels_dir}")

    thumb_h = int(thumb_w * 9 / 16)
    label_h = 50
    cell_w = thumb_w + 16
    cell_h = thumb_h + label_h + 16

    n = len(panel_files)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "black")
    draw = ImageDraw.Draw(sheet)

    try:
        font  = ImageFont.truetype("arial.ttf", 16)
        small = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        try:
            font  = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
            small = ImageFont.truetype("DejaVuSans.ttf", 13)
        except Exception:
            font = small = ImageFont.load_default()

    for i, pf in enumerate(panel_files):
        try:
            panel_num = int(pf.stem.split("-")[0])
        except (ValueError, IndexError):
            continue

        entry    = motion_map.get(str(panel_num), {})
        category = entry.get("category", "?")
        motion   = entry.get("motion", "?")
        conf     = entry.get("confidence", 0.0)
        color    = CATEGORY_COLORS.get(category, (100, 100, 100))

        img = Image.open(pf)
        img.thumbnail((thumb_w, thumb_h))
        col, row = i % cols, i // cols
        x = col * cell_w + 8
        y = row * cell_h + 8

        draw.rectangle(
            [x - 4, y - 4, x + thumb_w + 4, y + thumb_h + 4],
            outline=color, width=4,
        )
        sheet.paste(img, (x, y))

        label_y = y + thumb_h + 6
        draw.text((x, label_y),      f"#{panel_num:>3}  {category}", fill=color,         font=font)
        draw.text((x, label_y + 20), f"{motion}  ({conf:.2f})",      fill=(200, 200, 200), font=small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Review sheet: {output_path}  ({sheet.size[0]}x{sheet.size[1]}px, {n} panels)")
    print("\nBorder colors: blue=face, green=group, red=action, purple=wide, yellow=detail, gray=quiet")
    print(f"\nTo override: edit {motion_map_path} and change the 'motion' field.")
    print("Valid motions: zoom_in_slow, zoom_out_slow, pan_down, pan_up,")
    print("               zoom_in_center, drift_diagonal, static")
    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build motion classification review sheet")
    p.add_argument("--panels",     type=Path, required=True)
    p.add_argument("--motion-map", type=Path, required=True)
    p.add_argument("--output",     type=Path, required=True)
    p.add_argument("--cols",       type=int,  default=6)
    args = p.parse_args()
    build_review_sheet(args.panels, args.motion_map, args.output, cols=args.cols)
