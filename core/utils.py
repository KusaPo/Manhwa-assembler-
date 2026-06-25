"""Logging, validation, and formatting helpers."""

import logging
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def setup_logging(log_file: str = "assembly.log") -> None:
    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    date_format = "%H:%M:%S"

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def validate_script_input(
    raw_chapter_file: Path,
    script_txt: Path,
    script_json: Path,
) -> bool:
    logger = logging.getLogger("validator")
    if script_json.exists():
        logger.info(f"  Script JSON OK: {script_json}")
        return True
    if script_txt.exists() and script_txt.read_text(encoding="utf-8").strip():
        logger.info(f"  Script TXT OK: {script_txt}")
        return True
    if raw_chapter_file.exists() and raw_chapter_file.read_text(encoding="utf-8").strip():
        logger.info(f"  Raw chapter OK: {raw_chapter_file}")
        return True
    logger.error("  No script input found. Add input/raw_chapter.txt or input/script.txt")
    return False


def validate_strips_dir(strips_dir: Path) -> bool:
    logger = logging.getLogger("validator")
    if not strips_dir.exists():
        logger.error(f"  Missing raw strips directory: {strips_dir}")
        return False
    count = sum(
        1 for p in strips_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if count == 0:
        logger.error(f"  No strip images in {strips_dir}")
        return False
    logger.info(f"  Raw strips OK: {count} images")
    return True


def validate_panels_dir(panels_dir: Path) -> bool:
    logger = logging.getLogger("validator")
    if not panels_dir.exists():
        logger.error(f"  Missing panels directory: {panels_dir}")
        return False
    count = sum(
        1 for p in panels_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if count == 0:
        logger.error(f"  No panel images in {panels_dir}. Run --phase vision first.")
        return False
    logger.info(f"  Panels OK: {count} images")
    return True
