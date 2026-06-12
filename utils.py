"""
Utility Module
==============
Shared helpers: logging setup, validation, formatting.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_file: str = "assembly.log") -> None:
    """Configure root logger with console + file output."""
    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    date_format = "%H:%M:%S"
    
    # Clear any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def validate_inputs(script_file: Path, images_dir: Path, music_file: Path) -> bool:
    """
    Check that all required inputs exist. Returns True if valid.
    Music is optional; script + images are required.
    """
    logger = logging.getLogger("validator")
    valid = True

    if not script_file.exists():
        logger.error(f"  Missing script: {script_file}")
        logger.error(f"    Create this file with your video narration text.")
        valid = False
    else:
        content = script_file.read_text(encoding="utf-8").strip()
        if not content:
            logger.error(f"  Script file is empty: {script_file}")
            valid = False
        else:
            word_count = len(content.split())
            logger.info(f"  Script OK: {word_count} words (~{word_count / 155:.1f} min audio)")

    if not images_dir.exists():
        logger.error(f"  Missing images directory: {images_dir}")
        valid = False
    else:
        image_count = sum(
            1 for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        if image_count == 0:
            logger.error(f"  No images found in {images_dir}")
            valid = False
        else:
            logger.info(f"  Images OK: {image_count} panels found")

    if not music_file.exists():
        logger.warning(f"  No music file at {music_file} (optional, will skip)")
    else:
        logger.info(f"  Music OK: {music_file.name}")

    return valid


def format_duration(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
