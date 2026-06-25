"""
Utility Module
==============
Shared helpers: logging setup, validation, formatting.
"""

import logging
import sys
from pathlib import Path

from modules.sync.timeline import load_timeline_panels

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
            if p.suffix.lower() in IMAGE_EXTENSIONS
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


def validate_timeline_images(
    images_dir: Path,
    timeline_file: Path,
    audio_sync_mode: bool,
) -> bool:
    """
    In audio-sync mode, require every panel listed in timeline.json on disk.
    Returns True if valid (or check not applicable).
    """
    logger = logging.getLogger("validator")
    if not audio_sync_mode or not timeline_file.exists():
        return True

    if not images_dir.exists():
        return False

    on_disk = {
        p.name
        for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    expected = [panel.file for panel in load_timeline_panels(timeline_file)]
    missing = [name for name in expected if name not in on_disk]

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        logger.error(
            f"  {len(missing)} timeline panels missing from {images_dir}/."
        )
        logger.error(
            f"  Expected {len(expected)}, found {len(on_disk)}. "
            f"Missing: {preview}{suffix}"
        )
        logger.error(
            "  Copy all panel images matching timeline.json filenames before running."
        )
        return False

    logger.info(f"  Images OK: {len(expected)}/{len(expected)} timeline panels found")
    return True


def format_duration(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
