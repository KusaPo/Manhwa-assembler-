"""
Automated Manhwa Video Assembly Script
=======================================
Pipeline: script.txt + images/ + music.mp3 -> final_video.mp4

Workflow:
  1. Read script.txt
  2. Generate voiceover via ElevenLabs API
  3. Generate timed captions from script text (matched to audio duration)
  4. Load images, apply Ken Burns effect (zoom/pan)
  5. Sync images to voiceover timing
  6. Mix background music under voiceover
  7. Burn captions into video
  8. Export 1080p MP4

Usage:
  python assembly_script.py
  python assembly_script.py --dry-run
  python assembly_script.py --preview
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Local modules
from modules.script_processor import parse_script
from modules.voice_generator import generate_voiceover, get_audio_duration
from modules.image_processor import load_images, apply_ken_burns_effect
from modules.video_assembler import assemble_video
from modules.utils import setup_logging, validate_inputs, format_duration


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        print("Create config.json with your API keys. See README.md for template.")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        return json.load(f)


def load_timeline(timeline_path: str, images_dir: Path, image_paths):
    """
    Load the script-synced timeline (timeline.json) if present.

    Returns (ordered_image_paths, durations) where the images are reordered to
    match the timeline and each has a per-panel duration that reflects how long
    the narration spends on that scene. If no timeline is found, returns the
    original image_paths and None (caller falls back to even spread).
    """
    logger = logging.getLogger("assembler")
    if not os.path.exists(timeline_path):
        logger.info("      No timeline.json found - using even image spread")
        return image_paths, None

    with open(timeline_path, "r") as f:
        timeline = json.load(f)

    by_name = {p.name: p for p in image_paths}
    ordered, durations, missing = [], [], []
    for entry in timeline["panels"]:
        name = entry["file"]
        if name in by_name:
            ordered.append(by_name[name])
            durations.append(float(entry["dur"]))
        else:
            missing.append(name)

    # Any images on disk not listed in the timeline get appended at the end
    # with a neutral duration so nothing is silently dropped.
    leftover = [p for p in image_paths if p.name not in {e["file"] for e in timeline["panels"]}]
    if leftover:
        avg = sum(durations) / len(durations) if durations else 2.0
        for p in leftover:
            ordered.append(p)
            durations.append(avg)

    if missing:
        logger.warning(f"      {len(missing)} timeline panels missing from disk (skipped)")
    logger.info(
        f"      Timeline loaded: {len(ordered)} panels, script-synced pacing "
        f"({min(durations):.2f}s-{max(durations):.2f}s per panel)"
    )
    return ordered, durations


def main():
    parser = argparse.ArgumentParser(
        description="Assemble manhwa recap video from images + script"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without rendering"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render only first 60 seconds for QA"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Voiceover speed multiplier (0.8 = slower, 1.2 = faster)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--skip-voiceover",
        action="store_true",
        help="Reuse existing voiceover.mp3 if present (saves API credits)"
    )
    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = logging.getLogger("assembler")
    config = load_config(args.config)

    # Paths
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    images_dir = input_dir / "images"
    script_file = input_dir / "script.txt"
    music_file = input_dir / "music.mp3"
    voiceover_file = output_dir / "voiceover.mp3"
    final_video = output_dir / "final_video.mp4"

    # Validation
    logger.info("=" * 60)
    logger.info("MANHWA VIDEO ASSEMBLY - STARTING")
    logger.info("=" * 60)
    
    if not validate_inputs(script_file, images_dir, music_file):
        logger.error("Input validation failed. Fix issues above and retry.")
        sys.exit(1)

    # Step 1: Parse script
    logger.info("[1/6] Parsing script...")
    script_segments = parse_script(script_file)
    logger.info(f"      Found {len(script_segments)} script segments")

    # Step 2: Generate voiceover
    if args.skip_voiceover and voiceover_file.exists():
        logger.info("[2/6] Skipping voiceover generation (using existing file)")
    elif args.dry_run:
        logger.info("[2/6] [DRY RUN] Would generate voiceover via ElevenLabs")
    else:
        logger.info("[2/6] Generating voiceover via ElevenLabs API...")
        generate_voiceover(
            text=" ".join(script_segments),
            output_path=voiceover_file,
            voice_id=config["elevenlabs_voice_id"],
            api_key=config["elevenlabs_api_key"],
            speed=args.speed,
        )
        logger.info(f"      Voiceover saved: {voiceover_file}")

    if args.dry_run:
        logger.info("[DRY RUN] Skipping rendering. Exiting.")
        return

    # Step 3: Get audio duration
    audio_duration = get_audio_duration(voiceover_file)
    logger.info(f"[3/5] Voiceover duration: {format_duration(audio_duration)}")

    # Captions disabled - skip caption generation entirely
    logger.info("[4/5] Captions disabled, skipping...")

    # Step 5: Load images & validate
    logger.info("[5/5] Loading panel images...")
    image_paths = load_images(images_dir)
    logger.info(f"      Found {len(image_paths)} images")

    if len(image_paths) == 0:
        logger.error("No images found in input/images/. Add panels and retry.")
        sys.exit(1)

    # Apply script-synced ordering + per-panel durations (timeline.json)
    timeline_file = input_dir / "timeline.json"
    image_paths, image_durations = load_timeline(str(timeline_file), images_dir, image_paths)

    # Step 6: Assemble video
    logger.info("Assembling video (this takes ~10-15 min)...")
    assemble_video(
        image_paths=image_paths,
        voiceover_path=voiceover_file,
        music_path=music_file,
        captions_path=None,
        output_path=final_video,
        config=config,
        preview_mode=args.preview,
        image_durations=image_durations,
    )

    # Summary
    final_size_mb = os.path.getsize(final_video) / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("ASSEMBLY COMPLETE")
    logger.info(f"Output: {final_video}")
    logger.info(f"Duration: {format_duration(audio_duration)}")
    logger.info(f"Size: {final_size_mb:.1f} MB")
    logger.info(f"Resolution: {config['video_width']}x{config['video_height']} @ {config['fps']}fps")
    logger.info("Ready for YouTube upload.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
