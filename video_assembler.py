"""
Video Assembler Module
========================
Combines panel images + voiceover + background music + captions
into a final MP4 video using MoviePy.

The flow:
  1. Each image gets a duration (total_duration / num_images)
  2. Each image becomes a VideoClip with a Ken Burns make_frame function
  3. Clips are concatenated
  4. Voiceover + background music are mixed and attached
  5. Captions are overlaid as TextClips at their SRT timestamps
  6. Final composite is exported as H.264 MP4
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.audio.fx.volumex import volumex
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.VideoClip import TextClip, VideoClip
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout

from modules.caption_generator import parse_srt
from modules.image_processor import (
    fit_image_to_canvas,
    get_ken_burns_frame_function,
)

logger = logging.getLogger("video_assembler")


def assemble_video(
    image_paths: List[Path],
    voiceover_path: Path,
    music_path: Path,
    captions_path: Path,
    output_path: Path,
    config: dict,
    preview_mode: bool = False,
    image_durations: List[float] = None,
) -> Path:
    """
    Assemble the final video.
    
    Args:
        image_paths: List of panel image paths (in order)
        voiceover_path: MP3 voiceover file
        music_path: Background music MP3 (will loop if shorter than video)
        captions_path: SRT file with timed captions
        output_path: Where to save the final MP4
        config: Configuration dict (width, height, fps, volumes, etc.)
        preview_mode: If True, render only first 60 seconds
    
    Returns: Path to final video
    """
    # Config
    width = config["video_width"]
    height = config["video_height"]
    fps = config["fps"]
    bg_music_volume = config["bg_music_volume"]
    voiceover_volume = config["voiceover_volume"]
    zoom_intensity = config.get("zoom_intensity", 0.15)
    fade_duration = config.get("fade_duration", 0.2)
    caption_font = config.get("caption_font", "Arial-Bold")
    caption_font_size = config.get("caption_font_size", 48)
    caption_color = config.get("caption_color", "white")
    caption_stroke_color = config.get("caption_stroke_color", "black")
    caption_stroke_width = config.get("caption_stroke_width", 2)
    # Blur background fill settings (manhwa-recap style)
    blur_background = config.get("blur_background", True)
    blur_radius = config.get("blur_radius", 51)
    blur_dim = config.get("blur_dim", 0.7)
    foreground_scale = config.get("foreground_scale", 0.95)

    # Load voiceover to determine total video duration
    voiceover = AudioFileClip(str(voiceover_path))
    total_duration = voiceover.duration

    if preview_mode:
        logger.info("  PREVIEW MODE: rendering first 60 seconds only")
        total_duration = min(60.0, total_duration)
        voiceover = voiceover.subclip(0, total_duration)

    # How long each image is on screen.
    # If script-synced durations are supplied (one per image, matching the
    # narration's per-scene pacing), use them; otherwise fall back to the old
    # behaviour of spreading every image evenly across the audio.
    if image_durations is not None and len(image_durations) == len(image_paths):
        durations = [float(d) for d in image_durations]
        # Rescale so the durations sum to exactly the (possibly preview-trimmed)
        # audio length - keeps video locked to audio with no drift.
        scale = total_duration / sum(durations)
        durations = [d * scale for d in durations]
        logger.info(
            f"  Total duration: {total_duration:.1f}s | {len(image_paths)} images "
            f"(SCRIPT-SYNCED pacing, {min(durations):.2f}s-{max(durations):.2f}s per image)"
        )
    else:
        per_image_duration = total_duration / len(image_paths)
        durations = [per_image_duration] * len(image_paths)
        logger.info(
            f"  Total duration: {total_duration:.1f}s | "
            f"{len(image_paths)} images @ {per_image_duration:.2f}s each (even spread)"
        )

    # In preview mode, only build the panels that fit inside the 60s window.
    if preview_mode:
        kept, acc = [], 0.0
        for p, d in zip(image_paths, durations):
            if acc >= total_duration:
                break
            kept.append((p, min(d, total_duration - acc)))
            acc += d
        image_paths = [p for p, _ in kept]
        durations = [d for _, d in kept]

    # Build per-image clips with Ken Burns effect
    logger.info("  Building image clips with Ken Burns effects...")
    image_clips = []
    for i, img_path in enumerate(image_paths):
        clip = _build_image_clip(
            image_path=img_path,
            duration=durations[i],
            width=width,
            height=height,
            fps=fps,
            zoom_intensity=zoom_intensity,
            fade_duration=fade_duration,
            blur_background=blur_background,
            blur_radius=blur_radius,
            blur_dim=blur_dim,
            foreground_scale=foreground_scale,
        )
        image_clips.append(clip)
        if (i + 1) % 10 == 0:
            logger.info(f"    Prepared {i + 1}/{len(image_paths)} image clips")

    # Concatenate all image clips
    logger.info("  Concatenating image clips...")
    video = concatenate_videoclips(image_clips, method="compose")

    # In case the concatenated video drifts slightly off the audio,
    # trim or extend to exactly match
    video = video.set_duration(total_duration)

    # Build audio track: voiceover + looped background music
    logger.info("  Mixing audio tracks...")
    audio_tracks = [voiceover.fx(volumex, voiceover_volume)]
    
    if music_path.exists():
        music = AudioFileClip(str(music_path)).fx(volumex, bg_music_volume)
        # Loop music if it's shorter than the video
        if music.duration < total_duration:
            music = music.fx(audio_loop, duration=total_duration)
        else:
            music = music.subclip(0, total_duration)
        audio_tracks.append(music)
    else:
        logger.warning("  No music.mp3 found - using voiceover only")

    final_audio = CompositeAudioClip(audio_tracks)
    video = video.set_audio(final_audio)

    # Overlay captions (skip if disabled)
    if captions_path is not None and captions_path.exists():
        logger.info("  Overlaying captions...")
        caption_clips = _build_caption_clips(
            captions_path=captions_path,
            total_duration=total_duration,
            video_width=width,
            video_height=height,
            font=caption_font,
            font_size=caption_font_size,
            color=caption_color,
            stroke_color=caption_stroke_color,
            stroke_width=caption_stroke_width,
        )
        video = CompositeVideoClip([video] + caption_clips, size=(width, height))
    else:
        logger.info("  Captions skipped")

    # Export
    logger.info(f"  Encoding to MP4 (this is the slow part)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(output_path),
        fps=fps,
        codec="h264_nvenc",
        audio_codec="aac",
        audio_bitrate="192k",
        bitrate="6000k",
        preset="medium",
        threads=8,
        logger="bar",  # MoviePy progress bar
    )

    # Clean up resources
    video.close()
    voiceover.close()

    return output_path


def _build_image_clip(
    image_path: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
    zoom_intensity: float,
    fade_duration: float,
    blur_background: bool = True,
    blur_radius: int = 51,
    blur_dim: float = 0.7,
    foreground_scale: float = 0.95,
) -> VideoClip:
    """Build a single Ken Burns clip from an image with blurred-background fill."""
    # Load image and fit to canvas using the manhwa-style blur fill
    raw = cv2.imread(str(image_path))
    if raw is None:
        logger.warning(f"  Could not read {image_path.name}, using black placeholder")
        raw = np.zeros((height, width, 3), dtype=np.uint8)
    
    fitted = fit_image_to_canvas(
        raw,
        target_width=width,
        target_height=height,
        blur_background=blur_background,
        blur_radius=blur_radius,
        blur_dim=blur_dim,
        foreground_scale=foreground_scale,
    )
    
    # Get frame-generation function (memory-efficient)
    make_frame, _ = get_ken_burns_frame_function(
        image=fitted,
        duration_seconds=duration,
        effect_type="random",
        zoom_intensity=zoom_intensity,
    )
    
    clip = VideoClip(make_frame, duration=duration)
    clip = clip.set_fps(fps)
    
    # Apply fade in/out
    if fade_duration > 0 and duration > fade_duration * 2:
        clip = clip.fx(fadein, fade_duration).fx(fadeout, fade_duration)
    
    return clip


def _build_caption_clips(
    captions_path: Path,
    total_duration: float,
    video_width: int,
    video_height: int,
    font: str,
    font_size: int,
    color: str,
    stroke_color: str,
    stroke_width: int,
) -> List[TextClip]:
    """Build TextClips for each caption entry in the SRT file."""
    entries = parse_srt(captions_path)
    clips = []
    
    for start, end, text in entries:
        if start >= total_duration:
            break  # Skip captions beyond video duration
        
        end = min(end, total_duration)
        clip_duration = end - start
        if clip_duration <= 0:
            continue
        
        try:
            txt = TextClip(
                text,
                fontsize=font_size,
                font=font,
                color=color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                method="caption",
                size=(int(video_width * 0.85), None),
                align="center",
            )
        except Exception as e:
            # Fallback if font is missing
            logger.warning(f"  Font '{font}' failed, using default: {e}")
            txt = TextClip(
                text,
                fontsize=font_size,
                color=color,
                method="caption",
                size=(int(video_width * 0.85), None),
                align="center",
            )
        
        # Position: bottom-center with margin
        txt = txt.set_position(("center", int(video_height * 0.82)))
        txt = txt.set_start(start).set_duration(clip_duration)
        clips.append(txt)
    
    return clips
