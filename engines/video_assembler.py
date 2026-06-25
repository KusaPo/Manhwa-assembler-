"""
Assembly Engine — timestamp-driven Ken Burns video, SRT, and mux.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import imageio_ffmpeg
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

from core.config import RecapConfig, ProjectPaths
from core.models import SyncPlan, SyncPlanEntry, TimestampSegment, load_timestamps
from engines.image_processor import fit_image_to_canvas, get_ken_burns_frame_function

logger = logging.getLogger("engines.assembler")

DRIFT_FAIL_THRESHOLD_S = 2.0
DRIFT_RESCALE_THRESHOLD_S = 0.05


def _seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_from_timestamps(
    segments: List[TimestampSegment],
    output_path: Path,
    max_words: int = 12,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        entry_num = 1
        for seg in segments:
            words = seg.text.split()
            if len(words) <= max_words:
                chunks = [seg.text]
            else:
                chunks = [
                    " ".join(words[i:i + max_words])
                    for i in range(0, len(words), max_words)
                ]
            chunk_dur = seg.duration / len(chunks)
            for i, chunk in enumerate(chunks):
                start = seg.start + i * chunk_dur
                end = start + chunk_dur
                f.write(f"{entry_num}\n")
                f.write(f"{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}\n")
                f.write(f"{chunk}\n\n")
                entry_num += 1
    return output_path


def parse_srt(srt_path: Path) -> List[Tuple[float, float, str]]:
    entries = []
    content = srt_path.read_text(encoding="utf-8").strip()
    for block in content.split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        timing = lines[1]
        text = " ".join(lines[2:])
        start_str, end_str = timing.split(" --> ")
        entries.append((
            _srt_time_to_seconds(start_str),
            _srt_time_to_seconds(end_str),
            text,
        ))
    return entries


def _srt_time_to_seconds(srt_time: str) -> float:
    time_part, ms_part = srt_time.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms_part) / 1000.0


def assign_panels_to_segments(
    panel_paths: List[Path],
    segments: List[TimestampSegment],
    panels_per_segment_max: int = 2,
) -> SyncPlan:
    if not panel_paths:
        raise ValueError("No panel images available")
    if not segments:
        raise ValueError("No timestamp segments available")

    total_panels = len(panel_paths)
    total_segments = len(segments)
    entries: List[SyncPlanEntry] = []
    panel_idx = 0

    for seg in segments:
        seg_panels = min(
            panels_per_segment_max,
            max(1, round(seg.duration / sum(s.duration for s in segments) * total_panels)),
        )
        seg_panels = min(seg_panels, total_panels - panel_idx)
        if panel_idx >= total_panels:
            break
        if seg_panels < 1:
            seg_panels = 1

        per_panel_dur = seg.duration / seg_panels
        for _ in range(seg_panels):
            if panel_idx >= total_panels:
                break
            entries.append(
                SyncPlanEntry(
                    panel_index=panel_idx,
                    panel_path=panel_paths[panel_idx],
                    segment_index=seg.index,
                    duration=per_panel_dur,
                )
            )
            panel_idx += 1

    while panel_idx < total_panels and entries:
        extra_dur = 0.25
        entries.append(
            SyncPlanEntry(
                panel_index=panel_idx,
                panel_path=panel_paths[panel_idx],
                segment_index=segments[-1].index,
                duration=extra_dur,
            )
        )
        panel_idx += 1

    return SyncPlan(entries=entries)


def _verify_audio_stream(video_path: Path) -> bool:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-i", str(video_path)],
        capture_output=True,
        text=True,
    )
    return "Audio:" in result.stderr


class VideoAssembler:
    def __init__(self, config: RecapConfig, paths: ProjectPaths) -> None:
        self.config = config
        self.paths = paths

    def assemble(
        self,
        panel_paths: List[Path],
        *,
        preview_mode: bool = False,
        burn_captions: bool = True,
    ) -> Path:
        segments = load_timestamps(self.paths.timestamps_file)
        sync_plan = assign_panels_to_segments(
            panel_paths,
            segments,
            panels_per_segment_max=self.config.panels_per_segment_max,
        )
        sync_plan.save(self.paths.sync_plan_file)

        generate_srt_from_timestamps(segments, self.paths.subtitles_file)

        image_paths = [e.panel_path for e in sync_plan.entries]
        image_durations = [e.duration for e in sync_plan.entries]

        return assemble_video(
            image_paths=image_paths,
            image_durations=image_durations,
            voiceover_path=self.paths.voiceover_file,
            music_path=self.paths.music_file,
            captions_path=self.paths.subtitles_file if burn_captions else None,
            output_path=self.paths.final_video,
            config=self.config.as_dict(),
            preview_mode=preview_mode,
            sentence_count=len(segments),
        )


def assemble_video(
    image_paths: List[Path],
    image_durations: List[float],
    voiceover_path: Path,
    music_path: Path,
    captions_path: Path | None,
    output_path: Path,
    config: dict,
    preview_mode: bool = False,
    sentence_count: int = 0,
) -> Path:
    width = config["video_width"]
    height = config["video_height"]
    fps = config["fps"]
    bg_music_volume = config["bg_music_volume"]
    voiceover_volume = config["voiceover_volume"]
    zoom_intensity = config.get("zoom_intensity", 0.15)
    fade_duration = config.get("fade_duration", 0.3)
    caption_font = config.get("caption_font", "Arial-Bold")
    caption_font_size = config.get("caption_font_size", 48)
    caption_color = config.get("caption_color", "white")
    caption_stroke_color = config.get("caption_stroke_color", "black")
    caption_stroke_width = config.get("caption_stroke_width", 3)
    blur_background = config.get("blur_background", True)
    blur_radius = config.get("blur_radius", 51)
    blur_dim = config.get("blur_dim", 0.7)
    foreground_scale = config.get("foreground_scale", 0.95)

    voiceover = AudioFileClip(str(voiceover_path))
    if voiceover.duration <= 0:
        voiceover.close()
        raise ValueError(f"Voiceover has no audio content: {voiceover_path}")

    total_duration = voiceover.duration
    if preview_mode:
        logger.info("  PREVIEW MODE: first 60 seconds only")
        total_duration = min(60.0, total_duration)
        voiceover = voiceover.subclip(0, total_duration)

    durations = [float(d) for d in image_durations]
    inter_sentence_pause_ms = config.get("inter_sentence_pause_ms", 0)
    pause_total_s = inter_sentence_pause_ms * max(0, sentence_count - 1) / 1000.0
    expected_panel_sum = total_duration - pause_total_s
    panel_sum = sum(durations)
    drift = abs(panel_sum - expected_panel_sum)

    if drift > DRIFT_FAIL_THRESHOLD_S:
        voiceover.close()
        raise ValueError(
            f"Panel durations ({panel_sum:.1f}s) do not match voiceover "
            f"({total_duration:.1f}s). Check panels and timestamps."
        )
    if drift > DRIFT_RESCALE_THRESHOLD_S and panel_sum > 0:
        scale = expected_panel_sum / panel_sum
        durations = [d * scale for d in durations]
        logger.info(f"  Rescaled panel durations by {scale:.4f} (drift {drift:.2f}s)")

    if preview_mode:
        kept, acc = [], 0.0
        for p, d in zip(image_paths, durations):
            if acc >= total_duration:
                break
            kept.append((p, min(d, total_duration - acc)))
            acc += d
        image_paths = [p for p, _ in kept]
        durations = [d for _, d in kept]

    logger.info(
        f"  Building {len(image_paths)} clips "
        f"({total_duration:.1f}s total)"
    )
    image_clips = []
    for i, (img_path, dur) in enumerate(zip(image_paths, durations)):
        clip = _build_image_clip(
            img_path, dur, width, height, fps, zoom_intensity, fade_duration,
            blur_background, blur_radius, blur_dim, foreground_scale,
        )
        image_clips.append(clip)

    video = concatenate_videoclips(image_clips, method="compose")

    audio_tracks = [voiceover.fx(volumex, voiceover_volume)]
    if music_path.exists():
        music = AudioFileClip(str(music_path)).fx(volumex, bg_music_volume)
        if music.duration < total_duration:
            music = music.fx(audio_loop, duration=total_duration)
        else:
            music = music.subclip(0, total_duration)
        audio_tracks.append(music)
    else:
        logger.warning("  No background music — voiceover only")

    final_audio = CompositeAudioClip(audio_tracks).set_duration(total_duration)
    video = video.set_audio(final_audio)

    if captions_path and captions_path.exists():
        caption_clips = _build_caption_clips(
            captions_path, total_duration, width, height,
            caption_font, caption_font_size, caption_color,
            caption_stroke_color, caption_stroke_width,
        )
        video = CompositeVideoClip([video] + caption_clips, size=(width, height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  Encoding MP4...")
    video.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        bitrate="6000k",
        preset="medium",
        threads=8,
        logger="bar",
    )

    if not _verify_audio_stream(output_path):
        raise RuntimeError(f"Output video has no audio track: {output_path}")

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
    blur_background: bool,
    blur_radius: int,
    blur_dim: float,
    foreground_scale: float,
) -> VideoClip:
    raw = cv2.imread(str(image_path))
    if raw is None:
        logger.warning(f"  Could not read {image_path.name}, using placeholder")
        raw = np.zeros((height, width, 3), dtype=np.uint8)

    fitted = fit_image_to_canvas(
        raw, width, height, blur_background, blur_radius, blur_dim, foreground_scale,
    )
    make_frame, _ = get_ken_burns_frame_function(
        fitted, duration, "random", zoom_intensity,
    )
    clip = VideoClip(make_frame, duration=duration).set_fps(fps)
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
    entries = parse_srt(captions_path)
    clips = []
    for start, end, text in entries:
        if start >= total_duration:
            break
        end = min(end, total_duration)
        clip_duration = end - start
        if clip_duration <= 0:
            continue
        try:
            txt = TextClip(
                text, fontsize=font_size, font=font,
                color=color, stroke_color=stroke_color, stroke_width=stroke_width,
                method="caption", size=(int(video_width * 0.85), None), align="center",
            )
        except Exception as exc:
            logger.warning(f"  Font fallback: {exc}")
            txt = TextClip(
                text, fontsize=font_size, color=color,
                method="caption", size=(int(video_width * 0.85), None), align="center",
            )
        txt = txt.set_position(("center", int(video_height * 0.82)))
        txt = txt.set_start(start).set_duration(clip_duration)
        clips.append(txt)
    return clips
