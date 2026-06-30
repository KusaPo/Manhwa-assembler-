"""
Assembly Engine — ffmpeg-native Ken Burns pipeline with parallel clip rendering.

Each panel is rendered to a temp MP4 using ffmpeg's zoompan filter (no Python
frame loop). Clips are concatenated in one ffmpeg pass with audio mixing and
optional subtitle burn-in. Encoding uses h264_nvenc when available.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import imageio_ffmpeg
import mutagen.mp3

from core.config import RecapConfig, ProjectPaths
from core.models import SyncPlan, SyncPlanEntry, TimestampSegment, load_timestamps

logger = logging.getLogger("engines.assembler")

DRIFT_RESCALE_THRESHOLD_S = 0.05

SUBCLIP_TARGET_S = 5.5
SUBCLIP_MAX_S    = 7.0
MIN_HOLD_S       = 1.8

NVENC_MAX_WORKERS = 3  # consumer GPU NVENC session limit

EFFECT_TYPES = ["zoom_in", "zoom_out", "pan_left", "pan_right", "static"]
EFFECT_WEIGHTS = [35, 30, 15, 15, 5]


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sync-plan builders
# ---------------------------------------------------------------------------

def _claim_segments_by_word_count(
    narration: str, available: List[TimestampSegment]
) -> int:
    target_words = len(narration.split())
    if target_words == 0 or not available:
        return 0
    accumulated = 0
    for i, seg in enumerate(available):
        accumulated += len(seg.text.split())
        if accumulated >= target_words:
            return i + 1
    return len(available)


def _plan_subclips(narration_duration: float) -> Tuple[int, float]:
    d = max(narration_duration, MIN_HOLD_S)
    if d <= SUBCLIP_MAX_S:
        return 1, d
    n = max(2, round(d / SUBCLIP_TARGET_S))
    while d / n > SUBCLIP_MAX_S:
        n += 1
    return n, d / n


def build_sync_plan_from_script(
    panel_script_path: Path,
    panel_paths: List[Path],
    segments: List[TimestampSegment],
) -> SyncPlan:
    """Script-aware allocator: each panel's screen time equals its narration's
    TTS duration. Long narrations split into multiple Ken Burns sub-clips on
    the same image. Panels matched sequentially to script entries."""
    if not panel_paths:
        raise ValueError("No panel images available")
    if not segments:
        raise ValueError("No timestamp segments available")

    script_data = json.loads(panel_script_path.read_text(encoding="utf-8"))
    sorted_panels = sorted(panel_paths)

    entries: List[SyncPlanEntry] = []
    remaining = list(segments)
    clip_index = 0
    long_panels = 0

    for seq_idx, script_entry in enumerate(script_data):
        if seq_idx >= len(sorted_panels):
            logger.warning(f"  Script has more entries than panel images; stopping at entry {seq_idx}")
            break
        if not remaining:
            logger.warning("  Ran out of TTS segments before script ended")
            break

        panel_path = sorted_panels[seq_idx]
        narration = script_entry["narration"]
        n_claim = _claim_segments_by_word_count(narration, remaining)
        if n_claim == 0:
            n_claim = 1

        owned = remaining[:n_claim]
        remaining = remaining[n_claim:]
        narration_duration = sum(s.duration for s in owned)
        seg_index = owned[0].index

        n_subclips, per_dur = _plan_subclips(narration_duration)
        if n_subclips > 1:
            long_panels += 1

        for _ in range(n_subclips):
            entries.append(SyncPlanEntry(
                panel_index=clip_index,
                panel_path=panel_path,
                segment_index=seg_index,
                duration=per_dur,
            ))
            clip_index += 1

    if remaining and entries:
        leftover = sum(s.duration for s in remaining)
        last = entries[-1]
        entries[-1] = SyncPlanEntry(
            panel_index=last.panel_index,
            panel_path=last.panel_path,
            segment_index=last.segment_index,
            duration=last.duration + leftover,
        )
        logger.info(f"  Extended final panel by {leftover:.2f}s (trailing segments)")

    logger.info(
        f"  Built {len(entries)} clips from {len(script_data)} script entries "
        f"({long_panels} panel(s) split into sub-clips)"
    )
    return SyncPlan(entries=entries)


def assign_panels_to_segments(
    panel_paths: List[Path],
    segments: List[TimestampSegment],
    panels_per_segment_max: int = 2,
) -> SyncPlan:
    """Fallback proportional allocator (used when panel_script.json is absent)."""
    if not panel_paths:
        raise ValueError("No panel images available")
    if not segments:
        raise ValueError("No timestamp segments available")

    n_panels = len(panel_paths)
    n_segments = len(segments)
    total_duration = sum(s.duration for s in segments)

    raw = [s.duration / total_duration * n_panels for s in segments]
    allocs = [min(panels_per_segment_max, int(r)) for r in raw]
    remainder = n_panels - sum(allocs)

    fractions = sorted(range(n_segments), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    i = 0
    while remainder > 0 and i < len(fractions) * 4:
        idx = fractions[i % len(fractions)]
        if allocs[idx] < panels_per_segment_max:
            allocs[idx] += 1
            remainder -= 1
        i += 1
    if remainder > 0:
        longest = sorted(range(n_segments), key=lambda i: -segments[i].duration)
        for idx in longest:
            if remainder == 0:
                break
            allocs[idx] += 1
            remainder -= 1

    entries: List[SyncPlanEntry] = []
    panel_idx = 0
    carry = 0.0
    skipped = 0
    for seg, n in zip(segments, allocs):
        if n == 0:
            carry += seg.duration
            skipped += 1
            continue
        seg_dur = seg.duration + carry
        carry = 0.0
        per_panel_dur = seg_dur / n
        for _ in range(n):
            entries.append(SyncPlanEntry(
                panel_index=panel_idx,
                panel_path=panel_paths[panel_idx],
                segment_index=seg.index,
                duration=per_panel_dur,
            ))
            panel_idx += 1

    if carry > 0 and entries:
        last = entries[-1]
        entries[-1] = SyncPlanEntry(
            panel_index=last.panel_index,
            panel_path=last.panel_path,
            segment_index=last.segment_index,
            duration=last.duration + carry,
        )

    if skipped:
        logger.info(
            f"  {skipped} segment(s) had no panel — duration merged into neighbors"
        )
    return SyncPlan(entries=entries)


# ---------------------------------------------------------------------------
# ffmpeg Ken Burns helpers
# ---------------------------------------------------------------------------

def _check_nvenc(ffmpeg_exe: str) -> bool:
    try:
        r = subprocess.run([ffmpeg_exe, "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


def _zoompan_filter(
    effect: str,
    n_frames: int,
    fps: int,
    width: int,
    height: int,
    zoom_intensity: float,
) -> str:
    zi = zoom_intensity
    step = zi / max(n_frames, 1)

    if effect == "zoom_in":
        z = f"min(zoom+{step:.7f},{1 + zi:.5f})"
        x = "trunc(iw/2-(iw/zoom/2))"
        y = "trunc(ih/2-(ih/zoom/2))"
    elif effect == "zoom_out":
        z = f"if(eq(on,1),{1 + zi:.5f},max(zoom-{step:.7f},1))"
        x = "trunc(iw/2-(iw/zoom/2))"
        y = "trunc(ih/2-(ih/zoom/2))"
    elif effect == "pan_left":
        z = f"{1 + zi * 0.5:.5f}"
        x = f"trunc((iw-iw/zoom)*on/{n_frames})"
        y = "trunc(ih/2-(ih/zoom/2))"
    elif effect == "pan_right":
        z = f"{1 + zi * 0.5:.5f}"
        x = f"trunc((iw-iw/zoom)*(1-on/{n_frames}))"
        y = "trunc(ih/2-(ih/zoom/2))"
    else:  # static
        z = "1"
        x = "0"
        y = "0"

    return f"zoompan=z='{z}':x='{x}':y='{y}':d={n_frames}:s={width}x{height}:fps={fps}"


def _render_panel_clip(args: tuple) -> Path:
    """Render one image to a temp video clip via ffmpeg zoompan (runs in thread pool)."""
    (ffmpeg_exe, image_path, duration, out_path,
     fps, width, height, zoom_intensity, fade_duration, use_nvenc) = args

    n_frames = max(1, round(fps * duration))
    effect = random.choices(EFFECT_TYPES, weights=EFFECT_WEIGHTS, k=1)[0]
    zp = _zoompan_filter(effect, n_frames, fps, width, height, zoom_intensity)

    vf_parts = [zp]
    if fade_duration > 0 and duration > fade_duration * 2:
        vf_parts.append(f"fade=t=in:st=0:d={fade_duration:.3f}")
        vf_parts.append(f"fade=t=out:st={duration - fade_duration:.3f}:d={fade_duration:.3f}")
    vf = ",".join(vf_parts)

    if use_nvenc:
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "8M"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"]

    cmd = [
        ffmpeg_exe, "-y",
        "-loop", "1", "-framerate", str(fps),
        "-i", str(image_path),
        "-vf", vf,
        "-t", f"{duration:.4f}",
        *codec_args,
        "-an", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Clip render failed ({image_path.name}):\n{r.stderr[-800:]}")
    return out_path


def _verify_audio_stream(video_path: Path) -> bool:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ffmpeg, "-i", str(video_path)], capture_output=True, text=True)
    return "Audio:" in r.stderr


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

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

        if self.paths.panel_script_file.exists():
            logger.info(f"  Using script-aware allocator ({self.paths.panel_script_file.name})")
            sync_plan = build_sync_plan_from_script(
                self.paths.panel_script_file, panel_paths, segments,
            )
        else:
            logger.info("  panel_script.json missing — falling back to proportional allocator")
            sync_plan = assign_panels_to_segments(
                panel_paths, segments,
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
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    width = config["video_width"]
    height = config["video_height"]
    fps = config["fps"]
    bg_music_volume = config["bg_music_volume"]
    voiceover_volume = config["voiceover_volume"]
    zoom_intensity = config.get("zoom_intensity", 0.15)
    fade_duration = config.get("fade_duration", 0.3)

    # Voiceover duration via mutagen (no MoviePy needed)
    vo_meta = mutagen.mp3.MP3(str(voiceover_path))
    total_duration = vo_meta.info.length
    if total_duration <= 0:
        raise ValueError(f"Voiceover has no audio content: {voiceover_path}")

    if preview_mode:
        logger.info("  PREVIEW MODE: first 60 seconds only")
        total_duration = min(60.0, total_duration)

    # Trim and drift-rescale
    pairs = list(zip(image_paths, [float(d) for d in image_durations]))
    if preview_mode:
        kept, acc = [], 0.0
        for p, d in pairs:
            if acc >= total_duration:
                break
            kept.append((p, min(d, total_duration - acc)))
            acc += d
        pairs = kept

    panel_sum = sum(d for _, d in pairs)
    drift = abs(panel_sum - total_duration)
    if drift > DRIFT_RESCALE_THRESHOLD_S and panel_sum > 0:
        scale = total_duration / panel_sum
        pairs = [(p, d * scale) for p, d in pairs]
        logger.info(f"  Rescaled panel durations by {scale:.4f} (drift {drift:.2f}s)")

    # NVENC availability
    use_nvenc = _check_nvenc(ffmpeg_exe)
    logger.info(f"  Encoder: {'h264_nvenc (GPU)' if use_nvenc else 'libx264 (CPU)'}")

    # Parallel clip rendering
    temp_dir = output_path.parent / "_temp_clips"
    temp_dir.mkdir(exist_ok=True)

    render_args = [
        (ffmpeg_exe, img, dur, temp_dir / f"clip_{i:04d}.mp4",
         fps, width, height, zoom_intensity, fade_duration, use_nvenc)
        for i, (img, dur) in enumerate(pairs)
    ]
    clip_paths = [temp_dir / f"clip_{i:04d}.mp4" for i in range(len(pairs))]
    max_workers = NVENC_MAX_WORKERS if use_nvenc else (os.cpu_count() or 4)
    logger.info(f"  Rendering {len(pairs)} clips ({max_workers} parallel workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_render_panel_clip, a): i for i, a in enumerate(render_args)}
        done = 0
        log_every = max(1, len(pairs) // 10)
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
            done += 1
            if done % log_every == 0 or done == len(pairs):
                logger.info(f"  Rendered {done}/{len(pairs)} clips")

    # Concat list — use filename only; ffmpeg resolves relative to concat.txt's directory
    concat_list = temp_dir / "concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.name}'\n")

    # Build final ffmpeg command
    has_music = music_path.exists()
    if not has_music:
        logger.warning("  No background music — voiceover only")

    cmd = [ffmpeg_exe, "-y",
           "-f", "concat", "-safe", "0", "-i", str(concat_list),
           "-i", str(voiceover_path)]
    if has_music:
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]

    # Audio filter
    if has_music:
        af = (
            f"[1:a]volume={voiceover_volume}[vo];"
            f"[2:a]volume={bg_music_volume},"
            f"atrim=duration={total_duration:.3f}[bg];"
            f"[vo][bg]amix=inputs=2:duration=first[aout]"
        )
    else:
        af = f"[1:a]volume={voiceover_volume}[aout]"

    # Subtitle path — escape Windows drive-letter colon for ffmpeg filter string
    sub_filter = ""
    if captions_path and captions_path.exists():
        srt = str(captions_path).replace("\\", "/")
        srt = re.sub(r"^([A-Za-z]):", r"\1\\:", srt)
        sub_filter = f"subtitles='{srt}'"

    if sub_filter:
        # Subtitle burn-in requires decode → filter → re-encode
        fc = f"[0:v]{sub_filter}[vout];{af}"
        cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]"]
        if use_nvenc:
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    else:
        # No video processing — stream copy avoids a second encode pass
        cmd += ["-filter_complex", af, "-map", "0:v", "-map", "[aout]", "-c:v", "copy"]

    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path)]

    logger.info("  Encoding final video...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Final encode failed:\n{r.stderr[-2000:]}")

    shutil.rmtree(temp_dir, ignore_errors=True)

    if not _verify_audio_stream(output_path):
        raise RuntimeError(f"Output video has no audio track: {output_path}")

    return output_path
