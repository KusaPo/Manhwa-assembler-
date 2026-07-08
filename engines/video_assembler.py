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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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

# Random pool — zoom only, no panning (panning snaps at 1920px without upscaling)
EFFECT_TYPES = ["zoom_in", "zoom_out", "static"]
EFFECT_WEIGHTS = [50, 45, 5]

# Full set of named motions (random pool + content-classifier names)
ALL_MOTIONS = set(EFFECT_TYPES) | {
    "zoom_in_slow", "zoom_in_center", "zoom_out_slow",
    "pan_down", "pan_up", "drift_diagonal",
    "pan_left", "pan_right",
}

# Map content-classifier pan/drift motions → zoom equivalents (smooth without upscaling)
_PAN_TO_ZOOM = {
    "pan_down":       "zoom_in",
    "pan_up":         "zoom_out",
    "drift_diagonal": "zoom_in",
    "zoom_in_slow":   "zoom_in",
    "zoom_out_slow":  "zoom_out",
    "zoom_in_center": "zoom_in",
    "pan_left":       "zoom_in",
    "pan_right":      "zoom_out",
    "static":         "static",
}

# Rotation used for subclips 2+ (different effect from subclip 1 so it doesn't look like a replay)
_SUBCLIP_ROTATION = ["zoom_out", "zoom_in", "static", "zoom_in", "zoom_out"]


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


@dataclass
class _ClipJob:
    ffmpeg_exe: str
    image_path: Path
    duration: float
    out_path: Path
    fps: int
    width: int
    height: int
    zoom_intensity: float
    fade_duration: float
    use_nvenc: bool
    effect: str
    easing: str = "linear"      # "linear" | "ease_in" | "ease_out"
    hold_frames: int = 0        # cliffhanger static-freeze appended after motion
    white_flash: bool = False   # ACTION: white overlay fading out over first 0.15s
    screen_shake: bool = False  # ACTION: sine-wave translate for first 0.30s


def _zoompan_filter(
    effect: str,
    n_frames: int,
    fps: int,
    width: int,
    height: int,
    zoom_intensity: float,
    easing: str = "linear",
) -> str:
    """
    Absolute-expression zoompan (no incremental zoom+step accumulation).
    on is 1-indexed in zoompan, so on/N goes from 1/N to 1.0 over the clip.
    Easing shapes the progress curve without changing start/end zoom values.
    """
    zi = zoom_intensity
    N = max(n_frames, 1)

    # Easing progress: p goes from ~0 (frame 1) to 1.0 (frame N)
    if easing == "ease_in":
        p = f"pow(on/{N},2.6)"        # slow start, rapid acceleration
    elif easing == "ease_out":
        p = f"(1-pow(1-on/{N},2.0))"  # fast start, decelerates to final
    else:
        p = f"on/{N}"                  # linear

    if effect in ("zoom_in", "zoom_in_slow", "zoom_in_center"):
        z_end = 1.0 + zi
        z = f"min(1.0+{zi:.5f}*{p},{z_end:.5f})"
        x = "trunc(iw/2-(iw/zoom/2))"
        y = ("trunc(ih/2-(ih/zoom/2)+(ih*0.05))"
             if effect == "zoom_in_center"
             else "trunc(ih/2-(ih/zoom/2))")
    elif effect in ("zoom_out", "zoom_out_slow"):
        z_start = 1.0 + zi
        z = f"max({z_start:.5f}-{zi:.5f}*{p},1.0)"
        x = "trunc(iw/2-(iw/zoom/2))"
        y = "trunc(ih/2-(ih/zoom/2))"
    else:  # static (and any unmapped motion)
        z = "1"
        x = "0"
        y = "0"

    return f"zoompan=z='{z}':x='{x}':y='{y}':d={N}:s={width}x{height}:fps={fps}"


def _render_panel_clip(job: _ClipJob) -> Path:
    """Render one image to a temp video clip via ffmpeg zoompan (runs in thread pool)."""
    n_frames = max(1, round(job.fps * job.duration))
    zp = _zoompan_filter(
        job.effect, n_frames, job.fps,
        job.width, job.height, job.zoom_intensity, job.easing,
    )

    vf_parts = [zp]

    # White flash: `fade from white` replaces normal fade-in for action panels.
    # fade=t=in:color=white fades FROM white into the clip over flash_dur seconds.
    if job.white_flash:
        flash_dur = round(job.fps * 0.15)  # 4–5 frames @ 30fps
        vf_parts.append(f"fade=t=in:st=0:d={flash_dur / job.fps:.4f}:color=white")
    elif job.fade_duration > 0 and n_frames > round(job.fps * job.fade_duration * 2):
        fade_n = round(job.fps * job.fade_duration)
        vf_parts.append(f"fade=t=in:st=0:d={fade_n / job.fps:.4f}")

    # Normal fade-out (all panels including action)
    if job.fade_duration > 0 and n_frames > round(job.fps * job.fade_duration * 2):
        fade_n = round(job.fps * job.fade_duration)
        vf_parts.append(f"fade=t=out:st={(n_frames - fade_n) / job.fps:.4f}:d={fade_n / job.fps:.4f}")

    # Screen shake: pad frame by 2*amp each side, then crop with sine-wave offset.
    # Uses sin/cos so the shake is visually random without ffmpeg's random() quirks.
    if job.screen_shake:
        s_frames = max(2, round(job.fps * 0.30))  # 9 frames @ 30fps
        amp = 4  # pixels of max displacement
        pad = amp * 2
        vf_parts += [
            f"pad={job.width + pad}:{job.height + pad}:{amp}:{amp}",
            (
                f"crop={job.width}:{job.height}"
                f":x='{amp}+if(lt(n,{s_frames}),round({amp}*sin(n*1.3)),0)'"
                f":y='{amp}+if(lt(n,{s_frames}),round({amp}*cos(n*1.7)),0)'"
            ),
        ]

    # Cliffhanger: freeze last frame for hold_frames after Ken Burns finishes
    if job.hold_frames > 0:
        vf_parts.append(f"tpad=stop_mode=clone:stop={job.hold_frames}")
    vf = ",".join(vf_parts)

    total_frames = n_frames + job.hold_frames

    if job.use_nvenc:
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "8M"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"]

    cmd = [
        job.ffmpeg_exe, "-y",
        "-loop", "1", "-framerate", str(job.fps),
        "-i", str(job.image_path),
        "-vf", vf,
        "-frames:v", str(total_frames),
        *codec_args,
        "-an", "-pix_fmt", "yuv420p",
        str(job.out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Clip render failed ({job.image_path.name}):\n{r.stderr[-800:]}")
    return job.out_path


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
            script_path=self.paths.panel_script_file,
            vo_segments=segments,
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
    script_path: Optional[Path] = None,
    vo_segments=None,   # List[TimestampSegment] for BGM ducking; None = flat volume
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

    # Load motion map if content_classifier has run
    motion_map_path = output_path.parent / "motion_map.json"
    motion_map = {}
    if motion_map_path.exists():
        try:
            motion_map = json.loads(motion_map_path.read_text(encoding="utf-8"))
            logger.info(f"  Content-aware motion map loaded ({len(motion_map)} panels)")
        except Exception:
            pass

    # ---- Style engine: per-panel (effect, easing, hold_frames) ----------------
    try:
        from engines.recap_style_engine import (
            CLASSIFIER_LABEL_MAP, PanelCategory, detect_cliffhanger,
            snap_frames as _snap_frames, FPS as _STYLE_FPS,
        )
        _style_ok = True
    except ImportError:
        _style_ok = False

    # Per-category: (effect, easing)
    # (effect, easing, white_flash, screen_shake)
    _CAT_MOTION = {
        PanelCategory.DIALOGUE_HOLD:    ("zoom_in",  "linear",   False, False),
        PanelCategory.ACTION_ZOOM:      ("zoom_in",  "ease_in",  True,  True),
        PanelCategory.ESTABLISHING_PAN: ("zoom_out", "ease_out", False, False),
        PanelCategory.IMPACT_FLASH:     ("static",   "linear",   True,  False),
    } if _style_ok else {}

    # Load script text for cliffhanger phrase detection
    script_texts: List[str] = []
    if _style_ok and script_path and script_path.exists():
        try:
            sd = json.loads(script_path.read_text(encoding="utf-8"))
            script_texts = [e.get("narration", "") for e in sd]
        except Exception:
            pass

    def _pick_clip_params(img_path: Path, pair_idx: int, subclip_idx: int):
        """Return (effect, easing, hold_frames) for this clip."""
        hold = 0
        # Cliffhanger: last panel of script, or phrase match
        if _style_ok and script_texts:
            txt = script_texts[pair_idx] if pair_idx < len(script_texts) else ""
            if detect_cliffhanger(txt) or pair_idx == len(script_texts) - 1:
                hold = _snap_frames(2.0)  # 60 frames @ 30fps

        # Subclips 2+ rotate effect; no FX (flash/shake on first subclip only)
        if subclip_idx > 0:
            return _SUBCLIP_ROTATION[(subclip_idx - 1) % len(_SUBCLIP_ROTATION)], "linear", hold, False, False

        # Content-aware effect + easing + FX from style engine
        if _style_ok and motion_map:
            try:
                panel_num = int(img_path.stem.split("-")[0])
                map_entry = motion_map.get(str(panel_num), {})
                cat_label = map_entry.get("category", "")
                cat = CLASSIFIER_LABEL_MAP.get(cat_label)
                if cat and cat in _CAT_MOTION:
                    eff, eas, wf, ss = _CAT_MOTION[cat]
                    return eff, eas, hold, wf, ss
            except (ValueError, IndexError):
                pass

        # Fallback: content-map motion remapped to zoom, linear easing, no FX
        try:
            panel_num = int(img_path.stem.split("-")[0])
            entry = motion_map.get(str(panel_num))
            if entry and entry.get("motion") in ALL_MOTIONS:
                return _PAN_TO_ZOOM.get(entry["motion"], "zoom_in"), "linear", hold, False, False
        except (ValueError, IndexError):
            pass

        return random.choices(EFFECT_TYPES, weights=EFFECT_WEIGHTS, k=1)[0], "linear", hold, False, False

    # Detect consecutive same-panel runs (subclip splits)
    prev_path = None
    sc_idx = 0
    subclip_counters: List[int] = []
    for img, _ in pairs:
        sc_idx = (sc_idx + 1) if img == prev_path else 0
        subclip_counters.append(sc_idx)
        prev_path = img

    # Parallel clip rendering
    temp_dir = output_path.parent / "_temp_clips"
    temp_dir.mkdir(exist_ok=True)

    clip_jobs: List[_ClipJob] = []
    for i, (img, dur) in enumerate(pairs):
        eff, eas, hold, wf, ss = _pick_clip_params(img, i, subclip_counters[i])
        clip_jobs.append(_ClipJob(
            ffmpeg_exe=ffmpeg_exe, image_path=img, duration=dur,
            out_path=temp_dir / f"clip_{i:04d}.mp4",
            fps=fps, width=width, height=height,
            zoom_intensity=zoom_intensity, fade_duration=fade_duration,
            use_nvenc=use_nvenc, effect=eff, easing=eas, hold_frames=hold,
            white_flash=wf, screen_shake=ss,
        ))
    clip_paths = [j.out_path for j in clip_jobs]
    max_workers = NVENC_MAX_WORKERS if use_nvenc else (os.cpu_count() or 4)
    if _style_ok and motion_map:
        logger.info("  Style engine active (easing + cliffhanger detection)")
    logger.info(f"  Rendering {len(clip_jobs)} clips ({max_workers} parallel workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_render_panel_clip, j): i for i, j in enumerate(clip_jobs)}
        done = 0
        log_every = max(1, len(pairs) // 10)
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
            done += 1
            if done % log_every == 0 or done == len(clip_jobs):
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

    # Audio filter — BGM ducking if style engine + VO segment times are available
    if has_music:
        try:
            from engines.recap_style_engine import build_duck_volume_expr
            duck_expr = build_duck_volume_expr(
                [(s.start, s.end) for s in vo_segments]
            ) if vo_segments else f"volume={bg_music_volume}"
            logger.info("  BGM ducking active (-8dB under voiceover)")
        except Exception:
            duck_expr = f"volume={bg_music_volume}"
        af = (
            f"[1:a]volume={voiceover_volume}[vo];"
            f"[2:a]{duck_expr},"
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
