"""
FFmpeg Renderer — frame-exact, jitter-free assembly from sync_sheet_audio.csv.

Fixes two bugs in the zoompan-based renderer:

1. SHAKY ZOOM/PAN
   Root cause: zoompan rounds output coordinates to integer pixels. Slow pans
   snap rigidly frame-to-frame instead of gliding.
   Fix: pre-scale the source image by UPSCALE_FACTOR (default 4x → 7680 wide)
   using lanczos. zoompan operates on the upscaled source, so a 1-px integer
   step in output = 0.25 sub-pixels of motion at 4x. At 16x: 0 snap frames.

2. GHOSTING / REAPPEARING PANELS AT CUTS
   Root cause: float durations from the CSV don't land on frame boundaries.
   Clips bleed frames at seams.
   Fix: convert every clip's END time to an absolute frame number on the master
   timeline. Render with -frames:v <exact>. Concat with -c copy — no re-encode,
   no seam interpolation, no frame bleed.
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import imageio_ffmpeg

logger = logging.getLogger("engines.ffmpeg_renderer")

_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

FPS_DEFAULT = 30
TARGET_W = 1920
TARGET_H = 1080

# UPSCALE_FACTOR smoothness vs speed:
#   4x  (default) — CV 0.29, ~16% snap frames. Fast. Good for previews.
#   8x            — CV 0.21, ~5% snap. ~2.5x slower. Production quality.
#   16x           — CV 0.11, ZERO snap frames. ~5-6x slower. Final renders.
UPSCALE_FACTOR = 4

CRF = 18
PRESET = "fast"
VIDEO_CODEC = "h264_nvenc"
PIX_FMT = "yuv420p"

MOTIONS = {
    "zoom_in_slow":   ("min(1.0+(on/{D})*0.20,1.20)", "iw/2-(iw/zoom/2)",                       "ih/2-(ih/zoom/2)"),
    "zoom_out_slow":  ("max(1.20-(on/{D})*0.20,1.00)", "iw/2-(iw/zoom/2)",                       "ih/2-(ih/zoom/2)"),
    "pan_down":       ("1.10",                         "iw/2-(iw/zoom/2)",                       "(ih-ih/zoom)*(on/{D})"),
    "pan_up":         ("1.10",                         "iw/2-(iw/zoom/2)",                       "(ih-ih/zoom)*(1-on/{D})"),
    "zoom_in_center": ("min(1.0+(on/{D})*0.25,1.25)",  "iw/2-(iw/zoom/2)",                       "ih/2-(ih/zoom/2)+(ih*0.05)"),
    "drift_diagonal": ("min(1.0+(on/{D})*0.15,1.15)",  "iw/2-(iw/zoom/2)+(iw*0.05)*(on/{D})",   "ih/2-(ih/zoom/2)+(ih*0.05)*(on/{D})"),
    "static":         ("1.00",                         "iw/2-(iw/zoom/2)",                       "ih/2-(ih/zoom/2)"),
}

HOLD_ROTATION = [
    "zoom_in_slow", "pan_down", "zoom_in_center",
    "pan_up", "drift_diagonal", "zoom_out_slow",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ClipPlan:
    order: int
    panel_path: Path
    start_frame: int
    end_frame: int
    role: str
    sentence_idx: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration_s(self) -> float:
        return self.frame_count / FPS_DEFAULT


# ---------------------------------------------------------------------------
# Planning — absolute frame anchors, zero accumulated drift
# ---------------------------------------------------------------------------

def plan_clips_from_csv(
    csv_path: Path,
    panels_dir: Path,
    fps: int = FPS_DEFAULT,
) -> List[ClipPlan]:
    """Convert sync_sheet_audio.csv rows to frame-exact ClipPlans."""
    plans: List[ClipPlan] = []
    prev_end_frame = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            order = int(row["order"])
            end_frame = round(float(row["end_s"]) * fps)
            if end_frame <= prev_end_frame:
                logger.warning(f"  Clip {order}: end_frame {end_frame} <= prev, forcing +1")
                end_frame = prev_end_frame + 1

            panel_path = panels_dir / row["file"]
            if not panel_path.exists():
                webp = panel_path.with_suffix(".webp")
                if webp.exists():
                    panel_path = webp
                else:
                    logger.warning(f"  Panel not found: {panel_path.name}, skipping")
                    prev_end_frame = end_frame
                    continue

            plans.append(ClipPlan(
                order=order,
                panel_path=panel_path,
                start_frame=prev_end_frame,
                end_frame=end_frame,
                role=row["role"],
                sentence_idx=int(row["sentence_indices"]),
            ))
            prev_end_frame = end_frame

    if plans:
        total_frames = plans[-1].end_frame
        logger.info(
            f"  Planned {len(plans)} clips, {total_frames} frames "
            f"({total_frames / fps:.2f}s at {fps}fps)"
        )
    return plans


# ---------------------------------------------------------------------------
# Motion selection
# ---------------------------------------------------------------------------

def pick_motion(plan: ClipPlan, hold_index: int) -> str:
    if plan.role == "flash":
        return "static"
    if plan.duration_s < 2.0:
        return "zoom_in_slow"
    return HOLD_ROTATION[hold_index % len(HOLD_ROTATION)]


# ---------------------------------------------------------------------------
# Per-clip rendering
# ---------------------------------------------------------------------------

def render_single_clip(
    plan: ClipPlan,
    motion: str,
    output_path: Path,
    fps: int = FPS_DEFAULT,
    upscale: int = UPSCALE_FACTOR,
) -> None:
    """Render one frame-exact clip with smooth Ken Burns via upscaled zoompan."""
    src_w = TARGET_W * upscale
    D = plan.frame_count

    z_expr, x_expr, y_expr = MOTIONS[motion]
    z_expr = z_expr.format(D=D)
    x_expr = x_expr.format(D=D)
    y_expr = y_expr.format(D=D)

    vf = (
        f"scale={src_w}:-2:flags=lanczos,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={D}:s={TARGET_W}x{TARGET_H}:fps={fps},"
        f"format={PIX_FMT}"
    )

    if VIDEO_CODEC == "h264_nvenc":
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p2", "-b:v", "8M"]
    else:
        codec_args = ["-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", str(CRF)]

    cmd = [
        _ffmpeg, "-y", "-loglevel", "error",
        "-loop", "1",
        "-i", str(plan.panel_path),
        "-frames:v", str(D),
        "-vf", vf,
        *codec_args,
        "-r", str(fps),
        "-pix_fmt", PIX_FMT,
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Clip {plan.order} ({plan.panel_path.name}) failed:\n{r.stderr[-800:]}"
        )


# ---------------------------------------------------------------------------
# Lossless concat
# ---------------------------------------------------------------------------

def concat_clips_lossless(clip_paths: List[Path], output_path: Path) -> None:
    """Concat clips with -c copy — no re-encode, no seam interpolation."""
    concat_list = output_path.parent / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    cmd = [
        _ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    concat_list.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"Concat failed:\n{r.stderr[-800:]}")


# ---------------------------------------------------------------------------
# Audio + caption mux
# ---------------------------------------------------------------------------

def mux_audio_and_captions(
    video_path: Path,
    voiceover_path: Path,
    music_path: Optional[Path],
    captions_path: Optional[Path],
    output_path: Path,
    fps: int = FPS_DEFAULT,
    voiceover_volume: float = 1.0,
    music_volume: float = 0.15,
) -> None:
    inputs = ["-i", str(video_path), "-i", str(voiceover_path)]
    if music_path and music_path.exists():
        inputs += ["-stream_loop", "-1", "-i", str(music_path)]

    filters = []
    if captions_path and captions_path.exists():
        import re
        srt = re.sub(r"^([A-Za-z]):", r"\1\\:", str(captions_path).replace("\\", "/"))
        filters.append(f"[0:v]subtitles='{srt}'[v]")
        v_label = "[v]"
    else:
        v_label = "0:v"  # bare stream specifier, not a filter label

    if music_path and music_path.exists():
        filters.append(
            f"[1:a]volume={voiceover_volume}[a1];"
            f"[2:a]volume={music_volume}[a2];"
            f"[a1][a2]amix=inputs=2:duration=first[a]"
        )
    else:
        filters.append(f"[1:a]volume={voiceover_volume}[a]")

    if VIDEO_CODEC == "h264_nvenc":
        vcodec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
    else:
        vcodec_args = ["-c:v", VIDEO_CODEC, "-preset", PRESET, "-crf", str(CRF)]

    cmd = [
        _ffmpeg, "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", v_label, "-map", "[a]",
        *vcodec_args,
        "-r", str(fps),
        "-pix_fmt", PIX_FMT,
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Mux failed:\n{r.stderr[-800:]}")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def render_video(
    csv_path: Path,
    panels_dir: Path,
    voiceover_path: Path,
    output_path: Path,
    music_path: Optional[Path] = None,
    captions_path: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    fps: int = FPS_DEFAULT,
    upscale: int = UPSCALE_FACTOR,
    keep_intermediate: bool = False,
) -> Path:
    """
    Full pipeline: CSV → frame-exact clips → lossless concat → audio/caption mux.

    upscale: 4=fast preview, 8=production, 16=max smooth (zero snap frames).
    """
    work_dir = work_dir or output_path.parent / "_render_work"
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    plans = plan_clips_from_csv(csv_path, panels_dir, fps=fps)
    if not plans:
        raise RuntimeError("No clips to render — empty CSV or all panels missing")

    logger.info(f"  Rendering {len(plans)} clips at {upscale}x upscale...")
    clip_paths: List[Path] = []
    hold_idx = 0
    for plan in plans:
        motion = pick_motion(plan, hold_idx)
        if plan.role == "hold":
            hold_idx += 1
        clip_out = clips_dir / f"clip_{plan.order:04d}.mp4"
        render_single_clip(plan, motion, clip_out, fps=fps, upscale=upscale)
        clip_paths.append(clip_out)
        if len(clip_paths) % 20 == 0:
            logger.info(f"  Rendered {len(clip_paths)}/{len(plans)} clips")

    logger.info(f"  Concatenating {len(clip_paths)} clips...")
    video_only = work_dir / "video_only.mp4"
    concat_clips_lossless(clip_paths, video_only)

    logger.info("  Muxing audio...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mux_audio_and_captions(
        video_only, voiceover_path, music_path, captions_path, output_path, fps=fps,
    )

    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(f"  Done → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Frame-exact FFmpeg renderer")
    p.add_argument("--csv",        type=Path, required=True)
    p.add_argument("--panels",     type=Path, required=True)
    p.add_argument("--voiceover",  type=Path, required=True)
    p.add_argument("--music",      type=Path, default=None)
    p.add_argument("--captions",   type=Path, default=None)
    p.add_argument("--output",     type=Path, required=True)
    p.add_argument("--fps",        type=int,  default=FPS_DEFAULT)
    p.add_argument("--upscale",    type=int,  default=UPSCALE_FACTOR,
                   help="4=fast preview, 8=production, 16=max smooth")
    p.add_argument("--keep-intermediate", action="store_true")
    args = p.parse_args()

    render_video(
        csv_path=args.csv,
        panels_dir=args.panels,
        voiceover_path=args.voiceover,
        output_path=args.output,
        music_path=args.music,
        captions_path=args.captions,
        fps=args.fps,
        upscale=args.upscale,
        keep_intermediate=args.keep_intermediate,
    )
