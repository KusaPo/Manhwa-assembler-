"""
recap_style_engine.py
=====================
Style engine for the Manhwa Assembler pipeline.

Implements the reverse-engineered visual grammar of the target recap channel
as hard numerical constraints. This module owns THREE responsibilities:

  1. STYLE SPEC   — frozen per-category motion parameters (duration, zoom,
                    pan, easing, FX) as dataclasses.
  2. TIMELINE     — converts a sync plan (sentence-timed panel assignments
                    from build_sync_plan_from_script()) into a frame-exact
                    cut list, applying pacing rules, climax scaling,
                    cliffhanger overrides, and the 0.15s audio lead-lag.
  3. FILTERGRAPH  — emits per-clip FFmpeg filter chains (zoompan with
                    absolute frame anchoring + supersampling, white-flash,
                    screen shake, chromatic aberration, dblur) plus the
                    transition and BGM ducking graphs.

Integration points with existing pipeline:
  - content_classifier.py  -> maps its motion category output to
                              PanelCategory via CLASSIFIER_LABEL_MAP.
  - build_sync_plan_from_script() -> feed its output to build_cut_list().
  - ffmpeg_renderer.py     -> replace per-clip filter construction with
                              build_clip_filtergraph(); keeps the 4x lanczos
                              supersample + absolute-frame-anchor fixes.

All frame constants are DERIVED from time values via snap_frames(), so the
FPS constant below is the single source of truth. The reference channel's
spec was authored at 30fps; at 60fps every time-domain rule (0.15s lead,
0.4s crossfade, 0.8–1.2s climax window) is preserved exactly — only the
frame counts double.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================================
# SECTION 0 — FRAME MATH (60 FPS, half-up rounding)
# ============================================================================

FPS = 30
FRAME_DUR = 1.0 / FPS

# Supersample factor for zoompan jitter elimination (existing fix: 4x lanczos).
# Raise to 8/16 for very small source panels.
SUPERSAMPLE = 4


def snap_frames(seconds: float) -> int:
    """
    duration_frames = round(raw_duration_seconds * 30)

    IMPORTANT: Python's built-in round() is banker's rounding
    (round(4.5) == 4), which silently drops a frame on every *.x5 boundary
    and accumulates audio drift across a 10-minute video. We use explicit
    half-up rounding instead.
    """
    return int(math.floor(seconds * FPS + 0.5))


def snap_seconds(seconds: float) -> float:
    """Snap a float duration to the nearest exact frame boundary in seconds."""
    return snap_frames(seconds) * FRAME_DUR


# ============================================================================
# SECTION 1 — STYLE SPEC (frozen numerical baselines)
# ============================================================================

class PanelCategory(Enum):
    DIALOGUE_HOLD = "dialogue_reaction_hold"
    ACTION_ZOOM = "action_impact_zoom"
    ESTABLISHING_PAN = "establishing_scenery_pan"
    IMPACT_FLASH = "impact_flash_insert"


class Easing(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"     # slow start -> rapid acceleration, hard stop
    EASE_OUT = "ease_out"   # decelerates into the final frame
    STATIC = "static"


class PanMode(Enum):
    DRIFT = "drift"           # subtle diagonal/horizontal, direction free
    IMPACT_TRACK = "impact"   # fixed direction along the blow vector
    SCROLL = "scroll"         # vertical top->bottom or horizontal tracking
    STATIC = "static"


@dataclass(frozen=True)
class MotionSpec:
    category: PanelCategory
    hold_default: float          # seconds
    hold_min: float
    hold_max: float
    zoom_start: float
    zoom_end: float
    pan_mode: PanMode
    pan_distance: float          # fraction of frame dimension (0.05 = 5%)
    easing: Easing
    # FX flags
    white_flash: bool = False
    white_flash_dur: float = 0.0       # seconds
    screen_shake: bool = False
    shake_amp_px: int = 0              # output-resolution pixels
    shake_dur: float = 0.0             # seconds
    motion_blur: bool = False
    motion_blur_radius: int = 0        # px
    motion_blur_dur: float = 0.0       # seconds (from clip start)
    chromatic_aberration: bool = False
    ca_strength: float = 0.0           # 0.10 = 10% radial
    whip_blur: bool = False
    whip_blur_dur: float = 0.0         # seconds


STYLE_SPECS: dict[PanelCategory, MotionSpec] = {
    PanelCategory.DIALOGUE_HOLD: MotionSpec(
        category=PanelCategory.DIALOGUE_HOLD,
        hold_default=3.5, hold_min=2.5, hold_max=5.0,
        zoom_start=1.00, zoom_end=1.08,
        pan_mode=PanMode.DRIFT, pan_distance=0.05,
        easing=Easing.LINEAR,
    ),
    PanelCategory.ACTION_ZOOM: MotionSpec(
        category=PanelCategory.ACTION_ZOOM,
        hold_default=1.5, hold_min=0.8, hold_max=2.2,
        zoom_start=1.05, zoom_end=1.35,
        pan_mode=PanMode.IMPACT_TRACK, pan_distance=0.12,
        easing=Easing.EASE_IN,
        white_flash=True, white_flash_dur=0.15,
        screen_shake=True, shake_amp_px=4, shake_dur=0.30,
    ),
    PanelCategory.ESTABLISHING_PAN: MotionSpec(
        category=PanelCategory.ESTABLISHING_PAN,
        hold_default=5.5, hold_min=4.0, hold_max=7.0,
        zoom_start=1.10, zoom_end=1.00,
        pan_mode=PanMode.SCROLL, pan_distance=0.25,
        easing=Easing.EASE_OUT,
        motion_blur=True, motion_blur_radius=3, motion_blur_dur=0.5,
    ),
    PanelCategory.IMPACT_FLASH: MotionSpec(
        category=PanelCategory.IMPACT_FLASH,
        hold_default=0.25, hold_min=0.15, hold_max=0.35,
        zoom_start=1.20, zoom_end=1.20,
        pan_mode=PanMode.STATIC, pan_distance=0.0,
        easing=Easing.STATIC,
        chromatic_aberration=True, ca_strength=0.10,
        whip_blur=True, whip_blur_dur=0.05,
    ),
}

# Map content_classifier.py output labels -> PanelCategory.
# Extend/adjust to match the classifier's actual label vocabulary.
CLASSIFIER_LABEL_MAP: dict[str, PanelCategory] = {
    # original spec labels
    "dialogue":       PanelCategory.DIALOGUE_HOLD,
    "reaction":       PanelCategory.DIALOGUE_HOLD,
    "monologue":      PanelCategory.DIALOGUE_HOLD,
    "action":         PanelCategory.ACTION_ZOOM,
    "impact":         PanelCategory.ACTION_ZOOM,
    "system_screen":  PanelCategory.ACTION_ZOOM,
    "establishing":   PanelCategory.ESTABLISHING_PAN,
    "scenery":        PanelCategory.ESTABLISHING_PAN,
    "wide_shot":      PanelCategory.ESTABLISHING_PAN,
    "flash":          PanelCategory.IMPACT_FLASH,
    "montage_cut":    PanelCategory.IMPACT_FLASH,
    # content_classifier.py labels
    "face_closeup":   PanelCategory.DIALOGUE_HOLD,
    "group_scene":    PanelCategory.DIALOGUE_HOLD,
    "action_impact":  PanelCategory.ACTION_ZOOM,
    "wide_establish": PanelCategory.ESTABLISHING_PAN,
    "detail_narrow":  PanelCategory.DIALOGUE_HOLD,
    "static_quiet":   PanelCategory.IMPACT_FLASH,
}


# ============================================================================
# SECTION 2 — PACING CONSTANTS
# ============================================================================

class PacingMode(Enum):
    STANDARD = "standard"   # storytelling: target ASL 2.8s
    ACTION = "action"       # combat: target ASL 1.1s
    CLIMAX = "climax"       # peak: cut every 24–36 frames (250% frequency)


PACING_TARGET_ASL = {
    PacingMode.STANDARD: 2.8,
    PacingMode.ACTION: 1.1,
}

# Climax: source spec said "every 24–36 frames" at the channel's 30fps
# timebase — that is a TIME rule (0.8s–1.2s), so derive frames from seconds.
# At 60fps this becomes 48–72 frames; the on-screen pacing is identical.
CLIMAX_MIN_SEC = 0.8
CLIMAX_MAX_SEC = 1.2
CLIMAX_MIN_FRAMES = snap_frames(CLIMAX_MIN_SEC)   # 48 @ 60fps
CLIMAX_MAX_FRAMES = snap_frames(CLIMAX_MAX_SEC)   # 72 @ 60fps

# Audio sync: visual cut leads the sentence-end by exactly 0.15s.
AUDIO_LEAD_SEC = 0.15
AUDIO_LEAD_FRAMES = snap_frames(AUDIO_LEAD_SEC)   # 9 @ 60fps

# Cliffhanger / reveal: force static hold on final frame before next cut.
CLIFFHANGER_HOLD_SEC = 2.0
CLIFFHANGER_HOLD_FRAMES = snap_frames(CLIFFHANGER_HOLD_SEC)  # 120 @ 60fps

# Transitions
HARD_CUT_RATIO = 0.92
CROSSFADE_SEC = 0.4                   # spec: "12 frames / 0.4 seconds" — the
CROSSFADE_FRAMES = snap_frames(CROSSFADE_SEC)  # time is canonical: 24 @ 60fps

# BGM ducking
DUCK_DB = -8.0
DUCK_GAIN = 10 ** (DUCK_DB / 20.0)    # ≈ 0.3981
DUCK_THRESHOLD_DB = -20.0
DUCK_ATTACK_SEC = 0.05                # ramp in/out so ducking isn't a click
DUCK_RELEASE_SEC = 0.25

# Cliffhanger trigger phrases (lowercase substring match against script text)
CLIFFHANGER_PHRASES = (
    "to be continued", "little did", "what he didn't know", "system:",
    "level up", "awakening", "but then", "that was when", "until now",
    "skill acquired", "hidden quest", "class change",
)


# ============================================================================
# SECTION 3 — TIMELINE / CUT LIST
# ============================================================================

@dataclass
class PanelJob:
    """One classified panel entering the assembler."""
    panel_path: str
    category: PanelCategory
    src_w: int
    src_h: int
    # Optional overrides from the sync plan / classifier:
    sentence_end: Optional[float] = None   # VO sentence boundary this panel rides
    pan_angle_deg: Optional[float] = None  # impact direction, 0=right, 90=up
    is_cliffhanger: bool = False
    is_scene_change: bool = False          # major location shift -> crossfade
    scene_id: Optional[str] = None
    script_text: str = ""                  # narration text for phrase detection


@dataclass
class Cut:
    """One resolved clip on the frame-exact timeline."""
    panel: PanelJob
    start_frame: int
    n_frames: int
    spec: MotionSpec
    transition_in: str = "cut"             # "cut" | "crossfade" | "flash"
    sfx_frame: Optional[int] = None        # absolute frame for SFX hit
    cliffhanger_hold_frames: int = 0

    @property
    def duration_sec(self) -> float:
        return self.n_frames * FRAME_DUR

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.n_frames


def _clamp_to_spec(seconds: float, spec: MotionSpec) -> float:
    return max(spec.hold_min, min(spec.hold_max, seconds))


def detect_cliffhanger(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in CLIFFHANGER_PHRASES)


def resolve_duration(
    panel: PanelJob,
    spec: MotionSpec,
    pacing: PacingMode,
    rng: random.Random,
) -> int:
    """
    Resolve a panel's hold duration in FRAMES, applying:
      1. category default + clamp range
      2. pacing-mode ASL pull (standard 2.8s / action 1.1s)
      3. climax scaling (24–36 frame hard window)
      4. audio-boundary lead (handled at the cut level, see build_cut_list)
    """
    if pacing is PacingMode.CLIMAX and spec.category in (
        PanelCategory.ACTION_ZOOM, PanelCategory.IMPACT_FLASH
    ):
        # 250% cut frequency: every 24–36 frames, spec ranges overridden
        # except IMPACT_FLASH which is already sub-second.
        if spec.category is PanelCategory.IMPACT_FLASH:
            return snap_frames(_clamp_to_spec(spec.hold_default, spec))
        return rng.randint(CLIMAX_MIN_FRAMES, CLIMAX_MAX_FRAMES)

    # Pull the category default toward the mode's target ASL, then clamp.
    target_asl = PACING_TARGET_ASL.get(pacing, PACING_TARGET_ASL[PacingMode.STANDARD])
    # 60/40 blend keeps category identity while honoring section pacing.
    blended = 0.6 * spec.hold_default + 0.4 * target_asl
    # ±10% jitter so cuts don't feel metronomic.
    blended *= rng.uniform(0.9, 1.1)
    return snap_frames(_clamp_to_spec(blended, spec))


def build_cut_list(
    panels: list[PanelJob],
    pacing_by_index: Optional[dict[int, PacingMode]] = None,
    seed: int = 1337,
) -> list[Cut]:
    """
    Convert classified, script-ordered panels into a frame-exact cut list.

    AUDIO SYNC RULE: when a panel is tied to a VO sentence boundary
    (panel.sentence_end), its cut is retimed so the NEXT panel starts
    exactly AUDIO_LEAD_FRAMES (0.15s) BEFORE that boundary. Cuts therefore
    never land on/after a sentence end.

    CLIFFHANGER RULE: overrides duration and appends a 2.0s static hold
    on the final frame (rendered by freezing the last frame — see
    build_clip_filtergraph's tpad handling).
    """
    rng = random.Random(seed)
    pacing_by_index = pacing_by_index or {}
    cuts: list[Cut] = []
    cursor = 0  # absolute frame cursor

    for i, panel in enumerate(panels):
        spec = STYLE_SPECS[panel.category]
        pacing = pacing_by_index.get(i, PacingMode.STANDARD)

        n = resolve_duration(panel, spec, pacing, rng)

        # ---- Audio boundary lead-lag -------------------------------------
        if panel.sentence_end is not None:
            boundary_frame = snap_frames(panel.sentence_end)
            desired_end = boundary_frame - AUDIO_LEAD_FRAMES
            if desired_end > cursor:
                n = desired_end - cursor
                # Re-clamp into the category's legal window if the sentence
                # is pathologically long/short; audio wins ties by splitting
                # the panel is out of scope here, so clamp and log upstream.
                n = max(snap_frames(spec.hold_min),
                        min(snap_frames(spec.hold_max) if not panel.is_cliffhanger else n, n))

        # ---- Cliffhanger override ----------------------------------------
        hold = 0
        is_cliff = panel.is_cliffhanger or detect_cliffhanger(panel.script_text)
        if is_cliff:
            hold = CLIFFHANGER_HOLD_FRAMES

        # ---- Transition in -----------------------------------------------
        if panel.category is PanelCategory.IMPACT_FLASH:
            transition = "flash"          # masks montage seams, not a hard cut
        elif panel.is_scene_change:
            transition = "crossfade"      # exactly 12 frames, scene shifts only
        else:
            transition = "cut"            # the 92% default

        # ---- SFX anchor ----------------------------------------------------
        sfx = None
        if spec.category is PanelCategory.ACTION_ZOOM:
            sfx = cursor  # frame 0 of the clip: white-flash + shake init

        cuts.append(Cut(
            panel=panel,
            start_frame=cursor,
            n_frames=n + hold,
            spec=spec,
            transition_in=transition,
            sfx_frame=sfx,
            cliffhanger_hold_frames=hold,
        ))
        cursor += n + hold
        # Crossfades overlap the previous clip; pull the cursor back so total
        # runtime stays honest and both clips get tpad'd in the xfade graph.
        if transition == "crossfade" and len(cuts) > 1:
            cursor -= CROSSFADE_FRAMES

    return cuts


def validate_transition_budget(cuts: list[Cut]) -> dict:
    """Enforce the 92% hard-cut ratio. Call in CI / pre-render check."""
    total = max(1, len(cuts) - 1)
    hard = sum(1 for c in cuts[1:] if c.transition_in == "cut")
    ratio = hard / total
    return {
        "hard_cut_ratio": round(ratio, 4),
        "passes": ratio >= HARD_CUT_RATIO,
        "hard": hard,
        "crossfades": sum(1 for c in cuts[1:] if c.transition_in == "crossfade"),
        "flash_inserts": sum(1 for c in cuts[1:] if c.transition_in == "flash"),
        "total_transitions": total,
    }


# ============================================================================
# SECTION 4 — EASING EXPRESSIONS (ffmpeg expr, absolute frame anchored)
# ============================================================================
#
# All expressions use `on` (output frame number of zoompan) against an
# explicit total N, never `time`/`t` — this preserves the absolute frame
# anchoring fix that killed the frame-boundary ghosting.
#
#   p = progress in [0,1] at frame `on` of an N-frame clip
#     LINEAR   : p = on/(N-1)
#     EASE_IN  : p = (on/(N-1))^2.6   (slow start, rapid accel, hard stop)
#     EASE_OUT : p = 1-(1-on/(N-1))^2 (decelerates into the final frame)
# ============================================================================

EASE_IN_POWER = 2.6   # tuned: 2.0 feels soft, 3.0 overshoots perceptually
EASE_OUT_POWER = 2.0


def progress_expr(easing: Easing, n_frames: int) -> str:
    n1 = max(1, n_frames - 1)
    t = f"(on/{n1})"
    if easing is Easing.LINEAR:
        return t
    if easing is Easing.EASE_IN:
        return f"pow({t},{EASE_IN_POWER})"
    if easing is Easing.EASE_OUT:
        return f"(1-pow(1-{t},{EASE_OUT_POWER}))"
    return "0"  # STATIC


# ============================================================================
# SECTION 5 — PER-CLIP FILTERGRAPH BUILDER
# ============================================================================

@dataclass
class RenderTarget:
    width: int = 1920
    height: int = 1080


def _pan_vector(spec: MotionSpec, panel: PanelJob, rng: random.Random) -> tuple[float, float]:
    """
    Returns (dx, dy) as fractions of the pannable range, signed.
    Positive dx pans right->left content motion (viewport moves right).
    """
    d = spec.pan_distance
    if spec.pan_mode is PanMode.STATIC or d == 0:
        return (0.0, 0.0)
    if spec.pan_mode is PanMode.DRIFT:
        # subtle diagonal or horizontal; pick per-panel deterministically
        ang = rng.choice([0.0, 20.0, -20.0, 160.0, 200.0])
        return (d * math.cos(math.radians(ang)), -d * math.sin(math.radians(ang)))
    if spec.pan_mode is PanMode.IMPACT_TRACK:
        ang = panel.pan_angle_deg if panel.pan_angle_deg is not None else 0.0
        return (d * math.cos(math.radians(ang)), -d * math.sin(math.radians(ang)))
    if spec.pan_mode is PanMode.SCROLL:
        if panel.src_h > panel.src_w * 1.4:      # tall webtoon strip
            return (0.0, d)                       # top -> bottom
        return (rng.choice([-1.0, 1.0]) * d, 0.0)  # horizontal tracking
    return (0.0, 0.0)


def build_clip_filtergraph(
    cut: Cut,
    target: RenderTarget = RenderTarget(),
    rng: Optional[random.Random] = None,
) -> str:
    """
    Emit the full -vf chain for one panel clip (single image input -> video).

    Chain order (order matters):
      scale(SUPERSAMPLE, lanczos)  -> zoompan jitter fix
      zoompan(absolute-frame expr) -> Ken Burns motion
      scale(target)                -> back to output res
      [category FX: shake, flash, dblur, CA, whip]
      tpad(cliffhanger hold)       -> static final-frame freeze
      fps=30, format=yuv420p
    """
    rng = rng or random.Random(hash(cut.panel.panel_path) & 0xFFFF)
    spec = cut.spec
    W, H = target.width, target.height
    motion_frames = cut.n_frames - cut.cliffhanger_hold_frames
    N = max(1, motion_frames)
    p = progress_expr(spec.easing, N)

    up_w = W * SUPERSAMPLE
    up_h = H * SUPERSAMPLE

    # ---- zoom expression ----------------------------------------------------
    zs, ze = spec.zoom_start, spec.zoom_end
    if spec.easing is Easing.STATIC:
        z_expr = f"{zs:.4f}"
    else:
        z_expr = f"{zs:.4f}+({ze - zs:.4f})*{p}"

    # ---- pan expressions ------------------------------------------------------
    dx, dy = _pan_vector(spec, cut.panel, rng)
    # zoompan pannable range: iw - iw/zoom (in supersampled px). Center-anchored
    # start, drifting along (dx, dy) fractions of that range.
    x_expr = f"(iw-iw/zoom)/2 + ({dx:.4f})*(iw-iw/zoom)*{p}" if dx else "(iw-iw/zoom)/2"
    y_expr = f"(ih-ih/zoom)/2 + ({dy:.4f})*(ih-ih/zoom)*{p}" if dy else "(ih-ih/zoom)/2"

    chain: list[str] = [
        f"scale={up_w}:{up_h}:flags=lanczos:force_original_aspect_ratio=increase",
        f"crop={up_w}:{up_h}",
        (
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
            f":d={N}:s={W}x{H}:fps={FPS}"
        ),
    ]

    # ---- Screen shake (Action): ±4px random translation, first 0.30s ---------
    if spec.screen_shake:
        shake_frames = snap_frames(spec.shake_dur)          # 9 @ 30fps
        a = spec.shake_amp_px
        pad = a * 2
        chain += [
            f"pad={W + pad}:{H + pad}:{a}:{a}",
            (
                f"crop={W}:{H}"
                f":x='{a}+if(lt(n,{shake_frames}),round(random(1)*{2*a}-{a}),0)'"
                f":y='{a}+if(lt(n,{shake_frames}),round(random(2)*{2*a}-{a}),0)'"
            ),
        ]

    # ---- Directional motion blur (Establishing): 3px, first 0.5s -------------
    if spec.motion_blur:
        mb_frames = snap_frames(spec.motion_blur_dur)       # 15
        ang = math.degrees(math.atan2(-dy, dx)) if (dx or dy) else 0.0
        chain.append(
            f"dblur=angle={ang:.1f}:radius={spec.motion_blur_radius}"
            f":enable='lt(n,{mb_frames})'"
        )

    # ---- Whip blur (Impact Flash): 0.05s -> 3 frames @ 60fps ------------------
    if spec.whip_blur:
        wb_frames = max(2, snap_frames(spec.whip_blur_dur))  # floor of 2 frames
        chain.append(f"dblur=angle=0:radius=12:enable='lt(n,{wb_frames})'")

    # ---- Radial chromatic aberration (Impact Flash): 10% edges ----------------
    if spec.chromatic_aberration:
        k = spec.ca_strength * 0.5  # lenscorrection k1 scaling for edge-only CA
        chain += [
            "split=3[car][cag][cab]",
        ]
        # NOTE: this branch returns a complex graph, handled below.
        return _assemble_ca_graph(chain, cut, W, H, k)

    # ---- White flash (Action): 0.15s overlay at frame 0 ----------------------
    if spec.white_flash:
        flash_frames = snap_frames(spec.white_flash_dur)    # 5 (half-up)
        # Fade a full-white layer's opacity 1.0 -> 0.0 over flash_frames using
        # a geq-free approach: blend against white with a time-decaying factor.
        chain.append(
            "format=rgb24,"
            f"lutrgb=r='clip(val+(255-val)*max(0\\,({flash_frames}-N)/{flash_frames})*if(lt(N,{flash_frames})\\,1\\,0),0,255)'"
            f":g='clip(val+(255-val)*max(0\\,({flash_frames}-N)/{flash_frames})*if(lt(N,{flash_frames})\\,1\\,0),0,255)'"
            f":b='clip(val+(255-val)*max(0\\,({flash_frames}-N)/{flash_frames})*if(lt(N,{flash_frames})\\,1\\,0),0,255)'"
        )

    # ---- Cliffhanger static hold: freeze final frame for 2.0s ----------------
    if cut.cliffhanger_hold_frames > 0:
        chain.append(f"tpad=stop_mode=clone:stop={cut.cliffhanger_hold_frames}")

    chain.append(f"fps={FPS},format=yuv420p")
    return ",".join(chain)


def _assemble_ca_graph(base_chain: list[str], cut: Cut, W: int, H: int, k: float) -> str:
    """
    Complex filtergraph for radial chromatic aberration:
    R channel expands, B channel contracts (lenscorrection k1 ±), G untouched,
    then channels are re-merged. Barrel distortion is zero at center and
    maximal at edges — matching the '10% radial, edges only' spec.
    """
    pre = ",".join(base_chain[:-1])  # everything before split
    hold = ""
    if cut.cliffhanger_hold_frames > 0:
        hold = f",tpad=stop_mode=clone:stop={cut.cliffhanger_hold_frames}"
    return (
        f"{pre},format=gbrp,split=3[r_in][g_in][b_in];"
        f"[r_in]lenscorrection=k1={k:.4f}:k2=0:i=bilinear[r_out];"
        f"[b_in]lenscorrection=k1={-k:.4f}:k2=0:i=bilinear[b_out];"
        f"[g_in]copy[g_out];"
        f"[g_out][b_out][r_out]mergeplanes=0x001020:gbrp"
        f"{hold},fps={FPS},format=yuv420p"
    )


# ============================================================================
# SECTION 6 — TRANSITIONS (concat + xfade)
# ============================================================================

def transition_graph_snippet(prev_label: str, next_label: str, out_label: str,
                             cut: Cut, prev_end_sec: float) -> str:
    """
    Emit the filter_complex snippet joining two rendered clip streams.

      hard cut   -> concat (zero-cost, the 92% path)
      crossfade  -> xfade=fade over exactly 12 frames (0.4s), scene shifts only
      flash      -> concat (the Impact Flash clip itself IS the seam mask)
    """
    if cut.transition_in == "crossfade":
        offset = max(0.0, prev_end_sec - CROSSFADE_SEC)
        return (
            f"[{prev_label}][{next_label}]"
            f"xfade=transition=fade:duration={CROSSFADE_SEC:.6f}"
            f":offset={offset:.6f}[{out_label}]"
        )
    return f"[{prev_label}][{next_label}]concat=n=2:v=1:a=0[{out_label}]"


# ============================================================================
# SECTION 7 — AUDIO GRAPH (deterministic ducking + SFX anchoring)
# ============================================================================

def build_duck_volume_expr(vo_segments: list[tuple[float, float]]) -> str:
    """
    Deterministic BGM ducking driven by the sync plan's known VO segment
    times, instead of live sidechain detection. Since build_sync_plan_...
    already knows exactly when the voiceover is speaking, we duck by a fixed
    -8dB (gain 0.3981) inside those windows with short attack/release ramps.

    Emits a volume= expression for the BGM stream. Prefer this over
    sidechaincompress: it is sample-exact, render-deterministic, and free.
    """
    if not vo_segments:
        return "volume=1.0"
    a, r, g = DUCK_ATTACK_SEC, DUCK_RELEASE_SEC, DUCK_GAIN
    terms = []
    for (s, e) in vo_segments:
        s2, e2 = snap_seconds(s), snap_seconds(e)
        # ramp down over [s-a, s], hold g, ramp up over [e, e+r]
        terms.append(
            f"if(between(t,{s2 - a:.4f},{s2:.4f}),1-(1-{g:.4f})*((t-{s2 - a:.4f})/{a}),"
            f"if(between(t,{s2:.4f},{e2:.4f}),{g:.4f},"
            f"if(between(t,{e2:.4f},{e2 + r:.4f}),{g:.4f}+(1-{g:.4f})*((t-{e2:.4f})/{r}),1)))"
        )
    # nest: min() across all windows so overlapping segments still duck once
    expr = terms[0]
    for t in terms[1:]:
        expr = f"min({expr},{t})"
    return f"volume=volume='{expr}':eval=frame"


def sidechain_duck_fallback() -> str:
    """
    Fallback if VO segment times are unavailable: classic sidechain.
    threshold 0.1 ≈ -20dB; ratio/makeup tuned to approximate a -8dB duck.
    Usage: [bgm][vo]sidechaincompress=...[bgm_ducked]
    """
    return (
        "sidechaincompress=threshold=0.1:ratio=4:attack=50:release=250"
        ":makeup=1:level_sc=1"
    )


def sfx_events(cuts: list[Cut]) -> list[dict]:
    """
    Emit the SFX schedule: every ACTION_ZOOM fires its hit at clip frame 0 —
    the exact white-flash/screen-shake initialization frame.
    Feed into your adelay-based SFX mixdown.
    """
    events = []
    for c in cuts:
        if c.sfx_frame is not None:
            events.append({
                "frame": c.sfx_frame,
                "time_sec": round(c.sfx_frame * FRAME_DUR, 6),
                "adelay_ms": int(round(c.sfx_frame * FRAME_DUR * 1000)),
                "kind": "impact",
                "panel": c.panel.panel_path,
            })
    return events


# ============================================================================
# SECTION 8 — GPU ENCODING (hardware-accelerated output)
# ============================================================================
#
# ARCHITECTURE NOTE — where the GPU actually helps:
#   zoompan / lenscorrection / dblur / lutrgb have NO CUDA implementations,
#   so the filter graph runs on CPU regardless. The GPU accelerates the
#   ENCODE stage (NVENC), which dominates wall-time at 1080p60. The correct
#   pipeline is: CPU filters -> hwupload happens implicitly inside the
#   encoder -> NVENC. Do NOT insert scale_cuda/hwupload mid-graph; bouncing
#   frames GPU<->CPU around zoompan is slower than staying on CPU.
#
# Fallback chain: h264_nvenc (NVIDIA) -> h264_qsv (Intel) ->
#                 h264_videotoolbox (macOS) -> libx264 (CPU).
# ============================================================================

ENCODER_PROFILES: dict[str, list[str]] = {
    # NVENC: p5 = quality-leaning preset, still ~5-10x faster than x264.
    # -cq 19 @ 60fps is visually transparent for manhwa line art; YouTube
    # re-encodes anyway, so feed it a high-bitrate VBR master.
    "h264_nvenc": [
        "-c:v", "h264_nvenc",
        "-preset", "p5", "-tune", "hq",
        "-rc", "vbr", "-cq", "19",
        "-b:v", "18M", "-maxrate", "28M", "-bufsize", "36M",
        "-b_ref_mode", "middle", "-spatial-aq", "1", "-temporal-aq", "1",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
    ],
    "hevc_nvenc": [
        "-c:v", "hevc_nvenc",
        "-preset", "p5", "-tune", "hq",
        "-rc", "vbr", "-cq", "22",
        "-b:v", "12M", "-maxrate", "20M", "-bufsize", "24M",
        "-spatial-aq", "1", "-temporal-aq", "1",
        "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
    ],
    "h264_qsv": [
        "-c:v", "h264_qsv",
        "-preset", "slower", "-global_quality", "20",
        "-b:v", "18M", "-maxrate", "28M",
        "-pix_fmt", "nv12",
    ],
    "h264_videotoolbox": [
        "-c:v", "h264_videotoolbox",
        "-b:v", "18M", "-maxrate", "28M",
        "-allow_sw", "0", "-pix_fmt", "yuv420p",
    ],
    "libx264": [
        "-c:v", "libx264",
        "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
    ],
}

_ENCODER_PRIORITY = ["h264_nvenc", "h264_qsv", "h264_videotoolbox", "libx264"]


def detect_encoder(prefer: Optional[str] = None) -> str:
    """
    Probe the local ffmpeg build for available hardware encoders and return
    the best one. A listed encoder can still fail at runtime if the driver
    is missing, so we verify with a 1-frame smoke test, not just -encoders.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    try:
        listed = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return "libx264"

    order = ([prefer] if prefer else []) + _ENCODER_PRIORITY
    for enc in order:
        if enc == "libx264":
            return enc
        if enc not in ENCODER_PROFILES or f" {enc} " not in listed:
            continue
        # 1-frame smoke test: catches "encoder compiled in but no GPU/driver".
        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=256x256:r=60:d=0.1",
             "-frames:v", "1", "-c:v", enc, "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode == 0:
            return enc
    return "libx264"


def build_encode_args(encoder: Optional[str] = None) -> list[str]:
    """
    Output-side ffmpeg args for the final mux: verified GPU encoder profile
    + strict 60fps stamp + faststart for YouTube ingestion.
    """
    enc = encoder or detect_encoder()
    return [
        *ENCODER_PROFILES[enc],
        "-r", str(FPS),
        "-g", str(FPS * 2),            # 2s GOP — YouTube-recommended keyint
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
    ]


def build_render_command(
    filter_complex: str,
    inputs: list[str],
    output_path: str,
    encoder: Optional[str] = None,
    threads: int = 0,
) -> list[str]:
    """
    Assemble the full ffmpeg invocation. `threads=0` lets ffmpeg use all
    cores for the CPU filter graph — at 60fps the zoompan stage is the
    bottleneck, so keep SUPERSAMPLE at 4 (8/16 doubles/quadruples that cost).
    """
    cmd = ["ffmpeg", "-hide_banner", "-y", "-threads", str(threads)]
    for src in inputs:
        cmd += ["-loop", "1", "-i", src] if src.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")) else ["-i", src]
    cmd += ["-filter_complex", filter_complex]
    cmd += build_encode_args(encoder)
    cmd += [output_path]
    return cmd


# ============================================================================
# SECTION 9 — QUICK SELF-TEST
# ============================================================================

if __name__ == "__main__":
    rng = random.Random(7)
    demo = [
        PanelJob("p01.png", PanelCategory.ESTABLISHING_PAN, 800, 2400,
                 sentence_end=6.10, is_scene_change=False, scene_id="academy",
                 script_text="Our story begins at the towering gates of Hunter Academy."),
        PanelJob("p02.png", PanelCategory.DIALOGUE_HOLD, 800, 1100,
                 sentence_end=10.42, scene_id="academy",
                 script_text="Bi Gwang stared at his own reflection — except it wasn't his face."),
        PanelJob("p03.png", PanelCategory.ACTION_ZOOM, 800, 1000,
                 sentence_end=12.05, pan_angle_deg=205.0, scene_id="academy",
                 script_text="The fist came out of nowhere."),
        PanelJob("p04.png", PanelCategory.IMPACT_FLASH, 800, 900, scene_id="academy"),
        PanelJob("p05.png", PanelCategory.ACTION_ZOOM, 800, 1000,
                 sentence_end=15.90, pan_angle_deg=0.0, scene_id="academy",
                 script_text="SYSTEM: Hidden Quest unlocked."),
        PanelJob("p06.png", PanelCategory.ESTABLISHING_PAN, 800, 2600,
                 sentence_end=22.75, is_scene_change=True, scene_id="dungeon",
                 script_text="Deep beneath the city, the dungeon stirred."),
    ]
    pacing = {2: PacingMode.ACTION, 3: PacingMode.CLIMAX, 4: PacingMode.CLIMAX}
    cuts = build_cut_list(demo, pacing_by_index=pacing)

    print(f"{'idx':>3} {'start_f':>7} {'frames':>6} {'sec':>6} "
          f"{'category':<24} {'trans':<9} {'cliff':>5} {'sfx_f':>6}")
    for i, c in enumerate(cuts):
        print(f"{i:>3} {c.start_frame:>7} {c.n_frames:>6} {c.duration_sec:>6.2f} "
              f"{c.spec.category.value:<24} {c.transition_in:<9} "
              f"{c.cliffhanger_hold_frames:>5} {str(c.sfx_frame):>6}")

    print("\nTransition budget:", validate_transition_budget(cuts))
    print(f"\nFPS = {FPS}")
    print(f"AUDIO_LEAD_FRAMES = {AUDIO_LEAD_FRAMES} (0.15s -> 9 @ 60fps)")
    print(f"CLIMAX window = {CLIMAX_MIN_FRAMES}-{CLIMAX_MAX_FRAMES} frames "
          f"({CLIMAX_MIN_SEC}s-{CLIMAX_MAX_SEC}s)")
    print(f"CROSSFADE = {CROSSFADE_FRAMES} frames = {CROSSFADE_SEC}s")
    print(f"CLIFFHANGER_HOLD = {CLIFFHANGER_HOLD_FRAMES} frames")
    print(f"DUCK_GAIN = {DUCK_GAIN:.4f} ({DUCK_DB} dB)")
    # At 30fps: 0.15s=5f, 0.4s=12f, 0.8s=24f, 1.2s=36f
    assert AUDIO_LEAD_FRAMES == 5, AUDIO_LEAD_FRAMES
    assert CROSSFADE_FRAMES == 12, CROSSFADE_FRAMES
    assert (CLIMAX_MIN_FRAMES, CLIMAX_MAX_FRAMES) == (24, 36)

    try:
        enc = detect_encoder()
        print(f"\nDetected encoder: {enc}")
        print("Encode args:", " ".join(build_encode_args(enc)))
    except RuntimeError as e:
        print(f"\nEncoder detection skipped: {e}")

    print("\n--- Sample filtergraph: ACTION_ZOOM (p03) ---")
    print(build_clip_filtergraph(cuts[2]))
    print("\n--- Sample filtergraph: IMPACT_FLASH (p04) ---")
    print(build_clip_filtergraph(cuts[3]))
    print("\n--- SFX schedule ---")
    for e in sfx_events(cuts):
        print(e)
    print("\n--- BGM duck expr (2 VO segments) ---")
    print(build_duck_volume_expr([(0.5, 6.1), (7.0, 12.05)])[:400], "...")
