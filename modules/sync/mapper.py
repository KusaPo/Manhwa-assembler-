"""Panel-to-sentence sync mapping and duration planning."""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from modules.sync.script import Sentence
from modules.sync.timeline import TimelinePanel

logger = logging.getLogger("sync.mapper")


@dataclass
class PanelPlanEntry:
    order: int
    file: str
    segment: int
    role: str  # "hold" or "flash"
    duration: float
    sentence_global_indices: List[int] = field(default_factory=list)
    sentence_text: str = ""


@dataclass
class SyncPlan:
    panels: List[PanelPlanEntry]
    silence_after_ms: List[int]
    sentence_durations: List[float]

    @property
    def image_durations(self) -> List[float]:
        return [p.duration for p in self.panels]

    @property
    def image_files(self) -> List[str]:
        return [p.file for p in self.panels]


def _panels_for_scene(
    timeline_panels: Sequence[TimelinePanel],
    scene_id: int,
) -> List[TimelinePanel]:
    return [p for p in timeline_panels if p.segment == scene_id]


def _distribute_flash_panels(
    sentences: Sequence[Sentence],
    extra_panel_count: int,
) -> Dict[int, int]:
    flash_counts: Dict[int, int] = {s.global_idx: 0 for s in sentences}
    if extra_panel_count <= 0 or not sentences:
        return flash_counts

    ranked = sorted(sentences, key=lambda s: s.word_count, reverse=True)
    for i in range(extra_panel_count):
        flash_counts[ranked[i % len(ranked)].global_idx] += 1
    return flash_counts


def _merge_sentences_into_groups(
    sentences: Sequence[Sentence],
    group_count: int,
) -> List[List[Sentence]]:
    if group_count <= 0:
        return []
    if group_count >= len(sentences):
        return [[s] for s in sentences]

    total_words = sum(s.word_count for s in sentences)
    target_per_group = total_words / group_count

    groups: List[List[Sentence]] = []
    current: List[Sentence] = []
    current_words = 0

    for idx, sentence in enumerate(sentences):
        current.append(sentence)
        current_words += sentence.word_count

        groups_remaining = group_count - len(groups) - 1
        sentences_remaining = len(sentences) - idx - 1

        if groups_remaining == 0:
            continue

        if sentences_remaining <= groups_remaining:
            groups.append(current)
            current = []
            current_words = 0
            continue

        if current_words >= target_per_group:
            groups.append(current)
            current = []
            current_words = 0

    if current:
        if groups and len(groups) >= group_count:
            groups[-1].extend(current)
        else:
            groups.append(current)

    while len(groups) < group_count:
        groups.append([])

    return groups[:group_count]


def _map_scene(
    scene_id: int,
    sentences: Sequence[Sentence],
    panels: Sequence[TimelinePanel],
    sentence_durations: Dict[int, float],
    flash_duration: float,
) -> Tuple[List[PanelPlanEntry], Dict[int, int]]:
    panel_entries: List[PanelPlanEntry] = []
    silence_after: Dict[int, int] = {s.global_idx: 0 for s in sentences}

    if not panels:
        return panel_entries, silence_after
    if not sentences:
        logger.warning(f"  Scene {scene_id}: panels but no sentences, skipping")
        return panel_entries, silence_after

    panel_count = len(panels)
    sentence_count = len(sentences)

    if panel_count >= sentence_count:
        flash_counts = _distribute_flash_panels(
            sentences,
            panel_count - sentence_count,
        )
        panel_idx = 0
        for sentence in sentences:
            hold_duration = sentence_durations[sentence.global_idx]

            panel_entries.append(
                PanelPlanEntry(
                    order=0,
                    file=panels[panel_idx].file,
                    segment=scene_id,
                    role="hold",
                    duration=hold_duration,
                    sentence_global_indices=[sentence.global_idx],
                    sentence_text=sentence.text,
                )
            )
            panel_idx += 1

            for _ in range(flash_counts[sentence.global_idx]):
                panel_entries.append(
                    PanelPlanEntry(
                        order=0,
                        file=panels[panel_idx].file,
                        segment=scene_id,
                        role="flash",
                        duration=flash_duration,
                        sentence_global_indices=[sentence.global_idx],
                        sentence_text=sentence.text,
                    )
                )
                silence_after[sentence.global_idx] += int(flash_duration * 1000)
                panel_idx += 1
    else:
        groups = _merge_sentences_into_groups(sentences, panel_count)
        for panel, group in zip(panels, groups):
            if not group:
                continue
            hold_duration = sum(sentence_durations[s.global_idx] for s in group)
            panel_entries.append(
                PanelPlanEntry(
                    order=0,
                    file=panel.file,
                    segment=scene_id,
                    role="hold",
                    duration=hold_duration,
                    sentence_global_indices=[s.global_idx for s in group],
                    sentence_text=" | ".join(s.text for s in group),
                )
            )

    return panel_entries, silence_after


def build_sync_plan(
    scenes: List[List[Sentence]],
    timeline_panels: Sequence[TimelinePanel],
    sentence_durations: Sequence[float],
    flash_duration: float = 0.25,
) -> SyncPlan:
    duration_by_idx = {
        sentence.global_idx: sentence_durations[sentence.global_idx]
        for scene in scenes
        for sentence in scene
    }

    all_panels: List[PanelPlanEntry] = []
    all_sentences = [s for scene in scenes for s in scene]
    silence_after_ms = [0] * len(all_sentences)

    for scene in scenes:
        if not scene:
            continue
        scene_id = scene[0].scene_id
        scene_panels = _panels_for_scene(timeline_panels, scene_id)
        scene_entries, scene_silence = _map_scene(
            scene_id=scene_id,
            sentences=scene,
            panels=scene_panels,
            sentence_durations=duration_by_idx,
            flash_duration=flash_duration,
        )
        all_panels.extend(scene_entries)
        for global_idx, ms in scene_silence.items():
            silence_after_ms[global_idx] = ms

    for order, entry in enumerate(all_panels, start=1):
        entry.order = order

    logger.info(
        f"  Sync plan: {len(all_panels)} panel slots, "
        f"{sum(1 for p in all_panels if p.role == 'hold')} holds, "
        f"{sum(1 for p in all_panels if p.role == 'flash')} flashes"
    )

    return SyncPlan(
        panels=all_panels,
        silence_after_ms=silence_after_ms,
        sentence_durations=list(sentence_durations),
    )


def resolve_panel_paths(
    sync_plan: SyncPlan,
    images_dir: Path,
    fallback_paths: Sequence[Path],
) -> Tuple[List[Path], List[float]]:
    by_name = {p.name: p for p in fallback_paths}
    ordered: List[Path] = []
    durations: List[float] = []
    missing: List[str] = []

    for entry in sync_plan.panels:
        path = images_dir / entry.file
        if not path.exists():
            path = by_name.get(entry.file)
        if path is None or not path.exists():
            missing.append(entry.file)
            continue
        ordered.append(path)
        durations.append(entry.duration)

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} timeline panels missing from {images_dir}/. "
            f"Expected {len(sync_plan.panels)}, resolved {len(ordered)}. "
            f"Missing: {preview}{suffix}. "
            "Copy panel images from your asset folder before running."
        )

    used = {p.name for p in ordered}
    leftover = [p for p in fallback_paths if p.name not in used]
    if leftover:
        logger.warning(
            f"  {len(leftover)} images on disk are not referenced in the sync plan "
            "(ignored)"
        )

    return ordered, durations


def export_sync_plan(sync_plan: SyncPlan, output_path: Path) -> None:
    data = {
        "panels": [
            {
                "order": p.order,
                "file": p.file,
                "segment": p.segment,
                "role": p.role,
                "duration": round(p.duration, 3),
                "sentence_global_indices": p.sentence_global_indices,
                "sentence_text": p.sentence_text,
            }
            for p in sync_plan.panels
        ],
        "silence_after_ms": sync_plan.silence_after_ms,
        "sentence_durations": [round(d, 3) for d in sync_plan.sentence_durations],
        "total_video_duration": round(sum(p.duration for p in sync_plan.panels), 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Wrote sync plan: {output_path}")


def export_sync_sheet_csv(sync_plan: SyncPlan, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = 0.0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "order",
                "file",
                "segment",
                "role",
                "sentence_indices",
                "start_s",
                "end_s",
                "dur_s",
                "sentence_text",
            ]
        )
        for entry in sync_plan.panels:
            end = start + entry.duration
            writer.writerow(
                [
                    entry.order,
                    entry.file,
                    entry.segment,
                    entry.role,
                    ";".join(str(i) for i in entry.sentence_global_indices),
                    round(start, 2),
                    round(end, 2),
                    round(entry.duration, 2),
                    entry.sentence_text[:120],
                ]
            )
            start = end
    logger.info(f"  Wrote sync sheet: {output_path}")
