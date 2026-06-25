"""Shared data models for the recap pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TimestampSegment:
    index: int
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TimestampSegment":
        return cls(
            index=int(data["index"]),
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
        )


@dataclass
class ScriptScene:
    scene_id: int
    sentences: List[str]

    def to_dict(self) -> dict:
        return {"id": self.scene_id, "sentences": self.sentences}

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptScene":
        return cls(
            scene_id=int(data.get("id", data.get("scene_id", 1))),
            sentences=list(data["sentences"]),
        )


@dataclass
class ScriptDocument:
    scenes: List[ScriptScene]

    @property
    def all_sentences(self) -> List[str]:
        return [s for scene in self.scenes for s in scene.sentences]

    def to_dict(self) -> dict:
        return {"scenes": [s.to_dict() for s in self.scenes]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptDocument":
        return cls(scenes=[ScriptScene.from_dict(s) for s in data["scenes"]])

    @classmethod
    def load(cls, path: Path) -> "ScriptDocument":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_plaintext(cls, path: Path, delimiter: str = "---") -> "ScriptDocument":
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Script file is empty: {path}")

        if delimiter in text:
            scene_texts = [s.strip() for s in text.split(delimiter) if s.strip()]
        else:
            scene_texts = [text]

        scenes: List[ScriptScene] = []
        for scene_id, scene_text in enumerate(scene_texts, start=1):
            import re

            raw = re.split(r"(?<=[.!?])\s+", scene_text.strip())
            sentences = []
            buffer = ""
            for part in raw:
                if not part.strip():
                    continue
                if len(part.split()) < 4 and buffer:
                    buffer += " " + part.strip()
                else:
                    if buffer:
                        sentences.append(buffer.strip())
                    buffer = part.strip()
            if buffer:
                sentences.append(buffer.strip())
            scenes.append(ScriptScene(scene_id=scene_id, sentences=sentences))

        return cls(scenes=scenes)


@dataclass
class PanelAsset:
    index: int
    path: Path
    source_strip: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "path": str(self.path),
            "source_strip": self.source_strip,
        }


@dataclass
class AudioResult:
    voiceover_path: Path
    timestamps: List[TimestampSegment]
    timestamps_path: Path


@dataclass
class SyncPlanEntry:
    panel_index: int
    panel_path: Path
    segment_index: int
    duration: float

    def to_dict(self) -> dict:
        return {
            "panel_index": self.panel_index,
            "panel_path": str(self.panel_path),
            "segment_index": self.segment_index,
            "duration": round(self.duration, 3),
        }


@dataclass
class SyncPlan:
    entries: List[SyncPlanEntry] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"entries": [e.to_dict() for e in self.entries]},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "SyncPlan":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            entries=[
                SyncPlanEntry(
                    panel_index=e["panel_index"],
                    panel_path=Path(e["panel_path"]),
                    segment_index=e["segment_index"],
                    duration=float(e["duration"]),
                )
                for e in data["entries"]
            ]
        )


def save_timestamps(path: Path, segments: List[TimestampSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2, ensure_ascii=False)


def load_timestamps(path: Path) -> List[TimestampSegment]:
    with open(path, "r", encoding="utf-8") as f:
        return [TimestampSegment.from_dict(item) for item in json.load(f)]
