"""
Audio Engine — ElevenLabs TTS with sentence-level timestamps.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List

import imageio_ffmpeg
import requests
from mutagen.mp3 import MP3

from core.config import RecapConfig
from core.models import (
    AudioResult,
    ScriptDocument,
    TimestampSegment,
    save_timestamps,
)

logger = logging.getLogger("engines.audio")

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def _ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(args: List[str]) -> None:
    command = [_ffmpeg_path(), *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({result.returncode}): {result.stderr.strip() or result.stdout}"
        )


def get_audio_duration(audio_path: Path) -> float:
    return float(MP3(audio_path).info.length)


def _validate_api_key(api_key: str) -> None:
    if not api_key or api_key == "your-key-here":
        raise ValueError(
            "ElevenLabs API key not configured. Set elevenlabs_api_key in config.json."
        )


def _call_elevenlabs(
    text: str,
    output_path: Path,
    voice_id: str,
    api_key: str,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.3,
    model_id: str = "eleven_multilingual_v2",
) -> Path:
    _validate_api_key(api_key)
    url = ELEVENLABS_API_URL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": True,
        },
    }
    response = requests.post(url, json=payload, headers=headers, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs API error {response.status_code}: {response.text}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path


def _apply_speed(audio_path: Path, speed: float) -> None:
    if speed == 1.0:
        return
    temp_path = audio_path.with_suffix(".tmp.mp3")
    _run_ffmpeg(
        [
            "-y", "-i", str(audio_path),
            "-filter:a", f"atempo={speed}",
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(temp_path),
        ]
    )
    temp_path.replace(audio_path)


def _write_silence_mp3(path: Path, duration_ms: int) -> None:
    if duration_ms <= 0:
        return
    duration_s = duration_ms / 1000.0
    _run_ffmpeg(
        [
            "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", f"{duration_s:.3f}",
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(path),
        ]
    )


def build_timestamps(
    sentences: List[str],
    chunk_durations: List[float],
    inter_sentence_pause_ms: int = 0,
) -> List[TimestampSegment]:
    segments: List[TimestampSegment] = []
    cursor = 0.0
    pause_s = inter_sentence_pause_ms / 1000.0

    for idx, (text, duration) in enumerate(zip(sentences, chunk_durations)):
        start = cursor
        end = start + duration
        segments.append(
            TimestampSegment(index=idx, text=text, start=start, end=end)
        )
        cursor = end
        if pause_s > 0 and idx < len(sentences) - 1:
            cursor += pause_s

    return segments


class AudioGenerator:
    def __init__(self, config: RecapConfig, paths) -> None:
        self.config = config
        self.paths = paths

    def generate(
        self,
        script: ScriptDocument,
        *,
        skip_existing: bool = False,
    ) -> AudioResult:
        sentences = script.all_sentences
        if not sentences:
            raise ValueError("Script has no sentences")

        chunk_dir = self.paths.chunk_dir
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_durations: List[float] = []

        for idx, text in enumerate(sentences):
            chunk_path = chunk_dir / f"{idx:03d}.mp3"
            if skip_existing and chunk_path.exists():
                duration = get_audio_duration(chunk_path)
                logger.info(
                    f"  Chunk {idx + 1}/{len(sentences)} cached ({duration:.2f}s)"
                )
            else:
                logger.info(f"  Chunk {idx + 1}/{len(sentences)}: {text[:60]}...")
                _call_elevenlabs(
                    text=text,
                    output_path=chunk_path,
                    voice_id=self.config.elevenlabs_voice_id,
                    api_key=self.config.elevenlabs_api_key,
                )
                _apply_speed(chunk_path, self.config.tts_speed)
                duration = get_audio_duration(chunk_path)
            chunk_durations.append(duration)

        timestamps = build_timestamps(
            sentences,
            chunk_durations,
            inter_sentence_pause_ms=self.config.inter_sentence_pause_ms,
        )

        voiceover_path = self.paths.voiceover_file
        self._concatenate_chunks(sentences, chunk_dir, voiceover_path)

        timestamps_path = self.paths.timestamps_file
        save_timestamps(timestamps_path, timestamps)

        vo_duration = get_audio_duration(voiceover_path)
        ts_end = timestamps[-1].end if timestamps else 0.0
        logger.info(
            f"  Voiceover: {voiceover_path} ({vo_duration:.1f}s), "
            f"timestamps end: {ts_end:.1f}s"
        )

        return AudioResult(
            voiceover_path=voiceover_path,
            timestamps=timestamps,
            timestamps_path=timestamps_path,
        )

    def _concatenate_chunks(
        self,
        sentences: List[str],
        chunk_dir: Path,
        output_path: Path,
    ) -> None:
        pause_ms = self.config.inter_sentence_pause_ms
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="voice_concat_") as tmp:
            tmp_dir = Path(tmp)
            concat_list = tmp_dir / "concat.txt"
            entries: List[str] = []

            for idx in range(len(sentences)):
                chunk_path = chunk_dir / f"{idx:03d}.mp3"
                if not chunk_path.exists():
                    raise FileNotFoundError(f"Missing voice chunk: {chunk_path}")
                entries.append(f"file '{chunk_path.resolve()}'")
                if pause_ms > 0 and idx < len(sentences) - 1:
                    pause_path = tmp_dir / f"pause_{idx:03d}.mp3"
                    _write_silence_mp3(pause_path, pause_ms)
                    entries.append(f"file '{pause_path.resolve()}'")

            concat_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
            _run_ffmpeg(
                [
                    "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list),
                    "-c:a", "libmp3lame", "-b:a", "192k",
                    str(output_path),
                ]
            )
