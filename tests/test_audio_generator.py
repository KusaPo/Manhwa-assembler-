"""Tests for audio timestamp generation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config import RecapConfig, ProjectPaths
from core.models import ScriptDocument, load_timestamps
from engines.audio_generator import AudioGenerator, build_timestamps, get_audio_duration


@pytest.fixture
def sample_script() -> ScriptDocument:
    fixture = Path(__file__).parent / "fixtures" / "sample_script.json"
    return ScriptDocument.load(fixture)


@pytest.fixture
def paths(tmp_path) -> ProjectPaths:
    config = RecapConfig()
    paths = ProjectPaths.from_config(config, root=tmp_path)
    paths.ensure_output_dir()
    return paths


def test_build_timestamps_no_pause():
    segments = build_timestamps(
        ["One.", "Two.", "Three."],
        [2.0, 3.0, 1.5],
        inter_sentence_pause_ms=0,
    )
    assert len(segments) == 3
    assert segments[0].start == 0.0
    assert segments[0].end == 2.0
    assert segments[1].start == 2.0
    assert segments[1].end == 5.0
    assert segments[2].start == 5.0
    assert segments[2].end == 6.5


def test_build_timestamps_with_pause():
    segments = build_timestamps(
        ["One.", "Two."],
        [2.0, 3.0],
        inter_sentence_pause_ms=500,
    )
    assert segments[0].end == 2.0
    assert segments[1].start == 2.5
    assert segments[1].end == 5.5


def test_timestamp_sum_matches_expected():
    durations = [4.2, 3.1, 2.7]
    segments = build_timestamps(["a", "b", "c"], durations, 0)
    total = segments[-1].end
    assert abs(total - sum(durations)) < 0.001


@patch("engines.audio_generator._call_elevenlabs")
@patch("engines.audio_generator._apply_speed")
@patch("engines.audio_generator.get_audio_duration")
def test_audio_generator_writes_timestamps(
    mock_duration,
    mock_speed,
    mock_elevenlabs,
    sample_script,
    paths,
):
    mock_duration.side_effect = [2.0, 3.0, 1.5, 6.5]
    mock_elevenlabs.return_value = paths.chunk_dir / "000.mp3"

    config = RecapConfig(
        elevenlabs_api_key="test-key",
        elevenlabs_voice_id="test-voice",
    )
    gen = AudioGenerator(config, paths)

    with patch.object(gen, "_concatenate_chunks") as mock_concat:
        mock_concat.side_effect = lambda s, c, o: o.write_bytes(b"x") or None
        result = gen.generate(sample_script, skip_existing=False)

    assert result.timestamps_path.exists()
    loaded = load_timestamps(result.timestamps_path)
    assert len(loaded) == 3
    assert loaded[-1].end == pytest.approx(6.5, abs=0.01)
