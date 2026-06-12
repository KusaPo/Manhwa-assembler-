"""
Caption Generator Module
========================
Generates an SRT caption file from the script segments themselves,
timed proportionally to match the voiceover duration.

This avoids the need for Whisper - since we already have the exact words,
we just need to distribute them across the audio timeline.

Distribution is weighted by word count: longer segments get more screen time.
"""

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("caption_generator")


def _seconds_to_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _chunk_segment_for_captions(text: str, max_words_per_caption: int = 10) -> List[str]:
    """
    Split a long segment into caption-sized chunks.
    Captions should be short (5-12 words) for readability.
    """
    words = text.split()
    if len(words) <= max_words_per_caption:
        return [text]

    chunks = []
    for i in range(0, len(words), max_words_per_caption):
        chunk = " ".join(words[i:i + max_words_per_caption])
        chunks.append(chunk)
    return chunks


def generate_captions_from_script(
    script_segments: List[str],
    total_duration: float,
    output_path: Path,
    max_words_per_caption: int = 10,
) -> Path:
    """
    Generate an SRT file from script segments, timed proportionally to total audio duration.
    
    Args:
        script_segments: List of script chunks (from script_processor)
        total_duration: Total voiceover duration in seconds
        output_path: Where to save the .srt file
        max_words_per_caption: Cap each caption line at this many words
    
    Returns: Path to generated SRT file
    """
    # First, split each segment into caption-sized chunks
    all_chunks: List[str] = []
    for segment in script_segments:
        chunks = _chunk_segment_for_captions(segment, max_words_per_caption)
        all_chunks.extend(chunks)

    # Total word count across all chunks
    total_words = sum(len(chunk.split()) for chunk in all_chunks)
    if total_words == 0:
        raise ValueError("Cannot generate captions: script is empty.")

    # Each word gets a proportional share of the timeline
    seconds_per_word = total_duration / total_words

    # Build SRT entries
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        current_time = 0.0
        for i, chunk in enumerate(all_chunks, start=1):
            chunk_words = len(chunk.split())
            chunk_duration = chunk_words * seconds_per_word
            start_time = current_time
            end_time = current_time + chunk_duration

            f.write(f"{i}\n")
            f.write(f"{_seconds_to_srt_time(start_time)} --> {_seconds_to_srt_time(end_time)}\n")
            f.write(f"{chunk}\n\n")

            current_time = end_time

    logger.info(f"  Generated {len(all_chunks)} caption lines")
    return output_path


def parse_srt(srt_path: Path) -> List[Tuple[float, float, str]]:
    """
    Parse an SRT file back into (start_sec, end_sec, text) tuples.
    Used by the video assembler to overlay captions.
    """
    entries = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = content.split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        # Line 1 is index, line 2 is timing, lines 3+ are text
        timing = lines[1]
        text = " ".join(lines[2:])
        start_str, end_str = timing.split(" --> ")
        start = _srt_time_to_seconds(start_str)
        end = _srt_time_to_seconds(end_str)
        entries.append((start, end, text))

    return entries


def _srt_time_to_seconds(srt_time: str) -> float:
    """Convert HH:MM:SS,mmm to seconds."""
    time_part, ms_part = srt_time.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms_part) / 1000.0
