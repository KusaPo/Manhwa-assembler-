"""
Script Processing Module
========================
Reads script.txt and parses into segments.

Supports two formats:
  1. Delimiter-based: scenes separated by `---`
  2. Auto-split: splits on sentence boundaries
"""

import re
from pathlib import Path
from typing import List


def parse_script(script_path: Path, delimiter: str = "---") -> List[str]:
    """
    Parse script.txt into a list of segments.
    
    If the script contains `---` delimiters, segments are split on those.
    Otherwise, splits on sentence boundaries (. ! ?).
    
    Returns: list of text segments, each one a caption-able chunk.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Script file is empty: {script_path}")

    # Method 1: explicit delimiters
    if delimiter in text:
        segments = [s.strip() for s in text.split(delimiter) if s.strip()]
        return segments

    # Method 2: sentence-level split (for captions)
    # Splits on . ! ? while keeping reasonable chunk sizes
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Merge short sentences (< 4 words) with neighbors for better captions
    merged = []
    buffer = ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        word_count = len(sentence.split())
        if word_count < 4 and buffer:
            buffer += " " + sentence.strip()
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = sentence.strip()
    if buffer:
        merged.append(buffer.strip())

    return merged


def estimate_word_count(segments: List[str]) -> int:
    """Total word count across all segments. Used for duration estimation."""
    return sum(len(s.split()) for s in segments)


def estimate_duration_seconds(word_count: int, wpm: int = 155) -> float:
    """
    Estimate audio duration in seconds based on words per minute.
    Default 155 wpm matches typical narration speed.
    """
    return (word_count / wpm) * 60.0
