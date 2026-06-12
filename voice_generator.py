"""
Voice Generation Module
========================
Calls ElevenLabs API to convert script text into MP3 voiceover.
Uses ElevenLabs v2 multilingual model with configurable voice settings.
"""

import logging
from pathlib import Path
from typing import Optional

import requests
from pydub import AudioSegment

logger = logging.getLogger("voice_generator")

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def generate_voiceover(
    text: str,
    output_path: Path,
    voice_id: str,
    api_key: str,
    speed: float = 1.0,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.3,
    model_id: str = "eleven_multilingual_v2",
) -> Path:
    """
    Generate voiceover from text using ElevenLabs API.
    
    Args:
        text: Full script text (no delimiters)
        output_path: Where to save the MP3
        voice_id: ElevenLabs voice ID (get from your account dashboard)
        api_key: ElevenLabs API key
        speed: Playback speed multiplier (applied post-generation)
        stability: 0-1, lower = more expressive, higher = more consistent
        similarity_boost: 0-1, voice clone similarity strength
        style: 0-1, style exaggeration (0 = neutral, 1 = dramatic)
        model_id: ElevenLabs model to use
    
    Returns: Path to generated MP3
    """
    if not api_key or api_key == "your-key-here":
        raise ValueError(
            "ElevenLabs API key not configured. "
            "Set it in config.json. Get a key at https://elevenlabs.io/app/settings/api-keys"
        )

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

    logger.info(f"  Sending {len(text)} chars to ElevenLabs (voice: {voice_id})")
    response = requests.post(url, json=payload, headers=headers, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs API error {response.status_code}: {response.text}"
        )

    # Save raw output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)

    # Apply speed adjustment if needed (post-process via pydub)
    if speed != 1.0:
        logger.info(f"  Applying speed adjustment: {speed}x")
        audio = AudioSegment.from_mp3(output_path)
        # Speed change without pitch shift: change frame_rate then resample
        new_rate = int(audio.frame_rate * speed)
        sped = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
        sped = sped.set_frame_rate(audio.frame_rate)
        sped.export(output_path, format="mp3", bitrate="192k")

    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """Return duration of an audio file in seconds."""
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0  # pydub returns milliseconds
