# Manhwa Recap Video Generator

Automated Python pipeline that turns manhwa chapter summaries and vertical webcomic strips into YouTube-ready recap videos with AI narration, Ken Burns effects, background music, and synced subtitles.

**Pipeline:** `raw_chapter.txt` + `raw_strips/` → OpenAI script → ElevenLabs TTS + timestamps → panel slicing → final MP4

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install **FFmpeg** on your system (MoviePy encoding):

- **macOS:** `brew install ffmpeg`
- **Windows:** `winget install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

For burned-in captions, install **ImageMagick** (`brew install imagemagick` on macOS).

### 2. Configure API keys

```bash
cp config.example.json config.json
```

Fill in:
- `openai_api_key` — script generation (Phase 1)
- `elevenlabs_api_key` + `elevenlabs_voice_id` — voiceover (Phase 2)

### 3. Add inputs

```
input/
├── raw_chapter.txt      # Chapter summary (or use --source-url)
├── raw_strips/          # Vertical scroll webcomic images (.webp, .png, .jpg)
├── script.txt           # Optional: skip OpenAI if hand-written
└── music.mp3            # Optional background track
```

### 4. Run

```bash
python run.py
```

Output: `output/final_video.mp4`

---

## Pipeline Phases

| Phase | Command | Output |
|-------|---------|--------|
| Script | `python run.py --phase script` | `output/script.json` |
| Audio | `python run.py --phase audio` | `output/voiceover.mp3`, `output/timestamps.json` |
| Vision | `python run.py --phase vision` | `output/panels/*.webp` |
| Assembly | `python run.py --phase assembly` | `output/final_video.mp4`, `output/subtitles.srt` |

### Useful flags

```bash
python run.py --skip-tts          # Reuse cached voice chunks
python run.py --preview           # Render first 60 seconds only
python run.py --source-url URL    # Fetch chapter text from URL
python run.py --no-captions       # Skip burned-in subtitles
```

---

## How Sync Works

1. **Phase 2** generates one ElevenLabs TTS call per sentence and measures real MP3 duration with `mutagen`.
2. Cumulative durations produce `output/timestamps.json` — the sync contract for the whole pipeline.
3. **Phase 4** assigns 1–2 panels per sentence segment and builds Ken Burns clips for each panel's hold time.
4. Subtitles are generated directly from timestamps (no Whisper needed).

---

## Project Structure

```
├── run.py                     # CLI entry point
├── core/                      # Config, models, utils
├── engines/
│   ├── script_generator.py    # OpenAI recap + sanitization
│   ├── audio_generator.py     # ElevenLabs TTS + timestamps
│   ├── image_processor.py     # Strip slicing + blur-fill
│   ├── video_assembler.py     # Ken Burns + mux + SRT
│   └── pipeline.py            # Orchestrator
├── prompts/                   # LLM system prompt + sanitize rules
├── input/
└── output/                    # gitignored
```

---

## Development

```bash
pytest tests/
python run.py --phase audio --skip-tts   # test with cached TTS
```

### Skip Phase 1

If you already have `input/script.txt`, Phase 1 uses it directly without calling OpenAI.

---

## Troubleshooting

**"No script input found"** — add `input/raw_chapter.txt`, `input/script.txt`, or pass `--source-url`.

**"No images in raw_strips"** — add vertical webcomic scroll images to `input/raw_strips/`.

**"Panel durations do not match voiceover"** — regenerate audio (`python run.py --phase audio`) or add more panels in vision phase.

**"Encoded video has no audio track"** — verify FFmpeg is installed and `output/voiceover.mp3` plays in a media player.

**"ElevenLabs API error 401"** — check `elevenlabs_api_key` in `config.json`.

**Captions require ImageMagick** — install ImageMagick or use `--no-captions`.

---

## Costs

~$0.30–0.50 per 10-minute video (ElevenLabs TTS + OpenAI script generation).
