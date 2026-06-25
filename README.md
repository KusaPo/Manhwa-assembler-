# Manhwa Video Assembler

Automated Python pipeline that turns manhwa panels + a script into a finished YouTube-ready video.

**Pipeline:** `script.txt` + `images/` + `timeline.json` + `music.mp3` → per-sentence ElevenLabs voiceover → audio-synced panels → Ken Burns'd video → 1080p MP4

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

You also need **FFmpeg** installed on your system (MoviePy uses it for encoding):

- **Windows:** `winget install ffmpeg` or download from https://ffmpeg.org/download.html
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

For the **TextClip caption overlay**, MoviePy needs **ImageMagick** installed:

- **Windows:** Download from https://imagemagick.org/script/download.php#windows (check "Install legacy utilities" during install)
- **macOS:** `brew install imagemagick`
- **Linux:** `sudo apt install imagemagick`

### 2. Configure API keys

```bash
cp config.example.json config.json
```

Edit `config.json` and fill in:
- `elevenlabs_api_key` - get from https://elevenlabs.io/app/settings/api-keys
- `elevenlabs_voice_id` - get from https://elevenlabs.io/app/voice-library (pick a voice, copy its ID)

`config.json` is gitignored — never commit API keys.

### 3. Set up your project folder

**Important:** Every panel filename listed in `input/timeline.json` must exist in `input/images/` before you run the assembler. Panel images are not stored in git — copy them from your asset folder after cloning.

```
manhwa_assembler/
├── input/
│   ├── images/          ← drop panel screenshots here (001-panel.png, ...)
│   ├── script.txt       ← your narration text
│   ├── timeline.json    ← panel order + scene segments (required for audio-sync)
│   └── music.mp3        ← background track (optional)
├── output/              ← final video appears here
│   ├── voice_chunks/    ← cached per-sentence MP3s
│   ├── sync_plan.json     ← panel timing plan (review before render)
│   └── sync_sheet_audio.csv
├── assembly_script.py           # CLI entry point
├── config.example.json
├── config.json                  # local only (gitignored)
└── modules/
    ├── config.py                # AssemblyConfig + ProjectPaths
    ├── pipeline.py              # AssemblyPipeline orchestrator
    ├── common/utils.py
    ├── sync/                    # script parsing, timeline, panel mapping
    ├── audio/voice.py           # ElevenLabs TTS
    └── video/                   # assembler, images, captions
```

### 4. Run

```bash
python assembly_script.py
```

That's it. Coffee break. Come back in 10–15 min, `output/final_video.mp4` is ready.

---

## Usage Modes

```bash
# Standard run (audio-synced pacing — default)
python assembly_script.py

# Dry run (validate inputs, build sync plan, no API calls or render)
python assembly_script.py --dry-run

# Quick preview (renders only first 60 seconds for QA)
python assembly_script.py --preview

# Reuse cached voice chunks + voiceover.mp3 (saves API credits during iteration)
python assembly_script.py --skip-voiceover

# Legacy word-count pacing from timeline.json dur fields
python assembly_script.py --legacy-sync

# Speed up/slow down narration
python assembly_script.py --speed 1.1
```

---

## Audio-Synced Panel Pacing (default)

By default, the assembler generates **one ElevenLabs TTS call per sentence** and uses the **real MP3 duration** of each line to time panels:

- The **primary panel** for a sentence holds until that line finishes playing.
- **Extra panels** mapped to the same sentence get a brief flash after the line (default 0.25s each), with matching silence inserted in the voiceover so audio and video stay locked.
- When a scene has more sentences than panels, consecutive sentences are merged onto one panel (no mid-line cuts).

Review the generated schedule before rendering:

- `output/sync_plan.json` — machine-readable panel plan
- `output/sync_sheet_audio.csv` — open in a spreadsheet

Cached sentence audio lives in `output/voice_chunks/`. Re-rendering video is cheap with `--skip-voiceover`.

Config keys in `config.json`:

- `extra_panel_flash_duration` — seconds per flash panel (default `0.25`)
- `inter_sentence_pause_ms` — optional gap between sentences (default `0`)
- `voice_chunk_cache_dir` — where per-sentence MP3s are stored

**API cost:** ~150 ElevenLabs calls for a typical chapter script (~$0.30–0.45, similar to one monolithic call).

Use `--legacy-sync` to revert to the old `timeline.json` word-count duration estimates.

### Legacy word-count pacing (`--legacy-sync`)

The legacy mode times panels using **estimated** durations from `timeline.json` `dur` fields (rescaled to voiceover length), not per-sentence MP3 lengths.

- Sync is at the **scene** level, not line-perfect.
- Edit `dur` in `input/timeline.json` to tune timing.
- Run: `python assembly_script.py --legacy-sync --skip-voiceover`

### Panel pacing caveats (audio-sync)

- Sync is at the **sentence** level, not word-perfect within a line.
- Panel-to-sentence assignment is automatic within each scene; review `output/sync_sheet_audio.csv`.
- When a scene has more sentences than panels, consecutive sentences merge onto one hold panel.

---

## How the Captions Work

No Whisper API needed. The script reads `script.txt`, sends it to ElevenLabs for the voiceover, then **distributes the script text proportionally across the audio timeline** to generate captions. Each word gets a fair share of screen time based on the total audio length.

Captions are chunked at ~10 words per line for readability, with white text + black stroke at the bottom of the frame.

---

## Script Format

Write `script.txt` as plain text. Optional: separate scenes with `---` if you want explicit caption breaks.

**Example:**
```
The story begins in a world where hunters fight monsters from another dimension.

---

Sung Jin-Woo is the weakest of them all — until the day everything changes.
```

Without `---`, the script auto-splits on sentence boundaries.

---

## Ken Burns Effects

Each image gets a randomly-selected effect:
- **Zoom In** (30%) — start wide, end tight, slight upward pan
- **Zoom Out** (25%) — start tight, end wide, slight downward pan
- **Pan Left** (18%) — slight zoom + slow pan right to left
- **Pan Right** (18%) — slight zoom + slow pan left to right
- **Static** (9%) — no movement (used sparingly so it doesn't feel cheap)

Adjust intensity in `config.json`:
- `zoom_intensity: 0.10` = subtle
- `zoom_intensity: 0.15` = moderate (default)
- `zoom_intensity: 0.25` = aggressive

---

## Costs Per Video

Assuming a 10-minute video (~1,500 words):

| Item | Cost |
|---|---|
| ElevenLabs voiceover | ~$0.30 (out of $22/mo Creator plan = 100K credits/mo) |
| Whisper API | $0 (not used — captions come from script) |
| FFmpeg encoding | $0 (free, CPU time) |
| **Total** | **~$0.30/video** |

You can produce ~70 videos per month on a single Creator plan.

---

## Output Specs

- 1920×1080 @ 30fps
- H.264 video, AAC audio
- 6Mbps video bitrate, 192kbps audio
- Typically 500MB–1GB per 10-minute video
- Ready to upload directly to YouTube without re-encoding

---

## Troubleshooting

**"TextClip requires ImageMagick"** — install ImageMagick (see step 1 above).

**"No module named moviepy"** — run `pip install -r requirements.txt`.

**"ElevenLabs API error 401"** — your API key is wrong or expired.

**"ElevenLabs API error 429"** — you're out of credits this month. Upgrade plan or wait for reset.

**Captions look weird/off-timing** — the script's word distribution is approximate. For perfect sync, you'd need Whisper (add it back if you need precise word-level timing).

**"No module named audioop" / pyaudioop** — Python 3.13+ removed the `audioop` module. Install the backport: `pip install pyaudioop`

**Panels still feel off** — open `output/sync_sheet_audio.csv` to review the auto mapping. Use `--legacy-sync` to revert to word-count estimates.

**"timeline panels missing from input/images/"** — audio-sync mode requires every panel in `timeline.json` on disk (e.g. `001-panel.png` through `188-panel.png`). The run aborts before TTS if any are missing. Copy all panel images into `input/images/` with matching filenames.

**"Panel durations do not match voiceover"** — the sync plan and voiceover length disagree by more than 2 seconds. Usually means missing panels, stale `output/voiceover.mp3`, or incomplete `output/voice_chunks/`. Regenerate voiceover or fix images, then retry.

**"Voiceover not found" / silent final video** — ensure `output/voiceover.mp3` exists (run without `--skip-voiceover`, or copy `output/` from a machine that already generated it). After encode, the pipeline verifies the MP4 has an audio track; if it fails, install FFmpeg and confirm `voiceover.mp3` plays in a media player.

**"Encoded video has no audio track"** — FFmpeg failed to mux audio. Install system FFmpeg (`brew install ffmpeg` on macOS), confirm `config.json` has `voiceover_volume` > 0, and verify `output/voiceover.mp3` is not empty.

---

## File Structure

```
manhwa_assembler/
├── assembly_script.py             # Thin CLI entry point
├── config.example.json            # Template (copy to config.json)
├── config.json                    # Local API keys (gitignored)
├── requirements.txt
├── README.md
├── input/
│   ├── script.txt
│   ├── timeline.json
│   ├── images/
│   └── music.mp3                  # optional
├── output/                        # gitignored
└── modules/
    ├── config.py                  # AssemblyConfig, ProjectPaths
    ├── pipeline.py                # AssemblyPipeline orchestrator
    ├── common/
    │   └── utils.py               # Logging, validation, formatting
    ├── sync/
    │   ├── script.py              # Parse script.txt
    │   ├── timeline.py            # Load timeline.json
    │   └── mapper.py              # Panel/sentence sync plan
    ├── audio/
    │   └── voice.py               # ElevenLabs TTS + chunk concat
    └── video/
        ├── assembler.py           # MoviePy composition
        ├── images.py              # Ken Burns effects
        └── captions.py            # SRT generation (optional)
```

---

## Development

Validate the pipeline without API calls or rendering:

```bash
python assembly_script.py --dry-run
python assembly_script.py --dry-run --legacy-sync
```

Module map:
- **`modules/pipeline.py`** — orchestrates audio-sync and legacy flows
- **`modules/sync/`** — script parsing, timeline I/O, panel-to-sentence mapping
- **`modules/audio/`** — per-sentence ElevenLabs generation
- **`modules/video/`** — image processing and final MP4 export

Deprecated top-level shims (`modules/voice_generator.py`, etc.) re-export from the new subpackages for backward compatibility.

---

## Next Steps

After the basic pipeline works, common extensions:
1. **Intro/outro clips** — prepend a 5-sec channel intro and append a subscribe CTA
2. **Sound effects** — drop SFX on dramatic panels (impact, whoosh, etc.)
3. **Auto-thumbnail** — generate a thumbnail from the most dramatic panel
4. **Batch mode** — process multiple chapters in one run
5. **Auto-upload** — push the finished video to YouTube via the YouTube Data API
