# Manhwa Video Assembler

Automated Python pipeline that turns manhwa panels + a script into a finished YouTube-ready video.

**Pipeline:** `script.txt` + `images/` + `music.mp3` → ElevenLabs voiceover → timed captions → Ken Burns'd video → 1080p MP4

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

Edit `config.json` and fill in:
- `elevenlabs_api_key` - get from https://elevenlabs.io/app/settings/api-keys
- `elevenlabs_voice_id` - get from https://elevenlabs.io/app/voice-library (pick a voice, copy its ID)

### 3. Set up your project folder

```
manhwa_assembler/
├── input/
│   ├── images/          ← drop panel screenshots here (panel_001.jpg, panel_002.jpg, ...)
│   ├── script.txt       ← your narration text
│   └── music.mp3        ← background track (optional)
├── output/              ← final video appears here
├── assembly_script.py
├── config.json
└── modules/
```

### 4. Run

```bash
python assembly_script.py
```

That's it. Coffee break. Come back in 10–15 min, `output/final_video.mp4` is ready.

---

## Usage Modes

```bash
# Standard run
python assembly_script.py

# Dry run (validate inputs without rendering)
python assembly_script.py --dry-run

# Quick preview (renders only first 60 seconds for QA)
python assembly_script.py --preview

# Reuse existing voiceover.mp3 (saves API credits during iteration)
python assembly_script.py --skip-voiceover

# Speed up/slow down narration
python assembly_script.py --speed 1.1
```

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

**Video is too long/short vs audio** — `total_duration` is locked to audio length. If images feel rushed, add more images; if they feel slow, remove some.

---

## File Structure

```
manhwa_assembler/
├── assembly_script.py             # Main entry point
├── config.json                     # API keys + settings
├── requirements.txt
├── README.md
└── modules/
    ├── __init__.py
    ├── script_processor.py        # Parses script.txt
    ├── voice_generator.py         # ElevenLabs API client
    ├── caption_generator.py       # Builds .srt from script
    ├── image_processor.py         # Ken Burns effects
    ├── video_assembler.py         # MoviePy composition
    └── utils.py                   # Logging + validation
```

---

## Next Steps

After the basic pipeline works, common extensions:
1. **Intro/outro clips** — prepend a 5-sec channel intro and append a subscribe CTA
2. **Sound effects** — drop SFX on dramatic panels (impact, whoosh, etc.)
3. **Auto-thumbnail** — generate a thumbnail from the most dramatic panel
4. **Batch mode** — process multiple chapters in one run
5. **Auto-upload** — push the finished video to YouTube via the YouTube Data API
