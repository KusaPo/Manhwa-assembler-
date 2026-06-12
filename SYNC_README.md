# Script-Synced Pacing Patch

This patch makes the assembler time panels to the **narration**, instead of
spreading every panel evenly across the voiceover.

## The problem it fixes

The original `video_assembler.py` did:

    per_image_duration = total_duration / len(image_paths)

So every panel got the same screen time (~2.03s for 243 panels over 493s).
But the panels are *not* evenly distributed across the story — the temple
scene alone is 109 of the 243 panels. Result: the temple section ran for
**3:41** on screen while its narration is only **2:09**, and the cop fight ran
**2:01** against **2:58** of narration. Visuals and voiceover drifted apart.

## How the fix works

1. The script is split into its 4 `---` segments (the same split
   `script_processor` already uses). Each segment's share of the total word
   count sets its share of the audio time:

   | Scene | Panels | Narration time |
   |-------|--------|----------------|
   | S1 Alley / cop fight        |  60 | 2:58 |
   | S2 Shipping yard / Bang     |  40 | 1:54 |
   | S3 Temple / Buddha          | 109 | 2:09 |
   | S4 Hospital / reincarnation |  34 | 1:11 |

2. Each panel was assigned to the scene it depicts (read off the dialogue in
   the panels and matched to the script). Panels stay in chapter reading order
   *within* a scene — fight sequences are never shuffled.

3. Each scene's time is divided among its panels. Panels that contain dialogue
   linger ~30% longer than pure action panels, so the voiceover lands on the
   talking panels.

The full per-panel schedule is in **`timeline.json`** (machine-readable) and
**`sync_sheet.csv`** (open in any spreadsheet to review / tweak).

## Install

1. Replace these two files in your project:
   - `assembly_script.py`
   - `modules/video_assembler.py`
2. Copy `timeline.json` into `input/` (next to `script.txt`).
3. Make sure `input/images/` holds the 243 panels named
   `001-panel.png` … `243-panel.png` (the set from `webtoon-panels-ordered.zip`).
4. Run as usual:

       python assembly_script.py --skip-voiceover   # reuse existing voiceover.mp3

`--preview` still works (renders the panels that fall in the first 60s).
If `timeline.json` is absent, the assembler silently falls back to the old
even-spread behaviour, so this change is safe.

## Adjusting the timing yourself

Everything keys off `timeline.json`. To give a panel more/less time, edit its
`dur` value (seconds) — the assembler rescales all durations to lock exactly to
the voiceover length, so they don't need to sum perfectly. To move a panel
earlier/later in the video, reorder its entry in the `panels` list.

## Honest caveats

- Sync is at the **scene** level, not word-perfect. The recap paraphrases the
  original dialogue and the chapter tells parts out of the recap's order, so a
  panel can land a few seconds off the exact line. Scene-level sync removes the
  big drift; perfect word-level sync would need real audio timestamps (Whisper).
- The S1/S2 boundary (cop fight → shipyard) is the fuzziest; a handful of
  transition panels around there could belong to either scene. Adjust in
  `sync_sheet.csv` if you disagree.
