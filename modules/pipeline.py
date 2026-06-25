"""Assembly pipeline orchestrator."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from modules.audio.voice import (
    concatenate_voiceover_chunks,
    generate_voiceover,
    generate_voiceover_chunks,
    get_audio_duration,
)
from modules.common.utils import format_duration, validate_inputs, validate_timeline_images
from modules.config import AssemblyConfig, ProjectPaths
from modules.sync.mapper import (
    build_sync_plan,
    export_sync_plan,
    export_sync_sheet_csv,
    resolve_panel_paths,
)
from modules.sync.script import (
    estimate_sentence_durations,
    flatten_sentences,
    parse_scenes_with_sentences,
    parse_script,
)
from modules.sync.timeline import load_timeline_legacy, load_timeline_panels
from modules.video.assembler import assemble_video
from modules.video.images import load_images

logger = logging.getLogger("assembler")


@dataclass
class AssemblyArgs:
    dry_run: bool = False
    preview: bool = False
    speed: float = 1.0
    skip_voiceover: bool = False
    legacy_sync: bool = False


class AssemblyPipeline:
    def __init__(self, config: AssemblyConfig, paths: ProjectPaths) -> None:
        self.config = config
        self.paths = paths

    def run(self, args: AssemblyArgs) -> None:
        self.paths.ensure_output_dir()

        logger.info("=" * 60)
        logger.info("MANHWA VIDEO ASSEMBLY - STARTING")
        logger.info("=" * 60)

        if not validate_inputs(
            self.paths.script_file,
            self.paths.images_dir,
            self.paths.music_file,
        ):
            logger.error("Input validation failed. Fix issues above and retry.")
            sys.exit(1)

        audio_sync_mode = not args.legacy_sync
        if not validate_timeline_images(
            self.paths.images_dir,
            self.paths.timeline_file,
            audio_sync_mode,
        ):
            logger.error("Timeline image validation failed. Fix issues above and retry.")
            sys.exit(1)

        if audio_sync_mode:
            logger.info("Sync mode: AUDIO-SYNCED (per-sentence TTS durations)")
            image_paths, image_durations, sentence_count = self._run_audio_sync(args)
        else:
            logger.info("Sync mode: LEGACY (timeline.json word-count estimates)")
            image_paths, image_durations, sentence_count = self._run_legacy(args)

        if args.dry_run:
            return

        self._assemble(
            args, image_paths, image_durations, audio_sync_mode, sentence_count
        )

    def _run_audio_sync(
        self, args: AssemblyArgs
    ) -> tuple[List[Path], Optional[List[float]], int]:
        if not self.paths.timeline_file.exists():
            logger.error(
                f"Audio-sync mode requires {self.paths.timeline_file} "
                "(panel order and scene segments)."
            )
            sys.exit(1)

        logger.info("[1/6] Parsing script...")
        scenes = parse_scenes_with_sentences(self.paths.script_file)
        sentences = flatten_sentences(scenes)
        logger.info(f"      Found {len(scenes)} scenes, {len(sentences)} sentences")

        timeline_panels = load_timeline_panels(self.paths.timeline_file)
        logger.info(
            f"      Timeline: {len(timeline_panels)} panels from {self.paths.timeline_file}"
        )

        if args.dry_run:
            sentence_durations = estimate_sentence_durations(sentences)
            logger.info("[2/6] [DRY RUN] Would generate per-sentence voice chunks")
        elif args.skip_voiceover and self.paths.voiceover_file.exists():
            logger.info("[2/6] Skipping chunk generation (reusing cached chunks)")
            sentence_durations = self._load_cached_chunk_durations(sentences)
        else:
            logger.info("[2/6] Generating per-sentence voiceover chunks...")
            sentence_durations = generate_voiceover_chunks(
                sentences=sentences,
                chunk_dir=self.paths.chunk_dir,
                voice_id=self.config.elevenlabs_voice_id,
                api_key=self.config.elevenlabs_api_key,
                speed=args.speed,
                skip_existing=args.skip_voiceover,
            )

        sync_plan = build_sync_plan(
            scenes=scenes,
            timeline_panels=timeline_panels,
            sentence_durations=sentence_durations,
            flash_duration=self.config.extra_panel_flash_duration,
        )
        export_sync_plan(sync_plan, self.paths.sync_plan_file)
        export_sync_sheet_csv(sync_plan, self.paths.sync_sheet_file)

        if args.dry_run:
            total = sum(sync_plan.image_durations)
            logger.info(
                f"[DRY RUN] Estimated video duration: {format_duration(total)}"
            )
            logger.info(f"[DRY RUN] Sync plan written to {self.paths.sync_plan_file}")
            logger.info("[DRY RUN] Skipping rendering. Exiting.")
            return [], None, len(sentences)

        if not (args.skip_voiceover and self.paths.voiceover_file.exists()):
            logger.info("[3/6] Concatenating voice chunks with silence pads...")
            concatenate_voiceover_chunks(
                sentences=sentences,
                chunk_dir=self.paths.chunk_dir,
                output_path=self.paths.voiceover_file,
                silence_after_ms=sync_plan.silence_after_ms,
                inter_sentence_pause_ms=self.config.inter_sentence_pause_ms,
            )
        else:
            logger.info("[3/6] Reusing existing concatenated voiceover.mp3")

        audio_duration = get_audio_duration(self.paths.voiceover_file)
        logger.info(f"[4/6] Voiceover duration: {format_duration(audio_duration)}")

        logger.info("[5/6] Loading panel images...")
        image_paths = self._load_images_or_exit()
        try:
            image_paths, image_durations = resolve_panel_paths(
                sync_plan, self.paths.images_dir, image_paths
            )
        except FileNotFoundError as exc:
            logger.error(str(exc))
            sys.exit(1)
        logger.info(
            f"      Sync plan resolved: {len(image_paths)} panels, "
            f"{format_duration(sum(image_durations))} total"
        )
        return image_paths, image_durations, len(sentences)

    def _run_legacy(
        self, args: AssemblyArgs
    ) -> tuple[List[Path], Optional[List[float]], int]:
        logger.info("[1/6] Parsing script...")
        script_segments = parse_script(self.paths.script_file)
        logger.info(f"      Found {len(script_segments)} script segments")

        if args.skip_voiceover and self.paths.voiceover_file.exists():
            logger.info("[2/6] Skipping voiceover generation (using existing file)")
        elif args.dry_run:
            logger.info("[2/6] [DRY RUN] Would generate voiceover via ElevenLabs")
        else:
            logger.info("[2/6] Generating voiceover via ElevenLabs API...")
            generate_voiceover(
                text=" ".join(script_segments),
                output_path=self.paths.voiceover_file,
                voice_id=self.config.elevenlabs_voice_id,
                api_key=self.config.elevenlabs_api_key,
                speed=args.speed,
            )
            logger.info(f"      Voiceover saved: {self.paths.voiceover_file}")

        if args.dry_run:
            logger.info("[DRY RUN] Skipping rendering. Exiting.")
            return [], None, 0

        audio_duration = get_audio_duration(self.paths.voiceover_file)
        logger.info(f"[3/6] Voiceover duration: {format_duration(audio_duration)}")
        logger.info("[4/6] Captions disabled, skipping...")

        logger.info("[5/6] Loading panel images...")
        image_paths = self._load_images_or_exit()
        image_paths, image_durations = load_timeline_legacy(
            self.paths.timeline_file,
            image_paths,
        )
        return image_paths, image_durations, 0

    def _load_cached_chunk_durations(self, sentences) -> List[float]:
        durations: List[float] = []
        for sentence in sentences:
            chunk_path = self.paths.chunk_dir / f"{sentence.global_idx:03d}.mp3"
            if not chunk_path.exists():
                logger.error(
                    f"Missing cached chunk {chunk_path}. "
                    "Run without --skip-voiceover first."
                )
                sys.exit(1)
            durations.append(get_audio_duration(chunk_path))
        return durations

    def _load_images_or_exit(self) -> List[Path]:
        image_paths = load_images(self.paths.images_dir)
        logger.info(f"      Found {len(image_paths)} images on disk")
        if len(image_paths) == 0:
            logger.error("No images found in input/images/. Add panels and retry.")
            sys.exit(1)
        return image_paths

    def _assemble(
        self,
        args: AssemblyArgs,
        image_paths: List[Path],
        image_durations: Optional[List[float]],
        audio_sync_mode: bool,
        sentence_count: int = 0,
    ) -> None:
        if not self.paths.voiceover_file.exists():
            logger.error(
                f"Voiceover not found: {self.paths.voiceover_file}. "
                "Run without --skip-voiceover or copy output/voiceover.mp3."
            )
            sys.exit(1)

        voiceover_duration = get_audio_duration(self.paths.voiceover_file)
        if voiceover_duration <= 1.0:
            logger.error(
                f"Voiceover is missing or too short ({voiceover_duration:.2f}s): "
                f"{self.paths.voiceover_file}"
            )
            sys.exit(1)

        logger.info(
            f"Voiceover ready: {format_duration(voiceover_duration)} "
            f"({self.paths.voiceover_file})"
        )

        logger.info("[6/6] Assembling video (this takes ~10-15 min)...")
        assemble_config = self.config.as_dict()
        assemble_config["sentence_count"] = sentence_count
        assemble_video(
            image_paths=image_paths,
            voiceover_path=self.paths.voiceover_file,
            music_path=self.paths.music_file,
            captions_path=None,
            output_path=self.paths.final_video,
            config=assemble_config,
            preview_mode=args.preview,
            image_durations=image_durations,
            audio_sync_mode=audio_sync_mode,
        )

        final_size_mb = os.path.getsize(self.paths.final_video) / (1024 * 1024)
        audio_duration = get_audio_duration(self.paths.voiceover_file)
        logger.info("=" * 60)
        logger.info("ASSEMBLY COMPLETE")
        logger.info(f"Output: {self.paths.final_video}")
        logger.info(f"Duration: {format_duration(audio_duration)}")
        logger.info(f"Size: {final_size_mb:.1f} MB")
        logger.info(
            f"Resolution: {self.config.video_width}x{self.config.video_height} "
            f"@{self.config.fps}fps"
        )
        logger.info("Ready for YouTube upload.")
        logger.info("=" * 60)
