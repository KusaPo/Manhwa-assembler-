"""Recap pipeline orchestrator — four phases."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.config import ProjectPaths, RecapConfig
from core.models import ScriptDocument
from core.utils import (
    format_duration,
    validate_panels_dir,
    validate_script_input,
    validate_strips_dir,
)
from engines.audio_generator import AudioGenerator, get_audio_duration
from engines.image_processor import ImageProcessor
from engines.script_generator import ScriptGenerator
from engines.video_assembler import VideoAssembler

logger = logging.getLogger("engines.pipeline")


class Phase(str, Enum):
    ALL = "all"
    SCRIPT = "script"
    AUDIO = "audio"
    VISION = "vision"
    ASSEMBLY = "assembly"


@dataclass
class PipelineArgs:
    phase: Phase = Phase.ALL
    skip_tts: bool = False
    preview: bool = False
    source_url: Optional[str] = None
    burn_captions: bool = True


class RecapPipeline:
    def __init__(self, config: RecapConfig, paths: ProjectPaths) -> None:
        self.config = config
        self.paths = paths

    def run(self, args: PipelineArgs) -> None:
        self.paths.ensure_output_dir()

        logger.info("=" * 60)
        logger.info("MANHWA RECAP GENERATOR")
        logger.info("=" * 60)

        script: Optional[ScriptDocument] = None

        if args.phase in (Phase.ALL, Phase.SCRIPT):
            if not validate_script_input(
                self.paths.raw_chapter_file,
                self.paths.script_txt,
                self.paths.script_json,
            ) and args.phase == Phase.SCRIPT:
                sys.exit(1)
            if args.phase == Phase.SCRIPT or (
                args.phase == Phase.ALL
                and not self.paths.script_json.exists()
                and not self.paths.script_txt.exists()
            ):
                logger.info("[Phase 1] Script generation...")
                script = ScriptGenerator(self.config, self.paths).generate(
                    source_url=args.source_url
                )
            else:
                script = ScriptGenerator(self.config, self.paths).load_or_generate()

        if args.phase in (Phase.ALL, Phase.AUDIO):
            if script is None:
                script = ScriptGenerator(self.config, self.paths).load_or_generate()
            logger.info("[Phase 2] Audio generation...")
            AudioGenerator(self.config, self.paths).generate(
                script, skip_existing=args.skip_tts
            )

        if args.phase in (Phase.ALL, Phase.VISION):
            if not validate_strips_dir(self.paths.raw_strips_dir):
                if args.phase == Phase.VISION:
                    sys.exit(1)
                logger.warning("  Skipping vision phase — no raw strips")
            else:
                logger.info("[Phase 3] Vision processing...")
                ImageProcessor(self.config, self.paths).process_strips()

        if args.phase in (Phase.ALL, Phase.ASSEMBLY):
            if not self.paths.voiceover_file.exists():
                logger.error(f"Missing voiceover: {self.paths.voiceover_file}")
                sys.exit(1)
            if not self.paths.timestamps_file.exists():
                logger.error(f"Missing timestamps: {self.paths.timestamps_file}")
                sys.exit(1)
            if not validate_panels_dir(self.paths.panels_dir):
                sys.exit(1)

            vo_dur = get_audio_duration(self.paths.voiceover_file)
            logger.info(f"  Voiceover ready: {format_duration(vo_dur)}")

            panel_paths = ImageProcessor(self.config, self.paths).load_panel_paths()
            logger.info("[Phase 4] Video assembly...")
            VideoAssembler(self.config, self.paths).assemble(
                panel_paths,
                preview_mode=args.preview,
                burn_captions=args.burn_captions,
            )

            size_mb = os.path.getsize(self.paths.final_video) / (1024 * 1024)
            logger.info("=" * 60)
            logger.info("COMPLETE")
            logger.info(f"Output: {self.paths.final_video}")
            logger.info(f"Duration: {format_duration(vo_dur)}")
            logger.info(f"Size: {size_mb:.1f} MB")
            logger.info("=" * 60)
