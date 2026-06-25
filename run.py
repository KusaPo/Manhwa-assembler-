#!/usr/bin/env python3
"""
Automated Manhwa Recap Video Generator
======================================
CLI entry point for the 4-phase recap pipeline.

Usage:
  python run.py                    # full pipeline
  python run.py --phase script     # Phase 1 only
  python run.py --phase audio      # Phase 2 only
  python run.py --phase vision     # Phase 3 only
  python run.py --phase assembly   # Phase 4 only
  python run.py --skip-tts         # reuse cached voice chunks
  python run.py --preview          # render first 60s
"""

import argparse
from pathlib import Path

from core.config import ProjectPaths, RecapConfig
from core.utils import setup_logging
from engines.pipeline import Phase, PipelineArgs, RecapPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Manhwa Recap Video Generator"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "script", "audio", "vision", "assembly"],
        default="all",
        help="Run a single pipeline phase or all (default: all)",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Reuse cached per-sentence voice chunks",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render only first 60 seconds (assembly phase)",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default=None,
        help="Fetch raw chapter text from URL (script phase)",
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip burning subtitles into the video",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file",
    )
    ns = parser.parse_args()

    config = RecapConfig.from_json(Path(ns.config))
    paths = ProjectPaths.from_config(config)
    setup_logging(str(paths.log_file))

    args = PipelineArgs(
        phase=Phase(ns.phase),
        skip_tts=ns.skip_tts,
        preview=ns.preview,
        source_url=ns.source_url,
        burn_captions=not ns.no_captions,
    )
    RecapPipeline(config, paths).run(args)


if __name__ == "__main__":
    main()
