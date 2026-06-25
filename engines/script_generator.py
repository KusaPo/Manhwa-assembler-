"""
Script Engine — OpenAI recap generation and sanitization.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

import requests
from openai import OpenAI

from core.config import RecapConfig, ProjectPaths
from core.models import ScriptDocument

logger = logging.getLogger("engines.script")


def _load_sanitize_rules(prompts_dir: Path) -> List[tuple[str, str]]:
    rules_path = prompts_dir / "sanitize_rules.txt"
    if not rules_path.exists():
        return []
    pairs: List[tuple[str, str]] = []
    for line in rules_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            src, dst = line.split("->", 1)
            pairs.append((src.strip(), dst.strip()))
    return pairs


def sanitize_text(text: str, rules: List[tuple[str, str]]) -> str:
    result = text
    for src, dst in rules:
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        result = pattern.sub(dst, result)
    return result


def fetch_url_to_file(url: str, dest: Path) -> Path:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(response.text, encoding="utf-8")
    return dest


class ScriptGenerator:
    def __init__(self, config: RecapConfig, paths: ProjectPaths) -> None:
        self.config = config
        self.paths = paths
        self._sanitize_rules = _load_sanitize_rules(paths.prompts_dir)

    def generate(self, *, source_url: str | None = None) -> ScriptDocument:
        raw_path = self.paths.raw_chapter_file
        if source_url:
            logger.info(f"  Fetching chapter from URL...")
            fetch_url_to_file(source_url, raw_path)

        if raw_path.exists() and raw_path.read_text(encoding="utf-8").strip():
            raw_text = raw_path.read_text(encoding="utf-8").strip()
            script_text = self._rewrite_with_llm(raw_text)
        elif self.paths.script_txt.exists():
            logger.info("  Using existing input/script.txt (skipping LLM)")
            document = ScriptDocument.from_plaintext(self.paths.script_txt)
            document.save(self.paths.script_json)
            return document
        else:
            raise FileNotFoundError(
                f"No input found. Add {raw_path} or {self.paths.script_txt}"
            )

        script_text = sanitize_text(script_text, self._sanitize_rules)
        temp_path = self.paths.output_dir / "_generated_script.txt"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(script_text, encoding="utf-8")

        document = ScriptDocument.from_plaintext(temp_path)
        document.save(self.paths.script_json)
        logger.info(
            f"  Script saved: {self.paths.script_json} "
            f"({len(document.all_sentences)} sentences)"
        )
        return document

    def _rewrite_with_llm(self, raw_text: str) -> str:
        if not self.config.openai_api_key or self.config.openai_api_key == "your-key-here":
            raise ValueError(
                "OpenAI API key not configured. Set openai_api_key in config.json "
                "or provide input/script.txt to skip Phase 1."
            )

        system_path = self.paths.prompts_dir / "recap_system.txt"
        system_prompt = system_path.read_text(encoding="utf-8")

        client = OpenAI(api_key=self.config.openai_api_key)
        logger.info(f"  Calling OpenAI ({self.config.openai_model})...")
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def load_or_generate(self) -> ScriptDocument:
        if self.paths.script_json.exists():
            return ScriptDocument.load(self.paths.script_json)
        if self.paths.script_txt.exists():
            doc = ScriptDocument.from_plaintext(self.paths.script_txt)
            doc.save(self.paths.script_json)
            return doc
        return self.generate()
