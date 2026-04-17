"""Shared helpers for the experiments/tools validation toolchain."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_hops_importable() -> None:
    src = str(repo_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def write_generated_yaml(path: Path, banner: str, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(banner)
        yaml.safe_dump(document, handle, sort_keys=False)
