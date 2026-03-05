# src/paths.py
from __future__ import annotations

import sys
from pathlib import Path


def app_dir() -> Path:
    """
    Base dir for the whole app:
    - frozen: directory containing the exe
    - dev: project root (src/..)
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def assets_dir() -> Path:
    return app_dir() / "assets"


def ensure_assets_dirs() -> None:
    (assets_dir() / "profiles").mkdir(parents=True, exist_ok=True)
    (assets_dir() / "templates").mkdir(parents=True, exist_ok=True)