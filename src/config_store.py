# src/config_store.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from paths import app_dir


def config_path() -> Path:
    return app_dir() / "config.json"


def load_config() -> Dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg: Dict[str, Any]) -> None:
    p = config_path()
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass