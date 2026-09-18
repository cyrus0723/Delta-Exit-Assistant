# src/config_store.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from app_data import legacy_runtime_dir, user_data_dir


def config_path() -> Path:
    return user_data_dir() / "config.json"


def legacy_config_path() -> Path:
    return legacy_runtime_dir() / "config.json"


def load_config() -> Dict[str, Any]:
    p = config_path()
    if not p.exists():
        p = legacy_config_path()
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
        p.parent.mkdir(parents=True, exist_ok=True)
        temp = p.with_name(f"{p.name}.tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        temp.replace(p)
    except OSError:
        return
    except TypeError:
        # Keep the previous behavior of ignoring unserializable config values.
        return
