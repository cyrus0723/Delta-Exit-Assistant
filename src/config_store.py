from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


def app_root() -> Path:
    """
    Root folder for runtime files.
    - dev: project root
    - frozen: folder containing exe (onedir: dist/Delta-Exit-Assistant)
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def exe_dir() -> Path:
    # kept for backward compatibility
    return app_root()


def config_path() -> Path:
    return app_root() / "config.json"


def load_config() -> Dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: Dict[str, Any]) -> None:
    p = config_path()
    try:
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # last resort: ignore
        pass