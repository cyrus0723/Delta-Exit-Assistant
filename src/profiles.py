 # 新：Profile 数据结构 + 加载/校验
 
 # src/profiles.py
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from roi import RoiRel


@dataclass(frozen=True)
class TemplateItem:
    id: str
    label: str
    path: str  # relative to project root / bundle root


@dataclass(frozen=True)
class GameProfile:
    id: str
    display_name: str
    roi_rel: RoiRel
    templates: List[TemplateItem]


def resource_root() -> Path:
    """
    Resource root that works for:
    - dev run: project root (where assets/ exists)
    - pyinstaller: sys._MEIPASS
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))  # type: ignore[arg-type]
    # assume src/ is inside project root
    return Path(__file__).resolve().parent.parent


def resolve_resource_path(rel_path: str) -> str:
    """Resolve a relative resource path to an absolute OS path."""
    p = resource_root() / rel_path
    return str(p)


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_profiles_from_assets() -> List[GameProfile]:
    """
    Load profiles from assets/profiles/*.json.
    If none found / all invalid, return [].
    """
    root = resource_root()
    prof_dir = root / "assets" / "profiles"
    if not prof_dir.exists():
        return []

    profiles: List[GameProfile] = []
    for fp in sorted(prof_dir.glob("*.json")):
        data = _safe_read_json(fp)
        if not data:
            continue

        try:
            pid = str(data["id"]).strip()
            display_name = str(data.get("display_name", pid)).strip()

            rr = data["roi_rel"]
            roi_rel = RoiRel(
                x=float(rr["x"]),
                y=float(rr["y"]),
                w=float(rr["w"]),
                h=float(rr["h"]),
            )

            tpls = []
            for t in data.get("templates", []):
                tid = str(t["id"]).strip()
                label = str(t.get("label", tid)).strip()
                path = str(t["path"]).replace("\\", "/").strip()
                tpls.append(TemplateItem(id=tid, label=label, path=path))

            if not pid or not tpls:
                continue

            profiles.append(
                GameProfile(
                    id=pid,
                    display_name=display_name,
                    roi_rel=roi_rel,
                    templates=tpls,
                )
            )
        except Exception:
            continue

    return profiles


def fallback_delta_profile_from_legacy_config(cfg: dict) -> GameProfile:
    """
    Backward-compatible fallback:
    - ROI from legacy config.json pixel roi_* fields
    - templates from legacy assets/templates/success.png and fail.png
    - convert pixel ROI -> relative ROI using current screen is NOT reliable here
      so we provide a conservative default ROI_rel if legacy not present.
    This is only used if no assets/profiles exist.
    """
    # If legacy config has pixel ROI, we cannot convert to relative without screen size.
    # Provide default ROI_rel similar to your old ROI area.
    # Users should create assets/profiles/delta.json to be accurate.
    roi_rel = RoiRel(x=0.078, y=0.078, w=0.260, h=0.148)

    return GameProfile(
        id="delta",
        display_name="Delta",
        roi_rel=roi_rel,
        templates=[
            TemplateItem(
                id="delta_success",
                label="撤离成功",
                path="assets/templates/success.png",
            ),
            TemplateItem(
                id="delta_fail",
                label="撤离失败",
                path="assets/templates/fail.png",
            ),
        ],
    )


def pick_profile(profiles: List[GameProfile], selected_id: str) -> GameProfile:
    for p in profiles:
        if p.id == selected_id:
            return p
    return profiles[0]