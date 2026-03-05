# src/profiles.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from paths import assets_dir, ensure_assets_dirs
from roi import RoiRel


@dataclass(frozen=True)
class TemplateItem:
    id: str
    label: str
    path: str  # relative path like "assets/templates/xxx/win.png"


@dataclass(frozen=True)
class GameProfile:
    id: str
    display_name: str
    roi_rel: RoiRel
    templates: List[TemplateItem]


def resolve_path(rel_path: str) -> str:
    """
    Always resolve to external assets next to exe.
    """
    rel_path = rel_path.replace("\\", "/").strip()
    return str((assets_dir().parent / rel_path).resolve())


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_profile(data: dict) -> Optional[GameProfile]:
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

        tpls: List[TemplateItem] = []
        for t in data.get("templates", []):
            tid = str(t["id"]).strip()
            label = str(t.get("label", tid)).strip()
            path = str(t["path"]).replace("\\", "/").strip()
            tpls.append(TemplateItem(id=tid, label=label, path=path))

        if not pid:
            return None

        return GameProfile(id=pid, display_name=display_name, roi_rel=roi_rel, templates=tpls)
    except Exception:
        return None


def load_profiles_from_assets() -> List[GameProfile]:
    ensure_assets_dirs()
    prof_dir = assets_dir() / "profiles"
    by_id: Dict[str, GameProfile] = {}

    for fp in sorted(prof_dir.glob("*.json")):
        data = _safe_read_json(fp)
        if not data:
            continue
        p = _parse_profile(data)
        if p:
            by_id[p.id] = p

    return list(by_id.values())


def normalize_profile_id(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = s.strip("_")
    return s or "game"


def profile_file_path(profile_id: str) -> Path:
    ensure_assets_dirs()
    return assets_dir() / "profiles" / f"{profile_id}.json"


def templates_dir(profile_id: str) -> Path:
    ensure_assets_dirs()
    d = assets_dir() / "templates" / profile_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_profile(profile: GameProfile) -> Path:
    out = profile_file_path(profile.id)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "id": profile.id,
        "display_name": profile.display_name,
        "roi_rel": {"x": profile.roi_rel.x, "y": profile.roi_rel.y, "w": profile.roi_rel.w, "h": profile.roi_rel.h},
        "templates": [{"id": t.id, "label": t.label, "path": t.path} for t in profile.templates],
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out


def pick_profile(profiles: List[GameProfile], selected_id: str) -> GameProfile:
    for p in profiles:
        if p.id == selected_id:
            return p
    return profiles[0]