from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from roi import RoiRel
from config_store import app_root


@dataclass(frozen=True)
class TemplateItem:
    id: str
    label: str
    path: str  # relative path like "assets/templates/xxx.png"


@dataclass(frozen=True)
class GameProfile:
    id: str
    display_name: str
    roi_rel: RoiRel
    templates: List[TemplateItem]


def assets_root() -> Path:
    """
    Writable assets root beside exe:
      <app_root>/assets
    """
    return app_root() / "assets"


def resolve_resource_path(rel_path: str) -> str:
    """
    Always resolve from runtime app_root (writable assets),
    because you decided to ship assets folder next to exe.
    """
    rel_path = rel_path.replace("\\", "/").strip()
    return str((app_root() / rel_path).resolve())


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

        if not pid or not tpls:
            return None

        return GameProfile(id=pid, display_name=display_name, roi_rel=roi_rel, templates=tpls)
    except Exception:
        return None


def load_profiles_from_assets() -> List[GameProfile]:
    """
    Load ONLY from writable runtime assets:
      <app_root>/assets/profiles/*.json
    """
    by_id: Dict[str, GameProfile] = {}
    prof_dir = assets_root() / "profiles"
    if not prof_dir.exists():
        return []

    for fp in sorted(prof_dir.glob("*.json")):
        data = _safe_read_json(fp)
        if not data:
            continue
        p = _parse_profile(data)
        if p:
            by_id[p.id] = p
    return list(by_id.values())


def fallback_delta_profile_from_legacy_config(cfg: dict) -> GameProfile:
    roi_rel = RoiRel(x=0.078125, y=0.138889, w=0.260417, h=0.148148)
    return GameProfile(
        id="delta",
        display_name="Delta",
        roi_rel=roi_rel,
        templates=[
            TemplateItem(id="delta_success", label="撤离成功", path="assets/templates/delta/success.png"),
            TemplateItem(id="delta_fail", label="撤离失败", path="assets/templates/delta/fail.png"),
        ],
    )


def pick_profile(profiles: List[GameProfile], selected_id: str) -> GameProfile:
    for p in profiles:
        if p.id == selected_id:
            return p
    return profiles[0]


def normalize_profile_id(name: str) -> str:
    """
    Safe id for filename/folder.
    """
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = s.strip("_")
    return s or "game"


def profile_file_path(profile_id: str) -> Path:
    return assets_root() / "profiles" / f"{profile_id}.json"


def save_profile(profile: GameProfile) -> Path:
    """
    Save profile into <app_root>/assets/profiles/<id>.json
    """
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