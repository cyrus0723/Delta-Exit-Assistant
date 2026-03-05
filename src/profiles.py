# src/profiles.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config_store import exe_dir
from roi import RoiRel


@dataclass(frozen=True)
class TemplateItem:
    id: str
    label: str
    path: str  # relative path like "assets/templates/<game>/<file>.png"


@dataclass(frozen=True)
class GameProfile:
    id: str
    display_name: str
    roi_rel: RoiRel
    templates: List[TemplateItem]


# -------------------------
# Path policy (NO onefile / NO _MEIPASS)
# -------------------------

def runtime_root() -> Path:
    """
    Single source of truth for IO root:
    - frozen: directory where exe is located
    - dev: project root (same behavior as config_store.exe_dir())
    """
    return exe_dir()


def resolve_resource_path(rel_path: str) -> str:
    """
    Always resolve from exe_dir()/rel_path.
    Assets are expected to be external and writable (installer layout).
    """
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    return str(runtime_root() / rel_path)


def assets_dir() -> Path:
    return runtime_root() / "assets"


def profiles_dir() -> Path:
    return assets_dir() / "profiles"


def templates_dir() -> Path:
    return assets_dir() / "templates"


def ensure_assets_layout() -> None:
    profiles_dir().mkdir(parents=True, exist_ok=True)
    templates_dir().mkdir(parents=True, exist_ok=True)


# -------------------------
# Load / Save
# -------------------------

def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _scan_templates_for_profile(profile_id: str) -> List[TemplateItem]:
    """
    Auto scan assets/templates/<profile_id>/*.{png,jpg,jpeg,bmp}
    This enables "dynamic captured templates" without editing profile.json.
    """
    base = templates_dir() / profile_id
    if not base.exists():
        return []

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    items: List[TemplateItem] = []
    for fp in sorted(base.glob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in exts:
            continue
        stem = fp.stem.strip() or "template"
        tid = f"{profile_id}_{stem}"
        rel = f"assets/templates/{profile_id}/{fp.name}"
        items.append(TemplateItem(id=tid, label=stem, path=rel))
    return items


def load_profiles_from_assets() -> List[GameProfile]:
    """
    Load from: exe_dir()/assets/profiles/*.json
    - If templates field is missing/empty, auto-scan templates directory.
    """
    ensure_assets_layout()

    out: List[GameProfile] = []
    for fp in sorted(profiles_dir().glob("*.json")):
        data = _safe_read_json(fp)
        if not data:
            continue

        try:
            pid = str(data.get("id", "")).strip() or fp.stem
            display_name = str(data.get("display_name", pid)).strip() or pid

            rr = data.get("roi_rel") or {}
            roi_rel = RoiRel(
                x=float(rr.get("x", 0.0)),
                y=float(rr.get("y", 0.0)),
                w=float(rr.get("w", 0.0)),
                h=float(rr.get("h", 0.0)),
            )

            tpls: List[TemplateItem] = []
            for t in (data.get("templates") or []):
                tid = str(t.get("id", "")).strip()
                label = str(t.get("label", tid)).strip() or tid
                path = str(t.get("path", "")).replace("\\", "/").strip()
                if not tid or not path:
                    continue
                tpls.append(TemplateItem(id=tid, label=label, path=path))

            # If json has no templates, auto scan assets/templates/<pid>/
            if not tpls:
                tpls = _scan_templates_for_profile(pid)

            # Allow profile to exist with empty templates (still selectable),
            # but detector will do nothing until templates are added.
            out.append(
                GameProfile(
                    id=pid,
                    display_name=display_name,
                    roi_rel=roi_rel,
                    templates=tpls,
                )
            )
        except Exception:
            continue

    return out


def save_profile_json(profile_id: str, display_name: str, roi_rel: RoiRel, templates: Optional[List[TemplateItem]] = None) -> Path:
    ensure_assets_layout()
    p = profiles_dir() / f"{profile_id}.json"
    payload = {
        "id": profile_id,
        "display_name": display_name,
        "roi_rel": {"x": roi_rel.x, "y": roi_rel.y, "w": roi_rel.w, "h": roi_rel.h},
        "templates": [
            {"id": t.id, "label": t.label, "path": t.path.replace("\\", "/")}
            for t in (templates or [])
        ],
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def create_new_game_skeleton(profile_id: str) -> tuple[Path, Path]:
    """
    Create:
      - assets/profiles/<id>.json   (roi暂为空，templates为空)
      - assets/templates/<id>/      (空目录)
    Return: (profile_json_path, template_dir_path)
    """
    ensure_assets_layout()
    prof_path = profiles_dir() / f"{profile_id}.json"
    tpl_dir = templates_dir() / profile_id

    tpl_dir.mkdir(parents=True, exist_ok=True)

    # Create json if not exists; keep existing if already present
    if not prof_path.exists():
        roi_rel = RoiRel(x=0.0, y=0.0, w=0.0, h=0.0)
        save_profile_json(profile_id, profile_id, roi_rel, templates=[])
    return prof_path, tpl_dir


def is_valid_game_id(game_id: str) -> bool:
    """
    Conservative rule to avoid filesystem/path issues.
    """
    game_id = game_id.strip()
    if not game_id:
        return False
    if len(game_id) > 32:
        return False
    return re.fullmatch(r"[A-Za-z0-9_\-]+", game_id) is not None


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