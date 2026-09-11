# src/profiles.py
from __future__ import annotations

import json
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from roi import RoiRel
from alert_timer import DEFAULT_ESTIMATED_DURATION_MIN, normalize_duration_min


@dataclass(frozen=True)
class TemplateItem:
    id: str
    label: str
    path: str  # relative path like "assets/templates/xxx.png"


@dataclass(frozen=True)
class GameProfile:
    id: str
    display_name: str
    estimated_duration_min: int
    roi_rel: RoiRel
    templates: List[TemplateItem]


def resource_root() -> Path:
    """
    Read-only resource root:
    - dev: project root
    - pyinstaller: sys._MEIPASS (temp)
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))  # type: ignore[arg-type]
    return Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    """
    Writable runtime root:
    - dev: project root
    - frozen exe: directory where the exe is located
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resolve_resource_path(rel_path: str) -> str:
    """
    Resolve path with override priority:
      1) runtime_root()/rel_path  (user-captured templates)
      2) resource_root()/rel_path (bundled assets)
    """
    rel_path = rel_path.replace("\\", "/").strip()

    p_runtime = runtime_root() / rel_path
    if p_runtime.exists():
        return str(p_runtime)

    p_bundle = resource_root() / rel_path
    return str(p_bundle)


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_profiles_from_assets() -> List[GameProfile]:
    """
    Load bundled profiles first, then merge runtime profiles by id. This keeps
    built-in games available when a user adds just one custom profile beside a
    one-file executable.
    """
    by_id: dict[str, GameProfile] = {}

    bases = [resource_root()]
    if runtime_root() != resource_root():
        bases.append(runtime_root())

    for base in bases:
        prof_dir = base / "assets" / "profiles"
        if not prof_dir.exists():
            continue

        for fp in sorted(prof_dir.glob("*.json")):
            data = _safe_read_json(fp)
            if not data:
                continue

            try:
                pid = str(data["id"]).strip()
                display_name = str(data.get("display_name", pid)).strip()
                estimated_duration_min = normalize_duration_min(
                    data.get("estimated_duration_min", DEFAULT_ESTIMATED_DURATION_MIN)
                )

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
                    continue

                by_id[pid] = GameProfile(
                    id=pid,
                    display_name=display_name,
                    estimated_duration_min=estimated_duration_min,
                    roi_rel=roi_rel,
                    templates=tpls,
                )
            except Exception:
                continue

    return list(by_id.values())



def normalize_profile_id(name: str) -> str:
    """Create a stable, filesystem-safe id for a custom game profile."""
    normalized = re.sub(r"\s+", "_", (name or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized).strip("_")
    if normalized:
        return normalized
    digest = hashlib.sha1(name.strip().encode("utf-8")).hexdigest()[:8]
    return f"game_{digest}"


def profile_file_path(profile_id: str) -> Path:
    return runtime_root() / "assets" / "profiles" / f"{profile_id}.json"


def save_profile(profile: GameProfile) -> Path:
    """Persist a custom profile beside the executable for future runs."""
    out = profile_file_path(profile.id)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "id": profile.id,
        "display_name": profile.display_name,
        "estimated_duration_min": profile.estimated_duration_min,
        "roi_rel": {
            "x": profile.roi_rel.x,
            "y": profile.roi_rel.y,
            "w": profile.roi_rel.w,
            "h": profile.roi_rel.h,
        },
        "templates": [
            {"id": template.id, "label": template.label, "path": template.path}
            for template in profile.templates
        ],
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


def fallback_delta_profile_from_legacy_config(cfg: dict) -> GameProfile:
    roi_rel = RoiRel(x=0.078125, y=0.138889, w=0.260417, h=0.148148)
    return GameProfile(
        id="delta",
        display_name="Delta",
        estimated_duration_min=DEFAULT_ESTIMATED_DURATION_MIN,
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
