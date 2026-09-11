# src/profiles.py
from __future__ import annotations

import json
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
    Load profiles from:
      1) runtime_root/assets/profiles/*.json (if exists)  [optional override]
      2) resource_root/assets/profiles/*.json            [bundled]
    """
    profiles: List[GameProfile] = []

    # first runtime override
    for base in [runtime_root(), resource_root()]:
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

                profiles.append(
                    GameProfile(
                        id=pid,
                        display_name=display_name,
                        estimated_duration_min=estimated_duration_min,
                        roi_rel=roi_rel,
                        templates=tpls,
                    )
                )
            except Exception:
                continue

        # 如果 runtime 里找到了 profiles，就优先用 runtime，不再混用 bundle
        if profiles and base == runtime_root():
            return profiles

    return profiles


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
