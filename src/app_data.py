from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_DATA_DIRNAME = "DeltaExitAssistant"


def bundled_resource_root() -> Path:
    """Return the directory containing bundled, read-only application files."""
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))  # type: ignore[arg-type]
    return Path(__file__).resolve().parent.parent


def legacy_runtime_dir() -> Path:
    """Return the old portable application's directory for one-time migration."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Return the per-user writable location for settings and custom assets."""
    override = os.environ.get("DELTA_EXIT_ASSISTANT_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / APP_DATA_DIRNAME


def _copy_if_missing(source: Path, destination: Path) -> None:
    if destination.exists() or not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_if_changed_from_bundle(source: Path, bundle: Path, destination: Path) -> None:
    if destination.exists() or not source.is_file():
        return
    if bundle.is_file() and source.read_bytes() == bundle.read_bytes():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def migrate_legacy_user_data() -> None:
    """Migrate portable-version preferences and custom assets once, when present.

    The old application wrote beside its EXE. Bundled assets are intentionally
    skipped: only files differing from the bundled default are copied.
    """
    if not getattr(sys, "frozen", False):
        return

    legacy_root = legacy_runtime_dir()
    bundle_root = bundled_resource_root()
    destination_root = user_data_dir()

    try:
        _copy_if_missing(legacy_root / "config.json", destination_root / "config.json")

        legacy_sounds = legacy_root / "sounds"
        if legacy_sounds.exists():
            for source in legacy_sounds.rglob("*"):
                if source.is_file():
                    _copy_if_missing(source, destination_root / source.relative_to(legacy_root))

        legacy_assets = legacy_root / "assets"
        for category in ("profiles", "templates"):
            source_dir = legacy_assets / category
            if not source_dir.exists():
                continue
            for source in source_dir.rglob("*"):
                if source.is_file():
                    relative_path = source.relative_to(legacy_root)
                    _copy_if_changed_from_bundle(
                        source,
                        bundle_root / relative_path,
                        destination_root / relative_path,
                    )
    except OSError:
        # Startup must remain usable even if a legacy folder is unreadable.
        pass
