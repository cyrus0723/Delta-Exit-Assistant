# src/app.py
from __future__ import annotations

from single_instance import SingleInstance


def main() -> None:
    instance = SingleInstance()
    try:
        if instance.already_running:
            instance.show_existing_window()
            return
        from ui_tray import TrayApp
        TrayApp(instance).run()
    finally:
        instance.close()


if __name__ == "__main__":
    main()
