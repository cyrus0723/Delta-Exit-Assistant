# src/app.py
from __future__ import annotations

from ui_tray import TrayApp


def main() -> None:
    TrayApp().run()


if __name__ == "__main__":
    main()