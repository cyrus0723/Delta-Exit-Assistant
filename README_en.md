# 🎮 Delta Exit Assistant

[简体中文](README.md) | [日本語](README_ja.md) | [English](README_en.md)

Delta Exit Assistant is a Windows tray application that detects game result screens and reminds you to stop playing. It uses local screenshots and OpenCV template matching only. It does not inject into games, read memory, modify game files, or automate input.

## Features

* Tray application with start, stop, and quit controls
* Game profiles with relative ROI coordinates and result templates
* Built-in profiles for Delta Force and VALORANT
* Custom game workflow: add a game, select ROI, and capture templates
* Cooldown and hysteresis to avoid repeated notifications on the same screen
* Notification modes: toast, sound, or both
* Sleep reminder window with a global bed time and per-game lead time
* UI languages: Chinese, Japanese, and English

## Sleep Reminder Window

Sleep reminders are enabled by default. The global bed time is stored in `config.json`. Each game profile stores its own `sleep_lead_minutes`.

For example, if the bed time is `22:00` and the selected VALORANT profile uses `45` minutes, the reminder window is `21:15 ~ 22:00`. During that window, every newly detected result screen triggers a sleep reminder.

Default global settings:

```json
{
  "ui_language": "en",
  "sleep_reminder_enabled": true,
  "sleep_bed_time": "22:00",
  "sleep_stop_after_bed_time": true,
  "sleep_auto_start_detection": true,
  "sleep_title_tpl": "{game} sleep reminder",
  "sleep_msg_tpl": "It is time to rest. Stop playing after this match."
}
```

Use `Settings → Sleep reminder` to change the bed time, the current game's lead time, automatic detection startup, and sleep reminder text.

## Change UI Language

Right-click the tray icon and open `Settings → UI language`. Choose `中文`, `日本語`, or `English`. The tray menu and subsequent dialogs update immediately. Custom notification text is preserved.

## Profile Format

Each game uses an `assets/profiles/<game>.json` file:

```json
{
  "id": "valorant",
  "display_name": "VALORANT",
  "sleep_lead_minutes": 45,
  "roi_rel": { "x": 0.34, "y": 0.30, "w": 0.327, "h": 0.401 },
  "templates": [
    { "id": "valorant_win", "label": "Win", "path": "assets/templates/valorant/win.png" },
    { "id": "valorant_lose", "label": "Loss", "path": "assets/templates/valorant/lose.png" }
  ]
}
```

Older profiles without `sleep_lead_minutes` default to `30`.

## Usage

1. Run `Delta-Exit-Assistant.exe`.
2. Right-click the tray icon and select a game.
3. Click `Start detection`.
4. When a result screen is matched, the app sends a toast, plays a sound, or does both depending on settings.

For a custom game, choose `Add game…`, drag to select the result screen ROI, then capture win, loss, and optional draw templates from the tray menu.

## Development

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Package as a writable onedir ZIP distribution:

```powershell
python -m PyInstaller --clean --noconfirm -w --onedir --paths "src" `
  "src/app.py" --name "Delta-Exit-Assistant" --add-data "assets;assets"
```

Extract releases to a normal writable folder so the application can save `config.json`, profiles, and templates.

## Roadmap

* Multi-monitor support with explicit game-monitor selection. The current version captures the primary monitor only.

## License

MIT. Keep the original attribution and license text when redistributing derived versions.
