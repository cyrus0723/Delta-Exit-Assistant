"""Manage the current user's Windows logon startup entry."""
import subprocess
import sys
import winreg
from pathlib import Path

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "DeltaExitAssistant"


def startup_command():
    if getattr(sys, "frozen", False):
        args = [sys.executable]
    else:
        executable = Path(sys.executable)
        pythonw = executable.with_name("pythonw.exe")
        args = [str(pythonw if pythonw.exists() else executable), str(Path(__file__).resolve().with_name("app.py"))]
    return subprocess.list2cmdline(args)


def is_enabled():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, VALUE_NAME)
        return True
    except FileNotFoundError:
        return False


def set_enabled(enabled):
    if enabled:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, startup_command())
    else:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, VALUE_NAME)
        except FileNotFoundError:
            pass
