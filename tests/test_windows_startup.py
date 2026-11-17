from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import windows_startup
from ui_main import MainWindow


class WindowsStartupTests(unittest.TestCase):
    def test_enable_writes_quoted_executable(self):
        key = MagicMock()
        with patch.object(windows_startup.sys, "frozen", True, create=True), \
             patch.object(windows_startup.sys, "executable", r"C:\Program Files\Assistant\app.exe"), \
             patch.object(windows_startup.winreg, "CreateKeyEx") as create, \
             patch.object(windows_startup.winreg, "SetValueEx") as set_value:
            create.return_value.__enter__.return_value = key
            windows_startup.set_enabled(True)
        set_value.assert_called_once_with(key, windows_startup.VALUE_NAME, 0,
                                          windows_startup.winreg.REG_SZ,
                                          r'"C:\Program Files\Assistant\app.exe"')

    def test_disable_missing_entry_is_harmless(self):
        with patch.object(windows_startup.winreg, "OpenKey", side_effect=FileNotFoundError):
            windows_startup.set_enabled(False)
            self.assertFalse(windows_startup.is_enabled())

    def test_failure_restores_switch_and_reports_error(self):
        window = MainWindow.__new__(MainWindow)
        window.start_with_windows = Mock()
        window.start_with_windows.get.return_value = True
        window.dialogs = Mock()
        with patch("windows_startup.set_enabled", side_effect=PermissionError("denied")):
            window._set_windows_startup()
        window.start_with_windows.set.assert_called_once_with(False)
        window.dialogs.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
