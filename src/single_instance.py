"""Hold a session-local Windows mutex for the application's lifetime."""
import ctypes
from ctypes import wintypes


class SingleInstance:
    def __init__(self, name="Local\\DeltaExitAssistant.SingleInstance"):
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
        self._kernel.CreateMutexW.restype = wintypes.HANDLE
        self._kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._kernel.CreateEventW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
        self._kernel.CreateEventW.restype = wintypes.HANDLE
        self._kernel.SetEvent.argtypes = (wintypes.HANDLE,)
        self._kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        self._handle = self._kernel.CreateMutexW(None, False, name)
        error = ctypes.get_last_error()
        if not self._handle:
            raise ctypes.WinError(error)
        self.already_running = error == 183  # ERROR_ALREADY_EXISTS
        self._show_event = self._kernel.CreateEventW(None, False, False, "Local\\DeltaExitAssistant.ShowWindow")

    def show_existing_window(self):
        if self._show_event:
            self._kernel.SetEvent(self._show_event)

    def wait_for_show(self, timeout_ms=500):
        return bool(self._show_event and self._kernel.WaitForSingleObject(self._show_event, timeout_ms) == 0)

    def close(self):
        if self._show_event:
            self._kernel.CloseHandle(self._show_event)
            self._show_event = None
        if self._handle:
            self._kernel.CloseHandle(self._handle)
            self._handle = None
