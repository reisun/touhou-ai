"""Windows named-object ownership prevents competing live controllers."""
import ctypes
from ctypes import wintypes


class ProcessLease:
    def __init__(self, pid):
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.api.CreateMutexW.restype = wintypes.HANDLE
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        self.handle = self.api.CreateMutexW(None, False, f"Local\\TouhouAI.Live.{pid}")
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            self.close()
            raise RuntimeError("another live controller already owns this game")

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
