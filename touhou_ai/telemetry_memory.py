"""Windows pagefile-backed latest-frame transport, no filesystem data exchange.

A short, nonblocking named mutex protects the copy. Slow or absent viewers never
hold up the game. Each artifacts root gets an isolated channel. The mapping lives
until its last process handle closes; readers retain the latest complete packet.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import threading

CAPACITY = 8 * 1024 * 1024
HEADER = struct.Struct('<QI')
ROOT = Path(__file__).resolve().parents[1] / 'artifacts'


class TelemetryMemory:
    def __init__(self, root=ROOT):
        if os.name != 'nt':
            raise OSError('live shared telemetry requires Windows')
        name = 'Local\\TouhouAITelemetryV1-' + hashlib.sha256(
            str(Path(root).resolve()).casefold().encode()).hexdigest()[:24]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.ReleaseMutex.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.mutex = self.kernel.CreateMutexW(None, False, name + '-mutex')
        if not self.mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            self.memory = mmap.mmap(-1, CAPACITY, tagname=name)
        except BaseException:
            self.kernel.CloseHandle(self.mutex)
            raise

    def _acquire(self):
        result = self.kernel.WaitForSingleObject(self.mutex, 0)
        if result == 0x80:  # Abandoned writer: invalidate possibly incomplete data.
            self.memory[:HEADER.size] = HEADER.pack(0, 0)
        if result in (0, 0x80):
            return True
        if result == 258:
            return False
        raise ctypes.WinError(ctypes.get_last_error())

    def publish(self, data):
        payload = json.dumps(data, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(payload) > CAPACITY-HEADER.size:
            return False
        if not self._acquire():
            return False
        try:
            sequence, _ = HEADER.unpack(self.memory[:HEADER.size])
            self.memory[HEADER.size:HEADER.size+len(payload)] = payload
            self.memory[:HEADER.size] = HEADER.pack(sequence+1, len(payload))
        finally:
            self.kernel.ReleaseMutex(self.mutex)
        return True

    def read(self, previous=None):
        if not self._acquire():
            return None
        try:
            sequence, size = HEADER.unpack(self.memory[:HEADER.size])
            if not size or size > CAPACITY-HEADER.size or sequence == previous:
                return None
            payload = self.memory[HEADER.size:HEADER.size+size]
        finally:
            self.kernel.ReleaseMutex(self.mutex)
        return sequence, payload

    def close(self):
        self.memory.close()
        self.kernel.CloseHandle(self.mutex)


_channels = {}
_channels_lock = threading.Lock()


def channel(root=ROOT):
    key = str(Path(root).resolve()).casefold()
    with _channels_lock:
        if key not in _channels:
            _channels[key] = TelemetryMemory(root)
        return _channels[key]


def publish(data, root=ROOT):
    try:
        return channel(root).publish(data)
    except (OSError, ValueError):
        return False


def read(root=ROOT, previous=None):
    try:
        return channel(root).read(previous)
    except OSError:
        return None
