"""Read-only connectivity probe for the explicitly configured Touhou executable.

Address facts: projMiss/Th10Ai, Th10Hook/src/Th10Hook/Th10Apis.cpp.
This independently implemented probe does not inject code or write memory.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import struct
import time


def pe_image_base(data):
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[offset:offset + 4] != b"PE\0\0":
        raise ValueError("invalid PE signature")
    if struct.unpack_from("<H", data, offset + 24)[0] != 0x10B:
        raise ValueError("expected PE32 game")
    return struct.unpack_from("<I", data, offset + 24 + 28)[0]


class ReadOnlyProcess:
    max_read_size = 262144
    def __init__(self, pid, expected_path):
        if os.name != "nt":
            raise OSError("Windows is required")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.api.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                       wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.api.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.api.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.api.ReadProcessMemory.restype = wintypes.BOOL
        self.handle = self.api.OpenProcess(0x1000 | 0x0010, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            path = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(path))
            if not self.api.QueryFullProcessImageNameW(self.handle, 0, path, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            if os.path.normcase(path.value) != os.path.normcase(str(expected_path)):
                raise ValueError("PID does not match configured game executable")
        except BaseException:
            self.close()
            raise

    def read(self, address, size):
        if not 0 < size <= self.max_read_size:
            raise ValueError("read size outside probe bounds")
        buffer = ctypes.create_string_buffer(size)
        received = ctypes.c_size_t()
        if not self.api.ReadProcessMemory(self.handle, address, buffer, size, ctypes.byref(received)):
            raise ctypes.WinError(ctypes.get_last_error())
        if received.value != size:
            raise OSError("short process memory read")
        return buffer.raw

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def probe(pid, config, samples=5):
    path = Path(config["executable"]).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest().upper()
    if digest != config["sha256"].upper():
        raise ValueError("game hash changed; refusing address-based probe")
    base = pe_image_base(data)
    if base != 0x400000:
        raise ValueError("unsupported image base")
    process = ReadOnlyProcess(pid, path)
    try:
        if process.read(base, 64) != data[:64]:
            raise ValueError("loaded image header does not match configured executable")
        observations = []
        # Read a small contiguous globals block; values are explicitly experimental.
        for _ in range(samples):
            block = process.read(0x474C48, 0x44)
            observations.append({"monotonic_ns": time.monotonic_ns(),
                                 "power_raw": struct.unpack_from("<i", block, 0)[0],
                                 "character_raw": struct.unpack_from("<i", block, 0x20)[0],
                                 "lives_raw": struct.unpack_from("<i", block, 0x28)[0],
                                 "stage_frame_raw": struct.unpack_from("<i", block, 0x40)[0]})
            time.sleep(0.05)
        return {"backend": "th10-read-only-probe", "pid": pid, "sha256": digest,
                "process_memory_read": True, "gameplay_offsets_validated": False,
                "frame_synchronization_validated": False, "samples": observations}
    finally:
        process.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("game.local.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.pid, json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
