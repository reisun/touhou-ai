"""Inspect PE32 code without injection; packed builds can use read-only process bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from touhou_ai.live_inspect import SUPPORTED_HASH
from touhou_ai.windows_probe import pe_image_base


def image_bytes(address, size):
    config = json.loads(Path('game.local.json').read_text(encoding='utf-8'))
    data = Path(config['executable']).read_bytes()
    if hashlib.sha256(data).hexdigest().upper() != SUPPORTED_HASH:
        raise ValueError('unsupported image')
    base = pe_image_base(data)
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    count = struct.unpack_from('<H', data, pe+6)[0]
    optional_size = struct.unpack_from('<H', data, pe+20)[0]
    for index in range(count):
        offset = pe+24+optional_size+index*40
        _, rva, raw_size, raw_offset = struct.unpack_from('<4I', data, offset+8)
        relative = address-base-rva
        if 0 <= relative and relative+size <= raw_size:
            return data[raw_offset+relative:raw_offset+relative+size]
    raise ValueError('address outside mapped raw section')


if __name__ == '__main__':
    from capstone import Cs, CS_ARCH_X86, CS_MODE_32
    parser = argparse.ArgumentParser()
    parser.add_argument('address', type=lambda s: int(s, 0))
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--read-only-pid', type=int)
    args = parser.parse_args()
    if not 1 <= args.count <= 500:
        raise ValueError('count out of bounds')
    if args.read_only_pid:
        from touhou_ai.live_inspect import verify_game
        from touhou_ai.windows_probe import ReadOnlyProcess
        config = verify_game(args.read_only_pid)
        process = ReadOnlyProcess(args.read_only_pid, config['executable'])
        try:
            data = process.read(args.address, args.count*15)
        finally:
            process.close()
    else:
        data = image_bytes(args.address, args.count*15)
    for instruction in Cs(CS_ARCH_X86, CS_MODE_32).disasm(data, args.address, args.count):
        print(f'{instruction.address:#x}: {instruction.mnemonic} {instruction.op_str}')
