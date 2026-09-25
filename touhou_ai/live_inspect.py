"""Read-only instruction inspection for the pinned game during adapter development."""
import argparse
import hashlib
import json
from pathlib import Path

from touhou_ai.windows_probe import ReadOnlyProcess, pe_image_base

SUPPORTED_HASH = "3BDB72CF3D7C33C183359D368C801490DBCF54E6B3B2F060B95D72250B6866A3"


def verify_game(pid, config_path=Path("game.local.json")):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    path = Path(config["executable"]).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest().upper()
    if digest != SUPPORTED_HASH or digest != config["sha256"].upper() or pe_image_base(data) != 0x400000:
        raise ValueError("unsupported game executable")
    process = ReadOnlyProcess(pid, path)
    try:
        if process.read(0x400000, 64) != data[:64]:
            raise ValueError("loaded PE does not match")
    finally:
        process.close()
    return config


if __name__ == "__main__":
    import frida
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--address", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--xrefs", action="store_true")
    args = parser.parse_args()
    verify_game(args.pid)
    if not 0x401000 <= args.address < 0x500000 or not 1 <= args.count <= 250:
        raise ValueError("inspection bounds")
    session = frida.attach(args.pid)
    try:
        script = session.create_script("""
rpc.exports = {xrefs(address) {
    const bytes = [0,8,16,24].map(s => ((address >>> s) & 255).toString(16).padStart(2, '0')).join(' ');
    return Memory.scanSync(ptr(0x401000), 0x6f000, bytes).map(m => m.address.toString());
}, inspect(address, count) {
    let p = ptr(address), result = [];
    for (let i = 0; i < count; i++) {
        const ins = Instruction.parse(p);
        const hex = Array.from(new Uint8Array(p.readByteArray(ins.size))).map(b => b.toString(16).padStart(2,'0')).join('');
        result.push(p.toString() + ': ' + ins.toString() + ' // ' + hex);
        p = ins.next;
    }
    return result;
}};
""")
        script.load()
        print("\n".join(script.exports_sync.xrefs(args.address) if args.xrefs else
                        script.exports_sync.inspect(args.address, args.count)))
    finally:
        session.detach()
