// Diagnostic-only evidence from native successful shot overlap checks.
signature(0x4287a3, '8b7e14');
let shotContacts = [];
hooks.push(Interceptor.attach(ptr(0x4287a3), function () {
    if (!enabled || !gameplayGuard || shotContacts.length >= 1000) return;
    const shot = this.context.esi.sub(0x44), descriptor = shot.add(0x58).readPointer();
    const target = this.context.edi;
    shotContacts.push({frame:ptr(0x474c88).readS32(),
        position:[shot.add(0x14).readFloat(),shot.add(0x18).readFloat()],
        hitbox:[descriptor.add(0xc).readFloat(),descriptor.add(0x10).readFloat()],
        target:[target.readFloat(),target.add(4).readFloat()]});
}));
rpc.exports.shotcontacts = function () { return shotContacts; };
