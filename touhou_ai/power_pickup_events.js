// Read actual Power gains at the two verified item-pickup call sites.
signature(0x418930, '558b6c2408568bf0');
signature(0x41b3db, 'e850d5ffff');
signature(0x41b635, 'e8f6d2ffff');
hooks.push(Interceptor.attach(ptr(0x418930), {
    onEnter(args) {
        this.trackPower = false;
        if (!enabled || !gameplayGuard) return;
        const caller = this.returnAddress.toUInt32();
        if (caller !== 0x41b3e0 && caller !== 0x41b63a) return;
        try {
            const type = this.context.ebp.add(0x30).readS32();
            const nominal = ({1:1, 10:1, 4:20, 11:20})[type];
            if (!this.context.eax.equals(ptr(0x474c40)) || nominal !== args[0].toInt32()
                    || (caller === 0x41b3e0) !== (nominal === 1))
                throw new Error('unexpected Power pickup call');
            this.powerBefore = ptr(0x474c48).readS16();
            if (this.powerBefore < 0 || this.powerBefore > 100)
                throw new Error('invalid Power before pickup');
            this.powerNominal = nominal;
            this.powerType = type;
            this.trackPower = true;
        } catch (error) { combatError = String(error); }
    },
    onLeave() {
        if (!this.trackPower) return;
        try {
            const after = ptr(0x474c48).readS16();
            const amount = after-this.powerBefore;
            if (amount !== Math.min(this.powerNominal, 100-this.powerBefore))
                throw new Error('unexpected actual Power gain');
            if (amount === 0) return;
            if (combatEvents.length >= 4096) throw new Error('Power event overflow');
            combatEvents.push({id: combatSession+':'+(++combatSequence), kind:'power_gain',
                confirmed:true, source:'verified_power_pickup_v1', amount_raw:amount,
                before_raw:this.powerBefore, after_raw:after, item_type:this.powerType,
                stage:ptr(0x474c7c).readS32(), frame:ptr(0x474c88).readS32(), gate_tick:tick});
        } catch (error) { combatError = String(error); }
    }
}));
