# Live rewards v2

The live PPO collector uses `live_rewards.py`, independently of the legacy mock
environment's normalized damage rewards. Old checkpoints cannot resume this reward
version. The next training run creates a fresh model when no active run exists.

Requested weights: damage +1.5 per 1000 HP; kill +0.1; stage clear +10;
life lost -8; displayed power gained +1 per 1.00 and lost -6 per 1.00.
TH10 raw power is divided by 20. There is no separate bomb penalty or remaining
life bonus. Bombs and deaths can incur the measured power decrease penalty.

Connected: life decrease, net power change per action, normal stage N -> N+1.
Continue/start/recovery initialization is excluded from collection. Stage six
ending clear is not yet detected. Within-action simultaneous gains/losses are netted.

Superseded by `th10-rewards-v3`: damage and kill acquisition is now connected to
signature-checked native combat events, retaining these formula weights. See
`combat-rewards.md` for actual-game evidence and remaining boss/bomb validation.

OBS2 cumulative bars use abs(component sum), clamped to 30 for width; the numeric
value remains signed and unclamped. All components share the same 30-second window.
