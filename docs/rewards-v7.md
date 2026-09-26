# Live rewards v7

Restore the direct power-loss penalty to -6 per displayed 1.00 lost. Keep gamma
0.997, power gain +1 per 1.00, damage +3 per 1000 HP, kill +0.1, stage clear +10,
hit -15, and low-power holding -0.1575 * max(0, 4 - power) per game second.

Power 5 -> 4 now incurs the direct -6 penalty, but no low-power holding penalty.
Below power four, both applicable penalties are enabled. OBS2 shows power gain
and loss together, with low-power holding on a separate row.

Start a fresh continuous campaign without resuming v6 weights or optimizer state.
Previous models and recordings remain preserved and are excluded from the v7
growth history. Existing continuous recovery and safety-stop behavior is unchanged.
