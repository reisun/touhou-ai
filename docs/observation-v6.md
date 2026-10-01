# Observation v6

2026-09-27: `th10-dual-grid-v6`, rewards remain `th10-rewards-v16`.

Local grid remains 6 x 96 x 96. Global grid is now 12 x 56 x 48:

0. Bullet density
1. Bullet density at observed velocity offset +2F
2. Bullet density at observed velocity offset +4F
3. Enemy density, including bosses
4. Mean known current enemy HP / 100000
5. Mean known maximum enemy HP / 100000
6. Ordinary item density
7. Player location
8. Laser coverage
9. Field-validated laser coverage
10. Player shot coverage
11. Power item amount

HP means use only enemies with valid current and maximum HP in the cell.
Unknown HP does not contribute to the numerator or denominator. No known HP
means both channels are zero. Both HP channels are clipped to [0, 1] after
averaging and retain the existing float16 round trip. Current HP is an amount,
not an average of individual HP ratios. Bosses participate in these means.

Removed the known-HP fraction and separate boss density channels. The existing
player-state boss-presence flag and spell state remain unchanged. OBS retains
the spatial display; numeric HP layers are omitted as before.

Old observation checkpoints must not resume into v6. Start a fresh model;
retain previous runs and checkpoints. Learning improvements are not yet measured.
