# Real rewards v9

Authorized 2026-09-26 during the focused-bullet architecture switch.
All v8 weights and eligibility rules remain unchanged.
Low-power maintenance now uses -0.1575 * (5 - P) per game second.
P is validated in [0, 5]; no max expression is needed.
P=0 costs -0.7875/second, P=4 costs -0.1575/second, P=5 costs zero.
At 30 decisions/second and gamma=0.997, perpetual P=0 has discounted
maintenance value -8.75 (previously -7). This is not a cap on real reward.
OBS2 shows the formula directly. Fixed reward-input normalization is unchanged.
Previous campaigns are retained; a new campaign starts from fresh weights.
