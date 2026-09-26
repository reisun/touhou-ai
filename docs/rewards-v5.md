# Live rewards v5

Update v6: direct power-loss penalty is disabled (`power_down = 0`). Pickup
reward (+1 per displayed power) and low-power holding penalty are retained.
Power 5 -> 4 has no power-loss or holding penalty. Below four, holding penalty
accrues on subsequent gameplay steps. Old checkpoints remain preserved.

Gamma is 0.997 per policy step (two game frames, 30 decisions per game second).
This is not a weight of 0.997 at ten seconds: that weight is 0.997^300 = 0.406.

Low-power holding reward per game second:

    -0.1575 * max(0, 4 - displayed_power)

At constant power zero each two-frame step receives -0.021. The infinite
discounted sum, including the next transition reward with weight one, is
-0.021 / (1 - 0.997) = -7. This is the low-power component only, not the
model's predicted value or the undiscounted episode return.

Use power at the start of the action. Accrue only one/two-frame gameplay
transitions; exclude menus, dialogue, stage transitions and wall-clock training
time. Power four or five has no holding penalty and no holding bonus.
Keep the existing power pickup/loss and damage/kill/hit/stage rewards unchanged.
OBS2 displays low-power holding separately in the shared 30-second window.

Previous models and logs remain preserved. Automatic resume rejects old reward
versions/configuration; migration or a fresh run requires an explicit decision.
No real-game learning was started by this configuration change.
