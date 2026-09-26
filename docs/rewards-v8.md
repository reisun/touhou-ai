# Live rewards v8

Add an ineffective-bomb selection penalty of -7 per game second of continuous
selection, or -7/30 per two-frame policy action. This is NOT an infinite-horizon
discounted total of -7. Existing low-power holding, direct power-loss and all
other reward settings are unchanged; gamma remains 0.997.

Use the policy's chosen bomb action. Require power below 1.00 and bomb state zero
both before and after the one/two-frame gameplay step. Do not charge when the
action does not select bomb, a bomb is active, power reaches the usable threshold,
the game enters terminal input release, or time belongs to dialogue/menu/stage
bridging. Repeated snapshots/events cannot duplicate the penalty.

OBS2 reports this component separately within the same 30-second window. Start
a fresh continuous model as requested; preserve previous models and recordings.
