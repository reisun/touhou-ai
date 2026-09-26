# Stage continuity

One live learning episode runs until game over, not until a stage clear. A stage
change must not trigger PPO, checkpoint reload, Continue, or a game restart.

The collector bridges dialogue and loading after each policy action. Confirmation
is sent only during observed dialogue; loading uses neutral input. The first
playable observation resumes policy decisions without an extra neutral step.
Supported transitions are between adjacent main-game stages under Normal/Reimu B.
Pause, replay, unexpected modes, stage skips, and the unvalidated ending are not
automatically dismissed. A 1,800-frame bound, the run deadline, and STOP remain in
effect. Interrupted partial trajectories are not optimized.

The observation after the transition stays in the same rollout. Stage counters
may reset, but episode-start and terminal flags do not reset. Hit/bomb event IDs
include the stage so equal frame numbers cannot suppress later rewards. Stage
clear itself currently has no reward. Automatic dialogue/loading frames are
recorded separately from policy decisions; PPO discounts by decision, not by each
intervening game frame.

Validation: simulated dialogue/loading/adjacent-stage transitions, no extra input
on ready frames, game over, user pause, replay, wrong mode, cancellation, timeout,
and reward identity. A new live stage-crossing acceptance run has not been made
for this change. No survival guarantee is implied for subsequent AI gameplay.
