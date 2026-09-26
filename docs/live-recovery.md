# Unexpected-state recovery

During collection, process/read failures and unsupported or stalled screens trigger
bounded recovery (at most two per run). Stage transitions continue the same episode.
Manual pause, F8 emergency stop, STOP files and run deadlines do not trigger recovery.
Configuration/attachment failures and optimizer/checkpoint errors stop the run.

The controller detaches input and closes only its managed game. If graceful shutdown
fails, recovery may terminate the process whose start time and executable were
verified. Steam and unmanaged games are never terminated.

At least 32 fully validated transitions are required for a recovery update. The
failing action is excluded. Nonterminal prefixes bootstrap from the last valid
observation; a fault is not counted as a death. Shorter prefixes are discarded,
leaving the previous policy unchanged. Checkpoints are saved, reloaded and verified
before restarting Normal/Reimu B at stage one. Optimizer failures do not overwrite
previous verified checkpoints or trigger another run.

Recovery consumes a trial slot within the existing 1-5 trial/time limits. If the
last slot recovers, the new stage-one game is paused for the user rather than left
running without control. Startup failure is reported and requires attention.

Validation: fault-injected collector tests cover prefix update/save/reload/restart
and insufficient-data fallback. These tests do not crash or control the real game;
real-process crash/hang recovery remains to be observed in a live run.
