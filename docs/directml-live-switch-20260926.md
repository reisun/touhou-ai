# DirectML live update switch — 2026-09-26

User authorized the switch after offline validation. CPU inference remains in
the existing PyTorch 2.8 environment. At game over the complete real rollout
and pre-update model/optimizer are passed to a hidden, isolated DirectML worker
in `.venv-directml` (PyTorch 2.4.1, RX 5700 XT).

The worker uses the tested categorical masked-sum compatibility path and
rebinds optimizer parameters when changing devices. It returns a portable CPU
checkpoint. The parent checks finite/changed parameters and training counters;
the existing save/reload verification still checks actions, weights and step
count before the next game begins.

The worker has a 180-second timeout. If it fails or its output fails validation,
the untouched parent policy/optimizer and rollout are trained on CPU with
8 threads. Further GPU attempts are disabled for that run and the reason is
recorded in status. This protects against recoverable worker errors, not an
OS-level GPU driver failure or system crash. The large temporary rollout file
is removed after each attempt; worker logs and before/after checkpoints remain.

No AMD Software power, fan, clock, or voltage settings were changed. No
application-level thermal throttling/temperature monitoring is installed.
The already observed standard-deviation CPU fallback in DirectML is expected.

Validation before switching: 2 new compatibility/rollback tests and 21 existing
dual-grid/live tests passed. An actual isolated-worker integration run on the
1,743-scene benchmark completed its GPU update in 20.485 seconds and the whole
handoff/update/load in 27.219 seconds, without CPU fallback.

The prior CPU run stopped at the game-over boundary after update 34, preserving
55,062 decisions in `live-learning-20260926-204340-262e15/real-episode-19.zip`.
Checkpoint SHA256:
`96a9850b76792a6b9c18c2ca14b6ee780b7ff575840543f17e79b9350cd57b33`.

The successor was launched with `-ResumeLatest -DualGrid -DirectML -Continuous
-ContinueManaged -MaxSteps 18000 -NoUI`. Logs are
`.runtime/continuous-learning-directml.log` and
`.runtime/continuous-learning-directml.error.log`; the active run is recorded
in `.runtime/live-learning.json`.

First live acceptance succeeded in `live-learning-20260926-211554-9cb1dc`:
1,830 real decisions, GPU worker update 18.828 seconds, full update handoff
24.453 seconds, no CPU fallback. Parameters changed and checkpoint reload was
verified. Update count advanced to 35 and decisions to 56,892; the next play
started normally. First checkpoint SHA256:
`f9d882fc4c22f45a72489947601ce19b1a1491b1d0043a1f107d3153ff532232`.
