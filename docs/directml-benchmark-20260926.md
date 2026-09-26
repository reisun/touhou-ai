# DirectML offline PPO validation — 2026-09-26

## Scope

The live CPU learner remains running and its model/environment are unchanged.
The isolated `.venv-directml` environment uses torch-directml 0.2.5.dev240914,
PyTorch 2.4.1, and Stable Baselines3 2.7.0 on Radeon RX 5700 XT.
The production environment uses PyTorch 2.8.0+cpu. `pip check` passed in the
isolated environment. The package's `+cpu` version string does not mean the
DirectML extension is unused: the benchmark explicitly selects PrivateUse1/DML.

The actual dual-grid policy has 934,728 parameters. Inputs retain the 192px
local window at 2px cells and the whole-field 8px grid. Tests use the weights
and optimizer from `live-learning-20260926-201451-940156/real-episode-14.zip`.
The workload uses all 1,743 recorded scenes from
`live-learning-20260926-204340-262e15/episode-1.jsonl`.

Actions and advantages are synthetic and fixed across backends; values and old
log probabilities are generated once on the CPU. These are computation and
stability tests, not a reconstruction of the original on-policy rollout, and
not evidence of improved game performance. All runs use batch size 64 and
10 epochs (280 minibatch updates per measured round). Timing includes tensor
transfers, PPO diagnostics and a final synchronization/parameter transfer;
excludes initial loading, observation encoding, save/reload and compilation warmup.
Initialization includes a forward pass, but the first backward can still be cold.

The live learner and game remained active during measurements. CPU benchmarks
therefore include variable contention from live inference/optimization. CPU and
GPU test workloads were sequential except a brief initial failed GPU smoke test
near the end of the first CPU benchmark. These are practical shared-machine
measurements, not isolated hardware speed ratios.

## Compatibility findings

Unmodified SB3 PPO fails on the first backward with:
`DirectML scatter doesn't allow partially modified dimensions`.
The categorical log-probability gather is replaced, only inside the benchmark
process, by a mathematically equivalent mask/multiply/sum over the action axis.
Its output and gradient are checked against the original on CPU. The mask must
explicitly use the logits dtype to avoid a DirectML autograd assertion.

Moving the policy to DirectML can replace Parameter objects. Rebinding the
optimizer to the transferred parameters and restoring optimizer state is
necessary; otherwise a loop can execute without changing model weights.
The benchmark checks parameter binding and actual weight changes.

`aten::std.correction` falls back to the CPU during advantage normalization.
Thus this is mixed GPU/CPU execution, not a fully GPU-resident update.

## Results

| Backend | Rounds | Seconds per round | Median |
|---|---:|---|---:|
| Production PyTorch 2.8, CPU 8 threads | 3 | 35.84, 35.06, 52.26 | 35.84 |
| DirectML environment PyTorch 2.4.1, CPU 8 threads | 3 | 32.30, 31.42, 50.79 | 32.30 |
| DirectML with compatibility fixes | 3 | 19.96, 23.51, 23.56 | 23.51 |
| DirectML with fixes, 10-round stability run | 10 | range 18.98–30.64 | 20.48 |

All three successful configurations passed finite-parameter/metric checks,
portable checkpoint save, exact parameter reload on CPU, and another epoch
after reload. GPU-to-CPU value output max difference was 1.91e-6 in the initial
DirectML test. A longer GPU run and scene-diverse gradient check are recorded
separately in the JSON artifacts.

The 10-round run passed all rounds (2,800 minibatch updates), finite gradients,
finite weights/metrics, and actual parameter changes on every round. Portable
save/reload and a further GPU epoch after reload passed. Maximum value output
difference after CPU reload was 1.91e-6. A separate test selected eight scenes
evenly across the episode; CPU/GPU output checks and gradient checks passed,
with maximum gradient absolute difference 9.54e-7. This test also passed save,
CPU reload, and one GPU epoch after reload. Gradient checks used rtol 0.02 and
atol 0.0002; output checks used rtol 0.001 and atol 0.0001.

The GPU stability-run median is approximately 43% below the current CPU
environment median and 37% below the same-PyTorch CPU median. These percentages
are descriptive of these shared-machine samples, not a guaranteed speedup.
Ten rounds represent roughly four minutes of optimization, not an overnight
stability test. Model loading/device transfer and checkpoint overhead remain
outside the measured update interval.

## Adoption recommendation

Performance and short-run numerical stability justify a limited pilot of
GPU-only optimization with CPU inference. Do not switch by merely changing
`device`: the unmodified PPO implementation fails. A production integration
needs the tested categorical compatibility path, optimizer rebinding/state
transfer, and checkpoint portability. On GPU failure, reload the pre-update
CPU checkpoint/optimizer before retrying the complete update on CPU; never
continue from a partially modified failed GPU update.

Start with a bounded set of real episodes and compare complete game-over to
resume time, errors, and rollback behavior. Require a sustained run before
claiming long-term stability. This report does not authorize or perform the
production switch; the active learner remains on CPU with 8 update threads.

The earlier live CPU 1-thread estimate (47–69 seconds, average approximately
56 seconds) is the end-of-trace to saved-checkpoint interval for six different
episodes. It includes other work and differs in sample count, so it is not an
apples-to-apples baseline for these fixed-workload measurements.

## Reproduction and outputs

Run `scripts/benchmark_directml.py` with `--device prepare` in the production
environment to generate `artifacts/directml-benchmark/workload.npz`; then use
`--device cpu` in each environment and `--device dml-compatible` in the isolated
environment. All commands require `--output artifacts/directml-benchmark` and
`--checkpoint artifacts/live-learning-20260926-201451-940156/real-episode-14.zip`.
Preparation also requires `--trace` with the recorded episode path above.
`--rounds 10 --label=-stress` performs the longer run. The script never attaches
to the game, never edits live checkpoints, and exits nonzero on test failure.

JSON results include measured timings, optimizer metrics, exceptions, and
save/reload status. Compatibility changes are restricted to the benchmark;
no production GPU switch has been made.
