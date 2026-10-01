# Single-pass collection and timing (2026-09-27)

TimedBombPolicy.forward_with_distribution returns actions, values, log probabilities,
and the exact distributions from one feature extraction. forward retains its original
three-result API, so existing PPO updates and saved checkpoints remain compatible.
The live timed-bomb collector no longer reruns get_distribution for display probabilities.
The legacy non-timed policy path is unchanged.

Each episode JSONL row contains timing_ms for inference, step/observation/transition,
encoding and telemetry construction. Runtime timings separately report advance_wait_ms,
snapshot_ms and guard_snapshot_ms. These are nested components, not additive to the
step/observation/transition total. previous_step_timing_ms contains the preceding
step's completed measurements, including log serialization/write/flush, publishing,
and total_ms, with its one-based step number. last-step-timing.json preserves the final
completed step when the collection loop exits normally. Timing values are wall-clock
milliseconds; they do not enter policy observations, rewards or PPO settings.

Validation: 118 tests passed (one skipped) before runtime timing additions; subsequent
14 live tests and 4 timed-bomb tests passed, including seeded action/value/log-probability
and probability equality for bomb-decision and nondecision observations, one latent
extraction, and persisted timing records.

Model-only alternating comparison using a saved real model and identical observation,
100 measured iterations per path: double-call median 7.571 ms, single-call 4.257 ms;
p95 9.264 versus 5.336 ms. This is not a controlled end-to-end FPS comparison.

Deployment waited for game-over optimization, preserved update 32, and resumed from
real-episode-32.zip in live-learning-20260927-164802-28ddbe.
Initial live sample (430 records): inference median 3.96 ms, encoding 1.91 ms,
step/observation/transition 25.69 ms, log 0.35 ms, publishing 1.88 ms.
Decision timestamp interval median 34.57 ms, p95 47.27 ms; mean game advance 50.6 FPS.
Scenes and load differ from earlier samples: stable 60 FPS is not established.
