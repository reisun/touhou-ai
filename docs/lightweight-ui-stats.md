# Lightweight UI statistics (2026-09-27)

Normal learning no longer writes episode-N.jsonl full-observation diagnostic logs.
The PPO rollout buffer, observation encoding, action sampling, rewards and optimizer
are unchanged. Checkpoints, status manifests and per-episode summaries remain durable.

UiStats consumes every collected reward/value/terminal event into an in-memory 31-second
window. About once per second and at episode end, it publishes the complete window
through a separate Windows named shared-memory channel. Replacing a display packet
cannot lose rewards: each subsequent packet repeats the complete recent window.
Readers calculate the same wall-clock 30-second totals and TD errors, with no bootstrap
across episode boundaries. Old data expires even if learning is stopped. A dashboard
restart can reattach while the learner retains the channel. Memory-only recent history
is lost if every owner exits; no historical values are fabricated in its place.

Maximum confirmed progression is tracked during collection and stored as max_progress
in the verified episode entry in status.json. New growth entries never require full
observation logs. Old entries remain compatible through the existing cached legacy
reader. Status manifests are still read for history; full observation logs are not read
for new-run rolling reward windows. Old diagnostic files are retained, not deleted.

To investigate a future issue, explicitly enable scripts/live-learning.ps1 -DetailedLogs
or python -m touhou_ai.live_learning --detailed-logs. This restores full per-step JSONL
for that run only. Default is off. Existing lightweight console progress (every 120
steps), initial episode state, final timing summary and checkpoint files remain.

Validation covers exact totals and TD semantics, expiry, repeated reads, persisted
progression, rejection of unconfirmed milestones, opt-in detailed logs and successful
optimizer/save/reload in default mode without any episode JSONL files.

## Live verification

Resumed update 49 into live-learning-20260927-171751-ed391e with detailed_logs=false
and ui_stats_transport=shared_memory_v1. No episode JSONL files were created. The run
folder grew by zero bytes during a measured 10-second normal-play interval. Shared
memory contained 920 compact rows in 251,556 bytes; these replace a rolling window,
not an accumulating file. /api/obs showed live reward values and all 49 prior episodes.
Legacy history initial reconstruction took 51.69 seconds after dashboard restart;
subsequent requests took 47-78 ms. That initial full-log scan applies to old entries
without persisted max_progress, not newly collected episodes. Existing legacy logs
were preserved. Checkpoint writes and per-episode summaries still occur at boundaries.

Existing full suite: 123 passed, one skipped. New UiStats tests: 2 passed.
Default-no-log recovery learning/save/reload tests: 4 passed.
