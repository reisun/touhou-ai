# Live display transport and observation reuse

Live packets now pass from the learner / passive observer / runtime diagnostic to
Windows named shared memory (8 MiB per artifacts root), then through the existing
localhost /api/live and /api/live-stream interfaces to the browser. OBS only renders
the web page. There is no latest.json filesystem read/write or disk fallback in these
paths. A stale legacy latest.json is ignored.

The channel uses a pagefile-backed named mapping and a named mutex scoped to the
Windows session and normalized artifacts root. The mutex protects only the payload
copy; JSON serialization and socket transmission happen outside it. Writers and
readers attempt the lock with zero timeout. Contention skips the display update
instead of delaying gameplay. Oversized payloads are rejected without destroying
the preceding complete packet. An abandoned mutex invalidates potentially partial
content. The latest packet survives publisher or reader reconnect while another
process retains the mapping; it is lost when all handles close. This is display
transport, not durable recording. Publisher False increments existing telemetry_drops
in the live collector.

Episode JSONL, status manifests and model checkpoints remain durable. /api/obs still
reads those records to compute lossless 30-second reward summaries and historical
growth: this change removes the per-step latest-frame file exchange, not all disk I/O.
The web static assets and browser APIs remain unchanged.

Normal-play bridge_transition reuses a complete observation only when the stage has
not changed, the game is playable and nonterminal, and LiveRuntime confirms the same
enabled, parked gate tick. Partial snapshots, dialogue/loading, stage changes and
terminal states retain the final full refresh. The pre-action guard is unchanged.

Validation: 123 tests passed, 1 skipped. Includes cross-process memory exchange,
reader reconnect, oversized payload preservation, no waiting on contention, HTTP/SSE
with file reads forbidden, and observation reuse / stale gate / transition checks.

## Live verification

Resumed preserved update 42 from live-learning-20260927-164802-28ddbe/real-episode-10.zip
into live-learning-20260927-170506-7f42ec. Latest HTTP packet timestamps advanced during
15 seconds while legacy latest.json mtime remained unchanged. telemetry_drops was 0.
624 normal-play timing samples: publish median 0.709 ms; inference 3.623 ms;
full snapshot 6.777 ms; pre-action guard snapshot 2.537 ms; frame advance wait
15.414 ms; encode 2.222 ms; total median 33.347 ms and p95 35.04 ms.
Frame advance wait can increase as the game regains its native pacing, so it must
not be interpreted as all avoidable compute. These are live observations, not a
controlled same-scene before/after comparison or a guarantee of stable 60 FPS.
