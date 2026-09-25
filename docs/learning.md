# Learning and evaluation operations

## Implemented boundary

The runner currently accepts only backend=mock and reward_profile=mock_zero_test_only.
Rewards are always zero: metrics demonstrate plumbing, not learned game skill.
The source-based initial gameplay observation/action/reward design is implemented
separately; see [the Sharu-inspired profile](sharu-learning-profile.md).
Live collection and experiment budgets are still required before game learning.

The host GPU is AMD Radeon RX 5700 XT. This milestone uses CPU PyTorch only;
no GPU runtime or driver changes were made. Windows runs a standard-library
bridge; dependencies and training execution stay in Docker, also callable via WSL.

## Verification

From D:\repos\touhou-ai in PowerShell:

```powershell
./scripts/verify.ps1
```

This runs host/container protocol tests, learner tests, Gymnasium/SB3 checking,
a 64-step diagnostic PPO optimization, model reload, and two evaluation episodes.
The learner tests exercise STOP checkpoints, resume, SIGTERM handling, invalid
configuration rejection, and refusal to overwrite a run. The bridge is stopped
after verification only if verification itself started it.

## Background operation

`scripts/learning.ps1 start` starts the Compose training service. It requires
the mock bridge running and uses configs/mock-smoke.json by default. A unique
run directory is recorded in .runtime/training.json. Use `-ConfigFile name.json`
to select another reviewed mock configuration from configs.

`status` shows both container status and persisted run status; `logs` shows the
latest container log; `report` summarizes all persisted runs. `evaluate` reloads
the most recent saved model and writes a separate evaluation JSON.

Only one client may own the bridge episode. Do not concurrently run training,
smoke tests, contract checks, or evaluation against the same bridge. The PowerShell
wrappers reject competing checks while the managed training service is running.

`stop` sends Docker SIGTERM with a 30-second grace period. The runner handles
the signal, saves model.zip, marks status=stopped, and closes the environment.
Creating a STOP file in a run directory requests the same cooperative stop.
An abrupt host/power failure may leave status=running; inspect container state
and resume from the latest checkpoint in a NEW directory. A partially written
checkpoint is not guaranteed recoverable after power loss.

## Resume from a trusted local model

With the bridge running, substitute the actual run names below:

```powershell
docker compose --profile learning run --rm learner python -m touhou_ai.runner train --config /configs/mock-smoke.json --output /artifacts/resumed-run --resume /artifacts/prior-run/model.zip
```

Resume adds the configured number of timesteps to the saved policy. Optimizer
state is loaded; environment episode state and partially collected rollouts are
not restored. Core configuration compatibility is checked against the source
run's config.json. Model files must come from trusted local runs.

## Artifacts

| File | Purpose |
| --- | --- |
| config.json | Exact experiment configuration |
| versions.json | Core library versions |
| status.json | Starting/running/completed/stopped/failed and times |
| model.zip | Final or cooperatively stopped PPO model |
| checkpoint-N.zip | Periodic snapshots; may precede the next optimizer update |
| progress.json | Last checkpoint timestep |
| progress.csv, log.txt | Training metrics and text log |
| episodes.monitor.csv | Completed episode rewards and lengths |
| evaluation.json | Deterministic evaluation after a completed run |

No tokens are written into these artifacts. .env, .runtime, and artifacts are
ignored by Git. The repository has no external remote or automatic publishing.
