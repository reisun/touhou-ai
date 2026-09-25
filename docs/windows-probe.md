# Windows game connectivity probe

```powershell
./scripts/game.ps1 start
./scripts/probe.ps1
./scripts/game.ps1 stop
```

The probe requires the managed game record, checks its creation time, verifies
the configured executable SHA256, opens the process with limited query plus
VM_READ rights, verifies its executable path and PE header, and reads a small
globals region five times. Results are saved in artifacts/probes.

There are no memory writes, DLL injection, input events, or suspend/resume calls.
The values are raw diagnostics. A successful read does NOT prove that the
addresses correspond to gameplay fields in this Steam build. In particular,
title-screen zero values are not gameplay validation. The result explicitly
sets gameplay_offsets_validated and frame_synchronization_validated to false.

Address facts were researched from the Th10Ai project's Th10Apis.cpp:
https://github.com/projMiss/Th10Ai/blob/master/Th10Hook/src/Th10Hook/Th10Apis.cpp
No external source is built or loaded. This probe is independently implemented.
Reference Win32 API:
https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory

## Remaining game-adapter work

1. Verify player/bullet/laser/state fields against actual gameplay and this build.
2. Establish frame advancement control and measure observation/action alignment.
3. Implement bounded input with release on focus loss, timeout, and shutdown.
4. Verify death/game-over detection and repeatable episode restart.
5. Define the approved observation/action/reward contract, then add a real Gym env.

Until those checks pass, the learner refuses a real-game backend. No claim of
autonomous Touhou gameplay is made by the successful environment diagnostics.
