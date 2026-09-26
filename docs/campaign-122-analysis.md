# Old campaign: 122-update analysis

Campaign: live-learning-20260926-101422-e4a08b.
Architecture: th10-live-observed-v2, reward v8 (4-P deficit).
Analysis reads all 122 episode logs and optimizer summaries. Current training
was not modified. Detailed derived data: artifacts/old122-analysis.json.
Episode N uses the policy after update N-1; the saved update-122 policy has not
been separately evaluated here. These are evolving on-policy training samples,
not controlled evaluations of a fixed model or evidence of causality.

## First 20 versus last 22 episodes

| Metric | 1-20 | 101-122 |
| --- | ---: | ---: |
| Controlled game seconds/play | 50.57 | 78.13 |
| Seconds to first hit | 17.85 | 17.48 |
| Mean return | -156.34 | -87.34 |
| Damage/play | 5384 | 7866 |
| Damage/controlled second | 106.46 | 100.68 |
| Kills/play | 81.55 | 109.14 |
| Bomb activations/play | 5.05 | 2.73 |
| Mean Power (mean of episode means) | 0.649 | 1.055 |
| Time at Power 0 | 34.89% | 16.81% |
| Time at Power >=4 | 0% | 0.39% |
| Shot selected (mean episode fraction) | 51.34% | 70.37% |
| Invalid-bomb reward/play | -74.42 | -18.14 |

All 122 episodes lost three lives. No stage transition recorded. Longest
controlled play was 96.83 seconds (episode 120); menu/dialogue/bridge frames
are not counted in controlled time. Three name-entry recoveries were recorded.

## Findings

1. Survival duration increased but first-hit avoidance did not improve across
   endpoint windows. Last four first-hit times were 11.8, 11.97, 12.23 and,
   for episode 119 preceding these, 12.27 seconds. More time alive does not
   establish fine-grained bullet avoidance.
2. Invalid bomb penalties declined strongly; this accounts for most of the
   endpoint return improvement. It must not be mistaken for equivalent progress
   in dodging or offensive efficiency. Available/non-active bomb probability
   averaged 0.58% late; unavailable/non-active averaged 5.62%; active 68.20%.
3. High unavailable/non-active bomb probability is real: 68 decisions >=90%
   across the campaign, including 50 in the last 22 episodes. Thus the earlier
   active-bomb-only explanation was incomplete. Logs cannot establish whether
   position, bullet layout or other correlated features caused it.
4. Damage/second was flat rather than improving; shot was absent on about 30%
   of late sampled actions. Actual firing cadence is not directly inferred from
   this selection metric. High-power experience remained rare.
5. Returns are not a clean skill metric: episodes 81-100 averaged 53.22 seconds
   and -75.60 return; 101-122 lasted 78.13 seconds but returned -87.34. Late
   maintenance cost was -36.45/play and life-loss cost -45/play, against +23.60
   damage and +10.91 kills. Longer low-power survival accumulates more cost;
   this does not prove that the policy intentionally dies.
6. Optimizer logs were finite. Late mean explained variance was 0.890, but this
   measures fit to rollout return targets, not survival competence. Approximate
   KL rose from 0.0152 to 0.0289 and clip fraction from 0.169 to 0.293 between
   endpoint windows. Update volatility is worth watching, not a proven failure.

## Follow-up evaluation

Track first-hit time, damage/second, high-power exposure, shot selection, and
unavailable/non-active bomb probabilities separately from total return.
The focused architecture and reward-v9 change were applied together, so later
improvement cannot be attributed solely to removing the grid. This analysis
does not establish that the old grid caused the plateau. Fixed-checkpoint,
matched-condition evaluation would be needed for that claim.
