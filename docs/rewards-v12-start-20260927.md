# 報酬v12の新規学習開始（2026-09-27）

被弾罰−30への修正をサブエージェントに委任。97テスト成功・1件skip、PowerShell構文検査成功。
他の報酬係数、PPO、dual-grid観測、ボム周期、DirectML更新は維持。

- 旧run: live-learning-20260927-012502-a6141c。通常STOPで停止。90更新・151,126 stepsの成果物を保持。
- 新run: live-learning-20260927-034112-084a28。
- 起動: scripts/live-learning.ps1 rehearse -DualGrid -DirectML -Continuous -MaxSteps 18000 -NoUI。
- resumed_from=null、開始時0更新・0 steps、th10-rewards-v12 / hit=-30 を確認。
- 初期ゲーム状態: stage 1 / frame 2 / lives_raw 2 / power_raw 0。
- 初回: 1,038 steps、被弾3、総報酬−77.39825、ゲームオーバーでDirectML更新。
- パラメータ変更・real-episode-1.zip保存・再読込検証成功。更新時間17.125秒。
- 初回保存後、次のプレイへ継続していることを確認。回避能力改善の評価はまだ行っていない。

学習は連続実行中。停止は scripts/live-learning.ps1 stop。
