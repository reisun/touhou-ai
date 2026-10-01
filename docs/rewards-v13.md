# 報酬v13・新規学習（2026-09-27）

ユーザー指定により被弾を−60/回、ショットダメージを10 × 有効ダメージ/1000 × (1 + 0.5 × Power)へ変更。ボム中のダメージ報酬0、進行度、PPO、観測等は維持。

旧v12 run live-learning-20260927-034112-084a28 は33更新・58,761 stepsで通常停止。成果物は保持。
新run live-learning-20260927-043533-e1d462 をresumeなし、dual-grid / DirectML / continuous / MaxSteps 18000 / NoUIで開始。
報酬v13、resumed_from=null、0更新からの開始とOBS APIのhit=-60、damage=10を確認。表示サーバーは学習開始手順で自動再起動。

関連16テスト成功、PowerShell構文検査成功。初回1,038 steps、被弾3、報酬−154.7965、DirectML更新16.734秒。パラメータ変更、real-episode-1.zip保存・再読込成功。次のプレイへ継続中。
改善効果は未評価。
