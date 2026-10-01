# 通常報酬 v18：進行度のPower係数を0.1へ

2026-09-30、ユーザー指定により進行度報酬のPower係数だけを2/60（約0.0333）から0.1へ変更。

進行度報酬は `20/60 + 0.5 × 控え残機 + 0.1 × Power`。Powerは従来どおりゲーム生値/20、範囲0～5。Powerによる加算は最大1/6から0.5に増える。ボム中の抑制、イベント検証、重複排除は維持。被弾−1、ショット0.25×(1+0.5×Power)/1000HP、Power減少0も維持。

通常の旧報酬チェックポイント再開拒否は維持。今回の変更専用の `-UpgradeProgressPower` / `--upgrade-progress-power` を指定した場合だけ、完全に一致するv17の報酬設定から継続可能。モデル・最適化状態・乱数状態はそのまま引き継ぎ、保存済みの旧報酬履歴は改変しない。新runのstatusにreward_transitionとして旧・新設定を記録する。過去の総報酬との比較では変更点を区別する。

報酬計算・変更元検証・学習の関連15テスト成功。

実機反映：215更新時の `live-learning-20260930-125746-b6999d/real-episode-4.zip` を保存し、`live-learning-20260930-131411-785dbd` へ継続。playing、reward_version=v18、progress_power=0.1、resume_rng_restored=true、error=nullを確認。
