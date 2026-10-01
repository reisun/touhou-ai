# 小型CNNへの実機回避学習の切替（2026-09-29）

ユーザー指示により18択案は取り下げ。独立移動9+速度2を維持する。
シミュレーターで採用した小型CNNと操作/価値別勾配上限を、実機の回避専用学習へ接続した。

- 旧run: live-learning-20260928-124148-6c9ab8。
- 918更新・1,476,831判断でゲーム終了時の更新を保存して停止。
- real-episode-918.zipのSHA256を確認。旧run・設定・モデルは保持。
- 新run: live-learning-20260929-071910-9a4a78。旧モデルを読み込まずseed7の新規重みで開始。
- CNN: NarrowGridFeatures、各畳み込み4ch、観測解像度は維持。全パラメータ688,824。
- 操作側/価値側の観測処理を分離し、SeparateClipPPOで各勾配ノルムの上限0.5。
- 出力: 移動9+速度2の独立選択。ショット・ボムは操作・確率とも0。
- 報酬: 被弾-60のみ。2F判断、通常速度の実機プレイ、Normal/霊夢B。
- lr0.0001、3epochs、target_kl0.02、gamma0.9995、GAE0.95、batch64。
- 実機のゲーム終了時に収集済みエピソードで更新する運用は維持。
  シミュレーターの固定512判断rolloutへ変更したわけではない。
- 未採用の18択、短期危険度入力、補助損失、報酬変更は使わない。
- 詳細観測ログは無効、共有メモリ表示を継続。LIVE対象は新runへ自動更新。

## 接続と検証

実機のmodel選択を明示化し、statusへalgorithm/cnn_architectureを保存。
異なるモデル構成のcheckpointの暗黙再開を、launcherとlearnerの双方で拒否する。
DirectML workerへalgorithmを明示し、GPU更新・候補読込・実機保存再読込でもSeparateClipPPOを維持。
CPU fallbackの場合も同じクラスで更新する。通常PPOの旧runは従来どおり扱う。

既存11件、新規構成/再開保護3件の計14テスト成功。
小型CNNの模擬32判断で実DirectML更新、独立したactor/critic norm記録、保存再開を確認。

実機初回: 906判断・3被弾・報酬-180。DirectMLで3epochs更新。
GPU worker更新4.016秒、処理全体9.156秒。real-episode-1.zipを保存、再読込一致を確認。
実checkpointをSeparateClipPPOで再読込し、両NarrowGridFeaturesと別上限を確認。
実機32サンプルで操作と実入力bitが一致、射撃/ボムbit=0、hit以外の報酬=0を確認。
初回保存後、次のゲームも学習継続していることを確認。これは稼働確認であり回避性能改善の証明ではない。

証跡: artifacts/live-narrow-cutover-20260929/、artifacts/live-narrow-preflight-20260929/。
