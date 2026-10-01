# 回避専用の新規学習（2026-09-28）

目的：攻撃・進行報酬と復帰後の攻撃資源の利益を外し、現行の観測・ネットワークで回避を学習できるか調べる。

- 新規run: live-learning-20260928-012839-ba5dc8。旧重み・optimizerは読み込まない。
- 旧run live-learning-20260927-171751-ed391e は328更新、695,327判断で保存・停止。モデルは保持。
- 観測: th10-dual-grid-v6。既存CNN・PPO設定・2F判断を維持。
- 報酬: th10-evasion-death-only-v1。残機減少1回につき-60。他は0。
- 操作: 9方向と低速。ショット・ボムは確率0。PPOのサンプリング、log-prob、entropy、更新時も同じ制約を使用。
- 起動: scripts/live-learning.ps1 -Action rehearse -ContinueManaged -DualGrid -EvasionOnly -DirectML -Continuous -NoUI -MaxSteps 18000 -MaxSeconds 900
- ResumeLatestは付けずゼロから開始。旧報酬モードとの再開互換性チェックを維持。
- 詳細観測ログは無効のまま。共有メモリの表示とモデル保存は維持。

検証: scripts/check_evasion_only.py により強制無入力、収集/更新log-prob一致、有限entropy、CPU PPO更新、保存・再読み込み、hitだけの報酬と重複除外を確認。既存timed_bomb 4件とlive_learning 5件も成功。

評価上の注意：3デスで終了する試行の非割引総報酬は-180で一定になるため、総報酬だけで上達を判定しない。判断数/生存時間、初被弾時刻、固定経過時間内の被弾率を使用。gamma<1によりデスを遅らせる学習信号はある。無射撃なので敵の撃破・弾消しがなく通常プレイと難易度は異なる。低Powerでもボムを利用できないため、復帰Powerをボムに使う直接の利益はなくなるが、復帰無敵などは残る。

## 起動後の検証

初回実機更新はDirectMLのmasked log-probで無効操作の-infと0の積がNaNになり、CPUへの自動復旧で正常保存された。無効操作のlogitを有限の-1e9へ変更。float32の選択確率は0のままで、互換log-probも有限となることを確認。
回避専用モデル2更新目を保存し、live-learning-20260928-013127-4254d8へ継続。元の攻撃モデルは引き継いでいない。実機32サンプルで射撃/ボムの確率・操作とも0、hit以外の報酬0を確認。

修正後の実機DirectML更新成功：累計3更新・3,001判断。GPU更新11.812秒、保存再読込成功、次試行のプレイ継続を確認。

## 第2実験：更新量の抑制

ユーザー承認により旧実験を354更新・517,160判断、real-episode-352.zip（run: live-learning-20260928-013127-4254d8）で保存停止。
新規初期モデル（seed 7）から、回避実験専用の上書き設定 learning_rate=0.0001、n_epochs=3、target_kl=0.02 を採用。報酬・観測・構造・batch_size=64・gamma・GAEは維持。provisional_ppoは通常学習用として保持。
SB3の停止判定はミニバッチ近似KL > 1.5*target_kl（0.03）。厳密なKL上限ではなく、閾値を超えたミニバッチの更新を行わず打ち切る。
GPU候補検証は、target_kl有効時に1〜n_epochsの処理周回数を受け入れる。KL未設定では従来どおり全周回数を要求。0周・予算超過・設定改変は拒否。早期停止によるCPU再更新を防止。
異なるPPO設定のモデルを暗黙に再開しない。LIVE表示は新規runへ自動切替し、旧実験の履歴とは分離する。
関連テスト: DirectML 3件、live_learning 5件、discount 3件成功。

## Separate actor/critic feature experiment (2026-09-28)
- Motivation: update 243->244 showed opposing saved-block effects; critic-gradient interference is a hypothesis, not established cause of stalled performance.
- Config: evasion_policy_overrides.share_features_extractor=false. Fresh seed 7; death -60 only, shot/bomb disabled; lr .0001, epochs 3, target KL .02 unchanged.
- Actor and critic now have distinct DualGridFeatures instances. Resume rejects feature-sharing mismatch in launcher and learner; old manifests default to shared.
- Validation: scripts/check_separate_evasion.py verifies gradients cannot cross extractors in either direction, forced actions, finite PPO update, save/reload; 8 live-learning/DirectML tests passed.
- Compare survival seconds, initial death time and progress over equivalent 50-episode windows at 200+ updates against the shared fresh run 085557-4f30fd and its continuation. Single-seed live trajectories are not paired; this is an initial comparison, not proof of causality.
- Detailed logging stays off. Historical shared checkpoints retained. Separate features increase model cost; verify live DirectML update before completion.
- Started live-learning-20260928-124148-6c9ab8 fresh; previous shared run stopped and saved at update 286. First live DirectML update passed (3 epochs, 5.84s worker, checkpoint reload verified). Continuous learning active, detailed_logs=false.
