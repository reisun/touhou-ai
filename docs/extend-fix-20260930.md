# 3面1UP取得時の再起動修正

2026-09-30。live-learning-20260930-204920-22e068 の16ゲーム目、12,899判断、3面中ボス突破時点で `unexpected life change; do not infer a reward` が発生。recovery処理がゲームPID16776を再起動して28296へ切り替えていた。ユーザーの1UP取得直後という報告と整合する。

原因はlive_learning.hit_eventsが残機差（取得前−取得後）に0と1しか許さず、正常な1UPの−1も例外としていたこと。−1を正常に受け入れ、被弾イベントを作るのは差が1のときだけに修正。増加時に新しい報酬を追加していない。残機が一度に2以上増減する異常や未検証の画面遷移は引き続き拒否する。

全6面の1機増加、報酬0、直後の被弾罰−1、異常な画面状態拒否を回帰テストに追加。tests.test_live_learning / test_live_rewards / test_live_transition / test_live_recoveryの34件成功。実機で再度3面1UPを取得する自然再現は、このテスト確認とは別。

学習はゲーム終了時の保存境界で止め、最終checkpointの重み・optimizer・乱数状態を保持して再開する。報酬v18、モデル構成、PPO設定は維持。

反映完了：旧runの17ゲーム目は既知のname-entry遷移で復旧更新となり、有効9,525判断を保存、325更新のreal-episode-17.zipを確保。保存ハッシュ・reload_verified・reload_rng_verifiedを確認した。再開初回は管理JSONの一時的なファイル競合で起動前に失敗したが、再試行でlive-learning-20260930-223320-270c4cとして起動。325更新・playing・errorなし・resume_rng_restored=true・報酬v18を確認し、240判断まで進行を確認した。3面1UPの実プレイ再遭遇は未確認。
