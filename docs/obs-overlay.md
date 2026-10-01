# OBS browser sources

- `http://127.0.0.1:18767/?mode=obs1`: 530 x 1080.
- `http://127.0.0.1:18767/?mode=obs2`: 480 x 700, black background alpha 0.9 (10% transparent).

Use the corresponding browser-source width and height. Existing monitor routes and
assets are unchanged. The local dashboard must be running. These pages are read-only.

OBS1 uses the same telemetry schema, entity coordinates, collision rectangles,
input bits and direction ordering as the existing monitor. Pale gold is the most
probable direction; solid gold is the observed input. Stale input is not highlighted.
Growth shows verified real-game update returns and a trailing 20-update average;
the history currently includes different starting stages and recovery truncations,
so it is not a controlled evaluation of improvement.

OBS2 aggregates collector JSONL logs, not the potentially lossy latest-frame SSE.
All three displays share the exact wall-clock interval (now-30, now], refreshed
once per second. Stopping collection naturally empties the window after 30 seconds.
Value is V(s), the policy's discounted future return estimate, not a 30-second
forecast. Error is one-step TD residual r + gamma*V(next) - V(s), or r - V(s) on
confirmed terminal transitions. No cross-episode bootstrap is used. The latest
nonterminal sample has no error until the next observation arrives.

Only connected reward components are shown numerically (currently hit, power and
normal stage progression). Native damage and kill events are connected in reward
v3; see `combat-rewards.md` for the real-game validation scope. Growth histories
are restricted to the current reward version and bullet observation contract.
No simulated values are inserted into the live views.


## OBS1: 学習ごとの最大到達点（右軸）

試行報酬（灰色）・直近20回平均（金色）に、各回の最大到達点（青緑）を追加。
`ObsStats` は検証済み学習記録の run と episode 番号で `episode-N.jsonl` を対応付け、
telemetry.episode_id の一致も確認する。既存の events ログを利用するため、学習側の
ログ形式・報酬・チェックポイントは変更不要。完了ログの集計はサイズとmtimeでキャッシュする。

confirmed=true、kind=progress、検証済みsourceのイベントのみを採用し、
rank = (stage - 1) * 4 + 節目番号（中ボス到達=1、撃破=2、ボス到達=3、撃破=4）で
その試行内の最大値を growth[].max_progress={stage,milestone,rank} として返す。
報酬点数・残機・Powerには依存せず、前回より低い到達点もそのまま描画する。
APIのイベント未記録値はnullのまま保持する。現行バージョンの表示では、ユーザー指定により未記録をスタート（rank=0）として点と線を描く。右軸の下限は常にスタート。節目到達件数を表示する。
右軸は観測範囲に応じて面と節目の目盛を間引く。既存の報酬契約による履歴絞り込みは維持。

反映: dashboardサーバーを通常の手順で再起動し、OBSブラウザソースを再読み込みする。
学習プロセスとゲームの再起動は不要。実行中のサーバーはこの変更では停止していない。
検証: `python -B -m unittest tests.test_obs_stats tests.test_obs_growth tests.test_live_rewards`、
`node tests/obs_growth.cjs`。ブラウザで530×1080、成長曲線下端1054px、横幅530pxを確認。

## 学習開始時の表示更新

scripts/live-learning.ps1 rehearse は既存の表示サーバーを再起動し、報酬・観測契約の変更をOBS1/OBS2の集計へ反映する。-NoUIでも既存サーバーは更新し、ブラウザ画面を新たに開かない。-NoUIでサーバーが未作成なら起動しない。学習中の表示だけを更新する場合は従来どおりdashboard.ps1 stop / startを使用する。

観測 th10-dual-grid-v3 は全体15ch。青い矩形が現在の自機ショット判定の再描画。OBS2の横棒は全項目共通で0〜max(100,各項目の絶対値)。
