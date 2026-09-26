# 報酬v10・ボム12F判断（2026-09-27）

追記：進行度の実機検証と接続により、現在は報酬v11。
[検証結果・対象範囲・互換性](progress-validation-20260927.md)を参照。
以下の「未検証・無効」はv10実装時点の記録。

## 実装状態

`docs/reward-design-proposal.md` の確定内容を実装。進行度の計算・初回制限は実装したが、
**4節目の実機検出は未完成で無効**。確証のない到達／撃破を推測加点しない。
v10で長時間学習は開始していない。現在の学習キャンペーンは停止状態。

| 内訳キー | 計算 | 実機への接続 |
|---|---|---|
| damage | 5 × 有効HP / 1000 × (1 + 0.5 × Power) | イベント時点でbomb.state=0のみ |
| progress | 20 + 30 × 予備機数 + 5 × Power | 検出の検証待ち。現在0 |
| hit | −10 / 被弾 | 既存の予備機数減少検出 |
| power_down | 0 × 観測されたPower減少 | 項目を残し、係数0 |

通常撃破、ボム専用加点・罰、Power取得、低Power維持、面クリアの独立項目は廃止。
これらはWEIGHTS・報酬内訳・直前報酬入力に残していない。生の撃破イベントやPower観測は保存する。
Power減少のamountは前後のPower差（正の減少量）で、同じ区間内の取得と消費の相殺は分離しない。
係数0なので報酬合計に影響しない。Powerはraw/20、raw範囲0〜100。

## ボムの操作とPPO

実装は `touhou_ai/timed_bomb.py`、接続は `live_learning.py`。

- 方向・ショット・低速は従来どおり2ゲームFごと。
- ボムは収集開始を原点として12ゲームFごと（2F操作の0, 6, 12, ...番目）。
- 判断時は出力確率から抽選。ONなら2F押下し、その後10Fは必ず解放。区間中の再抽選はない。
- 初期ON確率は `1 - 0.5**(1/25) = 0.02734505258771447`。
  ボム出力の初期重みを0、bias差をlogit(p)として、入力によらずこの初期確率を保証する。
  重み・biasは通常の学習対象であり、固定値・上限ではない。
- Power不足や発動中でも、判断時の確率をマスクしない。非判断時の解放は周期の仕様。
- 50%閾値の発動方式は採用していない。収集はstochastic forwardを使用。
  再読込の同一性テストのみ従来のdeterministic予測を使用する。
- `bomb_clock` 1値を観測・rolloutに保存。値は12F周期の位相（0, 1/6, ..., 5/6）。
  actor/criticの特徴に含め、PPOのシャッフル後も同じ判断位相を使う。
- 非判断時のボムは抽選せず0。joint log probabilityとentropyのボム成分を0とする。
  PPO ratio・KL・entropy・勾配にも存在しないボム判断を加えない。
  判断時は他の3操作とボムのjoint probabilityを通常のPPO目的に使用する。
- メニュー・ロード・会話の自動操作中は収集時計を進めず、ボムを抽選しない。
  各プレイの開始／Continueで原点を再設定。ステージをまたぐ同一プレイでは時計を維持。
- gammaは追加決定により **0.9995 / 2ゲームF**。ボム判断の12Fを割引単位にしない。
  以下の割引率追記を参照。先行検証成果物の0.997は当時の記録として保持する。

## ボム境界・ダメージの意味

`th10_gate.js` の実HP減算0x40e1b6で、イベントと同時に以下をコピーする。

- `bomb_state`: `*( *(0x4776ec) + 0x28 )`
- `power_raw`: `*(0x474c48)`

`timer > 0` は使わない。状態が0/1以外、ポインタ不在、Power不正時は観測をエラーにし、
非発動として黙って加点しない。Power倍率はこのイベント時点の値を使う。
HP上限・フェーズHP下限による有効ダメージのクリップは既存処理を維持。

2F後のsnapshotだけでは境界を判別できない実例を今回採取した。
frame2651のdamageイベントはbomb_state=1、同じ操作終了のframe2652はstate=0。
イベント時点を使うことで、このダメージを除外できた。
場面移行ヘルパーがsnapshotを取り直した場合も、元の操作のイベントを保存して報酬に渡す。

これは攻撃元の厳密な分離ではない。ボム中のショットも除外する。
state=0になった後のボム残存ダメージは未保証で、存在すればショット近似に混入する。
イベントが発生する内部更新順序と全攻撃オブジェクトの寿命までは今回証明していない。

## 進行度の定義と未解決点

計算器は `midboss_arrival`, `midboss_defeat`, `boss_arrival`, `boss_defeat` を受け付け、
同一プレイの `(stage, milestone)` ごとに一度だけ加点する。
異なるIDで同じ節目が来ても二重加点しない。次面の同じ節目は別、Continueは別プレイ。
必要なイベント値はstage・milestone・イベント時点の予備機数・Power・検証済みsource・ID。
攻撃手段は問わない。単独の面クリア報酬はない。

採用する意味は「識別された中ボス／ボスとの戦闘開始」と「その敵の最終撃破」。
フェーズ切替・時間切れ退場・画面外消滅は最終撃破にしない。
現状の実機データからこれらの意味を満たすproducerを確定できず、
`VERIFIED_PROGRESS_SOURCES` は空。source未検証のイベントは拒否する。
進行度の報酬欄は将来の受け口として残すが、現在の実機では一切加点しない。

今回の調査:

- Steam固定バイナリの既存HP減算から撃破分岐までを再読取り。
  0x40e1b6の減算後、0x40e1c2で0x4127a0を呼び、返り値に応じた処理の後、
  0x40e212以降でHPを再検査し、0x40e231の死亡スコア分岐に入る。
  コールバックとHP閾値が介在するので、HP減少や単一のkillだけでは
  中ボス／ボスの同定と最終フェーズ撃破の意味を保証できない。
- 360Fの今回の実機記録ではbossフラグ付き同一敵address=420097000が継続。
  frame2578: HP6820/8400、frame2936: HP2580/8400。killイベントは0。
  この範囲は到達以前から戦闘中なので到達イベントの検証にはならず、
  midboss/bossの種別、最終撃破、timeoutの対照例も未取得。
- 今後は固定ECL処理／敵ライフタイム／最終フェーズの対応を突き合わせ、
  到達・通常撃破・タイムアウトを実機で比較してからproducerを有効化する必要がある。

**進行度が主評価という設計全体の実機運用は、ここが未完成。**
現段階のv10を完成した進行度学習として扱わない。

## 観測・互換性・表示

`th10-dual-grid-v2` / `th10-focused-bullets-v2`、報酬は `th10-rewards-v10`。
直前報酬は `[damage, progress, hit, power_down]` の4値、正規化scaleは `[1,100,10,1]`。
各値は `(v/scale)/(1+abs(v/scale))`。Power減少の入力値は係数0により0。
別途既存のPower観測は維持。時計1値を追加した。
設定JSON・manifest・モデルメタデータ・各stepログ・OBS表示を更新。
進行度は「検証待ち」と表示する。

旧v1/v9チェックポイントは契約不一致として再開を拒否。重みの自動移植は行わない。
旧成果は元の場所に保持。現行クラスに旧チェックポイントを直接読み込んで利用することも
サポート対象ではなく、必要なら対応する旧コード・契約で扱う。

## 検証と証拠

- Python91テスト実行、90成功・1件skip（WindowsではDocker SIGTERM検証を除外）。
  `tests.test_timed_bomb` の3テスト成功。
- `tests/test_timed_bomb.py`: 初期確率・5秒50%・12F周期・非判断のaction/logp/entropy・
  ボム出力層の非判断勾配0・PPO更新で初期確率が変化可能・保存再読込・収集器接続。
- `tests/test_live_rewards.py`: Power倍率、イベント時点優先、未知値拒否、
  廃止項目の拒否、Power減少0、進行度計算・節目単位重複防止（テスト専用source）。
- `tests/terminal_input_guard.cjs`: 発動／終了を同一バッチに含めたevent-time記録、
  不明ポインタ拒否、既存のダメージクリップ・入力解放。
- `artifacts/reward-v10-acceptance-20260927-a/trace.jsonl`: 実機180操作×2F。
  発動Power raw36→16。ボム中4128 HP除外、非ボム162 HPから1.539点。
  ショットを併用した近似検証であり、厳密なボム由来HPの集計ではない。
- 同フォルダstatusは追記決定前の3内訳での中間検証記録。上書きしていない。
- `artifacts/reward-v10-final-contract-20260927/status.json`: 最終4内訳で実機記録を
  再評価し1.539点を再確認。実シーン＋合成行動／報酬の独立した12step・1epochの
  DirectML PPO更新、混在周期logp一致、更新後checkpoint再読込成功。
  既存モデル・実機学習キャンペーンの更新は0。

## 停止と保全

`artifacts/live-learning-20260926-231728-79a274` に既存STOP機構で停止要求。
status=stopped、途中trajectoryは更新しない終了理由。最新はreal-episode-33.zip、
累計224389step・130更新、reload_verified=true。
SHA-256 `5caa42331cfaf1a9c8506b90d0c2e8ecbdeb1cdd55a1a4c3a38d9cb9379e9352` の一致を確認。
STOP維持、学習未再開。最終実機はPID8896、1面frame2936、ポーズ[2,2]、入力全て0。

関連: [設計](reward-design-proposal.md)、[ボム状態の先行検証](bomb-state-validation-20260926.md)。

## 割引率の追加決定・検証（2026-09-27）

ユーザーの追加確定でγを0.997から **0.9995** に変更。
`configs/sharu-inspired-v1.json` を実際のPPO生成に使い、`learning_discount.py` の
`th10-discount-2f-v1` と一致を検証する。報酬式v10・観測v2は維持し、
割引の契約は独立したmanifest/モデルメタデータとして保存する。
旧γまたは契約欠落の再開は、学習開始・出力作成前に拒否。
PowerShell側でもactive-run記録を書き換える前に拒否する。

設定値のGAE λ=0.95、n_steps=2048、batch_size=64、n_epochs=10は変更していない。
既存の実機収集はゲームオーバーまでの可変長rolloutなので、n_steps=2048を
新たな強制切断点にはしない。ボム時計にかかわらず全2F遷移をbufferへ保存する。
メニュー・会話・ロードの自動操作は従来どおり収集外であり、ここで割引ステップを追加しない。

収集側bufferのgamma・lambdaは設定と同じ値。DirectML worker側もロード済みモデルの
gamma・lambdaをbufferへ設定し、収集側で計算したadvantage/returnをそのまま利用する。
候補モデルがgammaまたはlambdaを変えていた場合は拒否し、既存のCPU復旧経路へ戻る。

`tests/test_learning_discount.py` で以下を検証:

- 本番設定によるPPO生成、モデル/bufferの実値0.9995、保存・再読込後の一致。
- DictRolloutBufferとGridRolloutBufferのadvantage/returnを独立した逆向きGAE再帰と照合。
- terminalでは最終bootstrap=0、打切りでは `gamma × 最終価値推定` を使用。
- rollout内のepisode境界を越えてbootstrapしない。短いrolloutにゼロ埋めを混ぜない。
- 旧0.997、契約欠落、12F単位と称する契約を拒否。
- `gamma**900 = 0.637556398`、`gamma**1800 = 0.406478160`。
  30秒で約63.76%、60秒で約40.65%。

通常のSTOP/予算終了による途中trajectoryは従来どおり破棄し、学習更新しない。
bootstrap付き更新が適用されるのは、既存の復旧条件を満たした有効な非終端prefixなど。
この停止・復旧方針は変更していない。

長期伝播の制約: γを上げてもGAEのTD残差の直接重みは `(gamma × lambda)**k`。
今回の積は0.949525で、半減は約13.38ステップ（0.446秒）。900ステップ先の
残差が一度のGAE計算で大きく直接伝わるわけではない。遠方の価値はcriticの推定と
反復更新に依存し、観測不足・進行度検出未完成・rollout切断の影響も残る。
λやrollout長は無断で調整していない。

証拠: `artifacts/reward-v10-gamma9995-20260927/status.json` と
`isolated-synthetic-update/result.json`。保存済み実機記録の報酬1.539を再確認し、
独立した12step・1epochの診断モデルでDirectML更新・保存・再読込を実施。
model gamma/buffer gammaは双方0.9995、lambdaは双方0.95、実機キャンペーン更新は0。
12stepは診断限定の短縮で、本番設定を変更していない。

追加後の全テスト: 94件実行、93成功・1件Windows上のDocker SIGTERM検証skip。
PowerShell構文・diff whitespace検査も成功。最新real-episode-33.zipのハッシュ一致と
STOP/status=stoppedを再確認し、長時間学習は再開していない。
