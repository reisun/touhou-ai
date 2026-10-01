# Fixed synthetic dodge diagnostic

Run `.venv/Scripts/python.exe scripts/train_fixed_dodge.py --steps 8192` from repository root. Outputs timestamped result.json and latest.zip under artifacts. CPU only, two Torch threads; no game control, live-model loading or modification.

A wall of 8px bullets descends at 3px/frame with a gap around x=48. Player starts at (0,330), moves 4.5px/frame or 2px/frame focused, diagonal speed normalized; bounds x +/-184, y32..432. Each decision advances 2 frames with eight collision substeps. Bullet half-size4 plus player half-size1 gives collision extent5. Survival ends successfully at160 frames after the wall passes the entire field; collision terminates immediately with -60. No positive reward. Geometry is synthetic, not a validated TH10 emulator.

Uses existing DualGridContract encoding, EvasionPolicy and separate DualGridFeatures, same PPO lr/epochs/gamma/lambda/entropy/KL as current live experiment. Fixed512-step rollout differs from game-over updates; one-life terminal episodes differ from live three-life games. This tests basic learnability, not an exact reproduction or transfer result. Training and assessment use the SAME fixed scene; no generalization claim. Stochastic evaluation uses50 episodes with fixed RNG123; deterministic evaluation uses one (identical) episode. Training RNG restored after assessment. Elapsed time includes assessments and prior saves; not pure training throughput.

Tests: stationary collision, scripted reachable gap, exact reset reproducibility, disabled shot rejection, observation-space match.

## First run: fixed-dodge-20260928-160900
Seed7,8192 training decisions, evaluation50 stochastic episodes per checkpoint: 0 steps8%,4096 steps6%,8192 steps10%. Deterministic success0 throughout. Total elapsed248.82s including evaluations. No convincing improvement in this short run. This CPU version does not yet deliver a large speedup over realtime with the full network; do not describe it as a high-throughput implementation. Player velocity observation is held zero and bomb_clock zero (no scheduled bombs); previous_rewards zero on nonterminal steps. These are simplified inputs, not exact live transition fidelity. Existing live training remains active.

## Numerical small-policy comparison
Run `scripts/train_fixed_dodge.py --numerical --steps 8192`. NumericalDodge subclasses identical FixedDodge physics;181 state values (player position, commanded velocity, focus and44 bullet positions/velocities) plus bomb clock. Independent pi/vf MLPs64x64; same PPO settings and reward. No oracle gap/action feature. Unlike first grid trial, own velocity is present, so representation and architecture change together; do not attribute results solely to CNN capacity. Five tests pass including300 matched physics transitions.

Run artifacts/numerical-dodge-20260928-161929: stochastic survival (50 trials, evaluation seed123) 8% at0,12% at4096,28% at8192;57.64s including evaluation vs248.82s grid trial (~4.3x faster). Reloaded continuation with training RNG19 reached16384,34% survival; additional48.66s. Deterministic policy fails at all checkpoints. Evidence of improved stochastic success, not solved task or proof of generalization. Further steps are not matched against a16384-step grid baseline. Live learner untouched.

## Controlled head-size comparison
`scripts/ablate_fixed_dodge.py` runs numerical/grid inputs crossed with64x64/256x128 policy and value heads. Both omit player velocity (numerical velocity slots zeroed), matching the original grid baseline. Physics, reward, PPO settings,8192 steps and seed7 held fixed. Original grid-large8192 checkpoint reused; all conditions evaluated for100 stochastic episodes with seed123, plus deterministic episode. The grid CNN encoder remains unchanged; this isolates head size, NOT total parameter count or CNN vs MLP architecture independently. Input pipeline remains a bundle of representation and encoder. Training-seed replication is not performed; evaluation trials quantify action sampling, not training variability.
Results artifacts/dodge-ablation-20260928-162415/results.json: numerical-large21%, numerical-small19%, grid-large13%, grid-small12% (100 stochastic trials each; all deterministic failures). Head-size differences1-2pp are small; observed input-pipeline difference7-8pp favors numerical but is modest and unreplicated. Grid-small still1,512,576 params vs grid-large1,669,760; numerical-small32,784 vs numerical-large161,552. Therefore this does not eliminate total encoder size as a cause. Conclusion: no observed benefit from shrinking downstream heads; investigate representation/encoder learning together, without claiming grid representation alone is proven faulty.

## Narrow grid encoder comparison
`scripts/ablate_narrow_grid.py` uses NarrowGridFeatures: all spatial convolution widths4, preserving kernel sizes, strides, layer counts, local12x12/global14x12 layouts, each branch output128, merged output256, player/reward branches and separate actor/critic structure. Heads remain256x128. Total parameters688824 vs1669760. FixedDodge input and dynamics unchanged; seed7,8192 steps, same PPO and100 stochastic evaluations seed123. This changes encoder capacity/width, not raster resolution. Different architecture initialization means a single run is not a paired statistical proof. Forward/logprob agreement, finite entropy, disabled actions and actor/critic gradient isolation checked.
Result artifacts/narrow-grid-20260928-163325/results.json: initial stochastic2% (50), trained12% (100), deterministic fails. Original grid-large13% (100), numerical-large21% (100). Narrow CNN width alone did not improve8192-step final performance in this run; cannot exclude capacity effects across seeds/budgets. Runtime183.10s incl evaluations; original run248.82s used different evaluation schedule/count, so timing is approximate, not clean throughput ratio. Reload passed. Do not infer grid representation is inherently unsuitable; encoder learning, credit assignment and short training remain candidates.

## Longer matched training comparison
`scripts/continue_dodge_comparison.py` resumes numerical-large (zero own velocity) and narrow-grid-large at8192 decisions with optimizer states preserved. Both restart training RNG19 and use evaluation RNG123,100 stochastic trials and one deterministic episode at8192,16384,32768. Evaluation preserves training NumPy/Torch RNG. Reward, scene, PPO settings and heads remain unchanged; continuation is not an uninterrupted replay of original RNG. Only one trained seed/scene; do not interpret evaluation trials as independent training replications.
Results artifacts/dodge-long-20260928-163957/results.json: numerical8192/16384/32768 stochastic21/42/40%; narrow CNN12/47/53%. All deterministic evaluations failed. Narrow CNN catches up with more training; earlier8192-step result insufficient to diagnose grid incompatibility. Final13pp advantage is one training trajectory and100 trials, not robust superiority. Both still fail frequently and deterministic failure suggests no reliable greedy solution. Numerical elapsed182.77s, narrow565.18s, including baseline/intermediate/final evaluations. Original models retained, final models saved separately; real learner unchanged.

## Greedy failure and frozen-probability audit (32768-step models)
`scripts/audit_dodge_greedy.py` records deterministic and100 stochastic trajectories, seed123. Numerical greedy chooses down37 times, dies frame74 at(0,432). Narrow CNN greedy chooses down-right37 times, dies frame74 at(184,432), passing the gap horizontally too early. Forcing any single first movement/focus combination (18 alternatives) then returning to greedy never succeeds. Successful stochastic paths mix horizontal/vertical actions; failures also include overshooting gap.

Control: freeze all action distributions at their initial-state values for entire episode, sample each decision with same RNG123. Numerical success rises40/100 ->48/100; narrow CNN53/100 ->57/100. This does not establish that freezing is beneficial across seeds, but demonstrates these fixed-scene success rates do not require ongoing observation-dependent policy feedback. Greedy top action never switches despite approach/bounds. Main finding: learned movement biases can yield random successful gap crossings without reliable state-dependent steering. Does not show encoder cannot perceive bullets, nor prove same behavior in real game. Full trace/control results artifacts/dodge-greedy-audit.json. Training unchanged.

## Left/right randomized gap diagnostic
Added SideGrid/SideNumerical (zero own-velocity slots), fresh seed7 models with large heads; NarrowGridFeatures on grid side. A seeded environment RNG picks +/- gap side per episode. Gap geometry mirrors x32..64 to x-64..-32 with44 sorted bullets. No side label or target direction is included in observation. Same physics, death-only -60,160F limit,512-step rollout and PPO settings;32768 decisions per model. Evaluation:50 trials for each side plus deterministic, fixed Torch seed123; mean x after20F logged. Final control mirrors bullet observations only (player and true physics unchanged) at every step. Evaluation RNG state restored afterward. Seven tests cover reachable gaps, opposite-view correctness without physics mutation, seed reproducibility, prior physics parity. This is one training seed and a synthetic task, not a real-game result.
Final results artifacts/side-dodge-20260928-170658/results.json:32768 steps numerical left48% right0%, CNN left50% right0% (50 each). Opposite-bullet observation control numerical46%/0%, CNN50%/0%; greedy fails both sides for both. Mean x after20F normal numerical-11.11/-9.69, CNN-10.04/-9.79; opposite numerical-9.23/-12.36, CNN-10.10/-9.86. Both develop left bias rather than observed gap-side steering at this budget; original right-gap success not evidence of reliable scene-conditioned avoidance. No claim of impossibility or proof that inputs are unreadable. Fresh seed7 only; note no training-side-count telemetry collected. Live learning untouched.

## Simulator speed improvements
Physics vectorizes independent bullets, preserving substep and player rounding order. Synthetic grid encoder caches bounded256 background grids (about33MiB maximum), preserves float16 roundtrip and exact observation values; unsupported scene variants fall back to DualGridContract.encode. Applies only to synthetic environment, not live encoder.

Offline batched evaluation uses actor extractor only, batch16, no redundant critic inference. Episode-local NumPy RNG seeded123 allows batch1 vs16 comparisons with identical randomness per episode. Protocol `numpy-per-episode-123-batch16-v1` differs from historical Torch-global sampling; do not compare small changes in historical percentages as performance improvements. Legacy evaluation functions retained. Models, distributions and deterministic decisions unchanged; batch floating-point roundoff can exist. CPU training runs inference1/update8 threads, model-local wrapper excluded from checkpoints. Training still one environment and512-step PPO rollout, not changed to vectorized training.

Measurements during ongoing real-game learning: CPU update threads1/2/4/8 took6.23/4.34/3.72/3.27s. Verified DirectML update3.05s but8.72s incl startup/transfer, so CPU chosen. First GPU instrumentation run serialized a measurement hook and was invalid; corrected run completed3 epochs. Final paired benchmark: learn512 reference9.76s vs optimized8.03s,32-episode evaluation8.46s vs3.71s optimized serial vs2.09s batch16; same evaluation outcomes/mean positions. Warm1000 encodes1.237s vs0.433s. Timings are measured examples, not guaranteed full-run speedups; live workload can vary. See artifacts/dodge-speed-comparison.json, dodge-speed-after.json, dodge-gpu-speed.json and dodge-physics-speed.json. Tests compare1000 exact transitions/all fields, both gap sides, actor probabilities, batch-size evaluation equality, optimizer/thread restoration and checkpoint reload.
Physics-only3000 decisions:7.257s ->0.550s (~13.2x). Final9 regression tests passed and all5 integrated training scripts compiled.

## Supervised gap-side readability diagnostic
`scripts/check_gap_classification.py`: fresh PPO actor feature extractor and256x128 actor MLP, replace final action logits with a2-class head. Train only actor+classifier, Adam1e-4, batches64,500 updates.256 matched left/right training pairs (512 observations),128 independent heldout pairs (256); positions vary x[-150,150], y[300,432], frames0..64, focus0/1. Each pair shares all non-bullet state; only wall gap side differs. Same zero-velocity input as preceding side task. Snapshots are synthetic sampled states, not necessarily reachable trajectories. Fixed train/test RNG31/47. Masking bullet channels makes paired observations identical and must give exactly50% balanced accuracy. Classification labels are provided ONLY as loss targets, never in observations. This tests learnability/readability, not learned PPO perception, control, or real-game transfer. Live models and rewards unchanged.
Final run artifacts/side-classifier-20260928-173822/results.json: heldout numerical25/100/500 updates95.31/100/100%; narrow CNN51.17/79.69/100%. At500, both all128 matched pairs correct on both sides; bullet-masked balanced accuracy50% for both. Numerical3.65s, narrow41.14s including generation/evaluation. Separate first100-update pilot retained173744. Evidence supports ability of same observation/actor architecture to learn side information under direct supervision; does not imply PPO-trained feature extractors already decode sides. Focus next on reward-to-perception/action credit assignment/optimization rather than assuming missing side information or insufficient architecture.

## Death-credit horizon diagnostic
`scripts/check_death_credit.py` changes only PPO gae_lambda from0.95 to1.0, gamma0.9995 and death-60 unchanged. Fresh seed7 numerical-large/narrow-CNN-large,16384 steps, same randomized side task. Uses optimized single-env pipeline and common per-episode evaluation RNG; stored original0.95 checkpoints at16384 re-evaluated with same new protocol in artifacts/death-credit-baseline.json. CPU threading/fast implementation differs from original training run, so not a bit-identical replay or multi-seed causal proof. GAE temporal-difference residual attenuation after30 decisions changes from (0.9995*0.95)^30~0.211 to0.9995^30~0.985; lambda1 still bootstraps at rollout boundaries, so not unrestricted full-episode Monte Carlo everywhere. No reward added or live learning changed.
Result artifacts/side-credit-lambda1-20260928-174137/results.json at16384: lambda0.95 baseline numerical left12/right6%, CNN28/0%; lambda1 numerical0/46%, CNN16/10%. Opposite-view lambda1 numerical0/44%, CNN16/12%; greedy0 both sides/all models. No evidence of successful scene-conditioned left/right steering from extending GAE horizon alone at this budget. It may improve one-side bias (numerical) without learning side use; not proof lambda tuning never helps. Next candidate diagnostic is supervised perception warm-start followed by PPO with same death reward, or explicit shorter credit-assignment task; not performed in this run.

## Supervised perception warm start followed by death-only PPO
`scripts/check_perception_warmstart.py` loads actor encoder AND actor256x128 MLP from side-classifier-20260928-173822 (100% heldout). Leaves randomized action head and critic untouched, with empty fresh Adam state; asserts this and transferred probe100%. Actor representation remains trainable. Same original gae_lambda0.95, death-60, randomized sides, fresh seed7 and16384 PPO steps; no classifier loss/labels during PPO. Classifier2-way output is only retained offline as a fixed diagnostic probe evaluated at0/8192/16384; decline in fixed probe accuracy indicates representation/readout drift, not necessarily complete loss of information. Baseline is re-evaluated original no-transfer0.95 run at16384 from death-credit-baseline.json. One seed; CPU speed changes can change floating-point training trajectories. Neither live learner nor its rewards changed.
Warm-start results artifacts/side-warmstart-20260928-174924/results.json: numerical8192 left18/right16%, probe100%, greedy initial direction responds to sides but overshoots;16384 left0/right48%, probe50%, opposite0/38%, greedy fails both. Narrow CNN8192 left30/right38%, probe100%;16384 left32/right46%, probe94.92%, mean x20F -15.41/+19.30; opposite observations left0/right0%, x20F +21.43/-16.88. Greedy left succeeds, right fails (single fixed scene each, not population100% claim). Supports learned observation-conditioned steering in warm-start CNN, unlike prior baseline28/0 and no opposite-view effect at same16384 budget. Still low survival and one-seed synthetic task; no claim of real-game cure or universal CNN advantage. Frozen original classification probe drift is not proof information fully disappeared in numerical model. Next useful diagnostic is freezing learned encoder+actor representation to preserve side cue while training action head, rather than adding reward directly.

## Three-reward avoidance diagnostic
`avoidance_rewards_sim.py` and `scripts/check_three_rewards.py` add hit-60, risk-reduction+2 once per wall, and observed pass+4 once per wall. Fresh seed7, original GAE0.95/PPO settings,16384 decisions/model. Stationary collision prediction uses current observed bullet positions/velocities and AABB over16F sampled0.25F; candidate executes chosen action2F then holds position. Reward only when stationary predicts hit and candidate predicts none; no reward on actual hit. Pass requires previously observed stationary threat and entire observed wall now below player by >5px, with no hit. One grouped wall budget+6, never reset during episode. Gap labels/future simulator trajectories are not used in reward. Synthetic wall grouping is simple; real-game reliable threat tracking still needs implementation/validation. Budget prevents unbounded farming/offsetting one-episode death, not all risk-seeking incentives. Early safe evasion without a recorded threat may receive no bonus (known coverage limitation).
Initial pilot using16F continuous action was aborted after testing exposed false negatives on valid2F maneuvers; final experiment restarted from fresh weights with2F action then hold, vectorized predictions. Two tests cover both encoders/both sides scripted pass, stationary hit and per-threat reward caps. Live learner unchanged. Compare original death-only16384 models re-evaluated in death-credit-baseline.json; one training seed only.
Final valid run artifacts/avoidance-three-20260928-181743 at16384: numerical left62/right0%; CNN2/36%. Death-only common-evaluation baseline12/6% and28/0%. Opposite observation control unchanged62/0 and2/36; greedy fails both for both. No observed conditional steering improvement despite higher aggregate success.100 evaluation episodes/model reward audit: numerical risk37, pass17, risk_then_death20, success_without_bonus14, average bonus1.42; CNN risk44, pass7, risk_then_death37, success_without_bonus12, avg1.16. Thus prediction improvement often not enough for eventual success; early safe evasion frequently misses bonus. Reward caps passed tests but do not establish absence of incentive to approach danger. These initial coefficients/eligibility rules should not be deployed as validated real-game reward. Earlier181530 pilot aborted, superseded by181743. Real tracking of moving threats/IDs remains unimplemented; current wall group is synthetic observed-motion grouping.

## Narrow-corridor alternating-gap diagnostic
`corridor_dodge.py` adds44 observed entities:16 stationary side-wall rectangles atx+-24, and two14-bullet horizontal waves moving3px/F startingy210/90. The first gap centers at+-8, second at the opposite side; random starting side, no label in observation. Side walls leave player centers within roughly+-19; each moving gap admits center in(1,15) or(-15,-1). Screen bounds unchanged, collisions use actual bullet half-size plus player half-size; quarter-frame substeps. Same160F horizon,2F actions, movement/focus only. Grid uses existing full encoder fallback; numerical retains181 inputs, bullet size not explicitly encoded (fixed scene geometry, not complete geometric equivalence to grid input). Three tests: stopping/large movement die; focused script with x limited+-8 survives both orders;500 matched grid/numerical transitions; mirrored observation parity. New baseline training uses death-60 only, seed7, original PPO settings,16384 steps each; prior3 rewards not silently transplanted because they assumed one wall. Tests establish task feasibility, not AI competence or real-game equivalence. Analysis of x20F only includes episodes reaching20F; empty sample set is null in updated evaluator.
Baseline results artifacts/corridor-dodge-20260928-183608/results.json at16384: numerical first-left/first-right2%/8%, narrow CNN2%/6% (50 each). Both deterministic policies fail both configurations, and opposite observed gaps leave outcomes unchanged. Thus no evidence of learned gap-conditioned fine steering at this budget. Initial numerical greedy episodes ended before20F; undefined x metric stored as null rather than NaN. Scripted focused maneuver succeeds both variants with max |x|8, while stationary and large horizontal motion fail. This adds a test for fine/alternating steering, not a faithful Touhou emulator or solved task. Future rewards should be evaluated across wide and narrow scenarios. Live learner untouched.

## 2026-09-28 秋符「オータムスカイ」再現診断

Normal BossCard1 のECL、弾ANM、実行中コードの読み取り照合に基づく最初の600Fの環境を追加。詳細は `docs/autumn-sky.md`。実機の学習は変更なし。
上端消去を修正した最終実行: `artifacts/autumn-sky-20260928-190412`。各16384判断、死亡-60のみ、24評価seed。確率選択の生存率は数値12.5%、CNN4.2%。平均生存6.90秒/5.85秒。最大確率選択は29.2%/0%。初期観測固定では両者45.8%で左下固定操作と一致。今回も観測に応じた回避の改善は確認できず、固定操作/観測固定の対照が必要。4テスト成功。
実機の乱数列・アニメーション境界・ボス移動を含むフレーム単位の一致は未確認。数値入力は全760発対応の3805入力へ拡張。

## 2026-09-28 速度固定・相対座標の比較

`docs/autumn-ablation.md`、`artifacts/autumn-ablation-20260928/summary.md`参照。
17条件、各16384判断、死亡罰のみ。数値の高速固定/速度切替は絶対・相対とも3学習seed。
高速固定で相対座標は3seedすべて確率選択の生存率・平均生存時間が改善。
3seed平均: 絶対2.8%・6.71秒、相対19.4%・7.56秒。相対の観測固定対照8.3%・7.13秒も上回る。
最大確率選択は絶対40.3%/相対38.9%で相対優位なし。速度切替では優位は一貫しない。
ユーザー訂正: 次回以降の数値入力の主条件は相対座標＋速度切り替え可能。高速固定への限定は取り消す。実機設定は未変更。
出力の検討候補: 現在の移動9方向/速度2種類の独立出力と、組み合わせ18通りの一括出力を比較する（未実施）。
CNNの速度固定による反応回避の改善は未確認。既存固定操作・観測固定対照は維持する。

## 2026-09-28 方向×速度18通り出力の比較

`docs/joint-movement.md`、`artifacts/autumn-joint-20260928/summary.md`参照。
数値相対座標3seed、CNN1seed、各16384判断・速度切替・死亡罰のみで比較。
確率選択の10秒生存率は数値独立8.3%→同時18通り6.9%、CNN4.2%→4.2%。
最大確率選択は改善するが観測固定が同等以上。反応回避の改善による採用根拠は得られず。
今後も数値は相対座標＋速度切替可能、独立出力を基準とする。18通りは実験実装として保存。
射撃との両立は未評価、実機設定未変更。2テスト成功。

## 2026-09-28 観測から操作を学ぶ過程の診断

詳細は docs/observation-credit-audit.md。保存モデルの新規16試行と128場面を診断。
数値独立は即時2Fの生存操作に87.7%（一様83.1%）を配分するが、16F停止参照では56.0%（一様53.9%）。
価値予測平均-18.10に対し実現割引報酬-43.56。actor/critic共通勾配上限の干渉も確認。
同一8バッチ・保存Adam状態の一歩比較では別上限にするとactor更新normが約5.98倍。
別上限、価値校正、実観測由来の危険予測補助課題の順に比較候補を記録。長期改善は未検証、実機未変更。

## 2026-09-28 更新上限分離の比較・限定採用

docs/separate-clip-comparison.md参照。数値/CNN双方3初期値×96未使用弾幕で比較。
確率選択の完走率: 数値12.15→15.63%（不安定で未採用）、CNN3.47→13.54%（全3初期値改善）。
CNNだけ今後のオフライン確率的学習の基準へ別上限を採用。configs/autumn-learning-baseline.jsonと新runnerへ反映。
CNNの観測固定は15.28%、最大確率選択は改善せず。反応回避や実機性能の改善は未確認。実機未変更。
死亡罰尺度-60→-1も各1初期値で予備比較。数値悪化、CNN改善するが固定観測同等で未採用。
固定観測評価のキャッシュ高速化を採用。分離更新の3テスト成功、保存再開・標準設定の推論も確認。

## 2026-09-28 実観測由来の危険予測補助学習

docs/risk-auxiliary-comparison.md参照。数値は従来上限、CNNは別上限、独立9方向+速度2、各16384判断。
初期方式3seed×96評価: 数値完走12.15→6.60%、CNN13.54→17.01%だが不安定で平均生存時間悪化。両方未採用。
危険予測の約65%は観測固定でも得られ、操作別正負均等評価はほぼ50%。方向の偏りを覚える近道を確認。
操作別の正負均等損失も各1seedで追加比較したが、回避改善は不十分で未採用。
初期6モデル・追加2モデル保存。教師近似は32場面576操作で97.57%一致、生成中央値0.248ms。
既存設定維持を確認。実機未変更。計7テスト成功。次候補は短期危険度の操作入力への直接追加（未実施）。

## 2026-09-29 短期危険度の操作入力への直接追加

docs/direct-risk-comparison.md参照。移動9+速度2独立出力を維持し、18危険指標をactor最終層へ追加。
両入力3seed×16384判断・各96弾幕評価。確率完走率:数値12.15→7.29%、CNN13.54→14.58%。
数値は悪化、CNNは不安定で平均生存も悪化。推論時の危険入力除去で成績が下がる傾向もなく両方未採用。
数値最大確率選択は改善するが危険入力除去でも維持。CNN最大確率選択は全3seedで観測固定と一致。
入力認識を計算で助けても、死亡罰から操作へ反映する過程の改善は不十分。6テスト成功、標準設定維持、実機未変更。

## 2026-09-29 採用済み設定で独立9+2対結合18を再比較

docs/joint-adopted-comparison.md参照。数値は共通勾配上限、CNNは別上限。未採用の危険入力なし。
各3初期値16384判断、同じ96弾幕評価。確率完走率: 数値12.15→7.29%、CNN13.54→15.97%。
CNNは2初期値改善・1悪化で不安定。数値悪化。両モデルの基準は独立9+2のまま維持。
18分布は方向・速度の結び付きを持つが、それだけで有利な回避学習へ繋がるとは確認できない。
7テスト成功、6モデルと対照評価を保存、実機未変更。

## 2026-09-29 小型CNNの9+2対18・安定性追試

docs/cnn-output-stability.md参照。新規7初期値×両方式、16384判断、各192弾幕。旧3組も評価を192へ拡張。
追加7初期値でも縮小傾向はあるが、不確かさを含めて安定性向上が確認できたとは言えない。
追加分の標準偏差:9+2 9.56pt、18 5.23pt。比0.547、95%区間0.216〜1.831。
既存設定・実機設定未変更。モデル14個と全20モデル分の評価を保存。
