"""Write evidence and limits for the passive live learner diagnosis."""
import json,pathlib,math
import numpy as np
root=pathlib.Path('artifacts/live-action-diagnosis-20260929')
s=json.loads((root/'summary.json').read_text());telemetry=json.loads((root/'telemetry.json').read_text())
contacts=[]
for i,d in enumerate(telemetry):
    if not d.get('reward') or not d['reward']['components']['hit'] or not d['player']:continue
    if d['entities']['bullets'] or d['entities']['lasers']:continue
    pos=d['player']['position'];enemies=d['entities']['enemies']
    if not enemies:continue
    e=min(enemies,key=lambda e:math.dist(pos,e['position']))
    history=[x for x in telemetry[max(0,i-3):i+1] if x['episode_id']==d['episode_id']]
    contacts.append(dict(episode=d['episode_id'],frame=d['game_frame'],position=pos,enemy=e,
        enemy_center_distance=math.dist(pos,e['position']),risk_grid_sum=float(np.asarray(d['ai_observation']['action_grid']).sum()),
        preceding_frames=[x['game_frame'] for x in history],preceding_bullet_counts=[len(x['entities']['bullets']) for x in history]))
(root/'enemy-contact-evidence.json').write_text(json.dumps(contacts,indent=2))
lines=['# 実機予測入力モデルの不調：受動診断','','2026-09-29。学習・入力操作・報酬を変更せず、保存記録と100秒間の受動観測を調査。',
 f"解析を固定した時点：{s['snapshot_updates']}更新・{s['snapshot_steps']:,}判断。新run live-learning-20260929-175742-f21a7b。",
 '比較対象は直前の小型CNN、live-learning-20260929-071910-9a4a78。双方とも死亡罰−60、射撃/ボム無効。','',
 '## 不調の確認','','| 比較 | ゲーム番号 | 平均ゲーム内時間 | 中ボス以降 | ボス到達 |','|---|---|---:|---:|---:|']
for key,label in [('current_last100','現在'),('old_same_updates','前回・同更新数'),('old_same_steps','前回・同程度の判断数'),('old_final100','前回・最終')]:
    r=s['windows'][key];lines.append(f"| {label} | {r['first']}〜{r['last']} | {r['seconds']:.2f}秒 | {r['midboss']}/{r['n']} | {r['boss']}/{r['n']} |")
lines+=['','時間は初期3機を使い切るまで。初被弾までではない。比較窓の開始条件はすべてNormal/霊夢B・残機2・1面1F。',
 '各構成1回の実機学習で、同一乱数の対照実験ではない。成績差を追加予測の単独の因果効果とは確定しない。','',
 '## 直接見えている失敗','','通常操作中（player status 1）の1,067サンプルで、上の選択確率平均57.17%、画面上部y<100の滞在67.29%。',
 '全1,318プレイサンプルでは上57.48%、入力マスク不一致0。方向が逆に送られているのではなく、方針自身が上を強く選んでいる。',
 '初回の敵本体接触と整合する被弾は以下の3例。直前も弾0、レーザー0、敵が至近距離にあり、被弾報酬−60を記録。','',
 '| ゲーム | 被弾frame | ゲーム内秒 | 自機位置（表示座標） | 最寄り敵の中心距離 | 予測入力合計 |','|---|---:|---:|---|---:|---:|']
for c in contacts:
    lines.append(f"| {c['episode'].split('-')[-1]} | {c['frame']} | {c['frame']/60:.2f} | {c['position']} | {c['enemy_center_distance']:.2f}px | {c['risk_grid_sum']:.0f} |")
lines+=['','予測入力は弾・レーザーのみで、敵本体の接触は含まない。この場面では全操作0となる。',
 '敵の情報自体はglobal_gridの密度等に存在するため、観測全体に敵が存在しないわけではない。',
 '敵接触イベントを直接フックした判定ではないが、弾/レーザーがない複数フレームの敵との重なりと被弾から、敵本体接触が強く支持される。',
 'また約12〜18秒の被弾では近接弾を確認。上部へ戻る偏りは最初の接触後も続いている。','',
 '## 学習側の問題','','同じ受動収集95場面を各保存モデルに入力。実際の学習経路上の価値校正ではなく、共通場面への反応診断。',
 '過去報酬入力は不明なので0に統一。モデル間比較では同条件とし、実行中方針の正確な再現とは扱わない。','',
 '| 保存モデル | critic最終Tanh飽和 | 価値予測の場面間標準偏差 |','|---|---:|---:|']
for key in ('new-1','new-25','new-100',f"new-{s['snapshot_updates']}",f"old-{s['snapshot_updates']}"):
    r=s['checkpoints'][key];lines.append(f"| {key} | {100*r['critic_saturation']:.2f}% | {r['value_std']:.6f} |")
lines+=['','新モデルは25更新で価値側が飽和し、現在は場面にほぼ同じ価値を返す。',
 '直近100更新の説明分散はほぼ0、前回の同更新数は0.892。場面・残機・危険度による将来評価を区別できていない状態。',
 '上偏重・早期被弾・価値予測の崩れが並行して定着している。死亡罰−60による尺度の問題は、③のシミュレーター結果とも整合する有力な改善対象。',
 'ただし前回も−60で学習できていたため、罰の大きさだけが唯一の原因とは言えない。実機で−1の効果を確認したわけでもない。','',
 '## 除外できた不具合と残る限界','','直近100回の更新・保存再読込は全成功、途中打切り/復旧なし。入力マスク不一致0。',
 'KL停止は28/100回、72回は3epoch完了。前回の同更新数は35/100回なので、過剰なKL停止を主因とする根拠はない。',
 '予測入力の追加結合も更新済みで、ゼロにすると方針は変化する。入力が配線されていない状態ではない。',
 '追加予測の敵本体未対応は実機で現れた適用範囲の不足。弾のみのシミュレーターでは検証されていなかった。','',
 '## 次の対策の優先順','','1. 採用した死亡罰−1で価値学習が崩れにくくなるか、次の実機新規学習で検証。既存モデルへ途中で報酬尺度だけを差し替えない。',
 '2. 敵本体接触を予測に含める。ただし敵の接触可能状態・当たり判定を実測照合し、非接触敵まで危険扱いしないようにする。',
 '3. 上方向固定などの操作偏りと最初の被弾時刻を継続指標にし、ゲーム終了時間だけで判断しない。','',
 '現在の実機runは停止・再開・報酬変更していない。③はシミュレーター標準v4へ採用した。','']
pathlib.Path('docs/live-action-diagnosis-20260929.md').write_text('\n'.join(lines),encoding='utf-8')
print('enemy-contact cases',len(contacts))
