"""Render verified aggregate results into the experiment report."""
import json,pathlib,numpy as np
root=pathlib.Path('artifacts/spatial-inputs-20260929')
summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
bench=json.loads((root/'benchmark.json').read_text(encoding='utf-8'))['results']
response=json.loads((root/'input-response.json').read_text(encoding='utf-8'))['results']
names={'control':'基準（2px）','pixel1':'① 1px','action_grid':'② 操作別予測','geometry':'③ 当たり判定と経路'}
lines=['## 実測結果','','同条件で一から学習した12モデル、各16,384判断。主評価は確率選択、各96弾幕。',
       '各モデルの保存再読込一致と、4評価方式×96走行の完備・集計一致を確認（合計4,608走行）。','',
       '| 入力 | 初期値7 | 初期値17 | 初期値27 | 平均完走率 | 平均生存時間 |',
       '|---|---:|---:|---:|---:|---:|']
for v,name in names.items():
    r=summary[v];rates=' | '.join(f"{p['survival']*100:.2f}%" for p in r['pairs'])
    lines.append(f"| {name} | {rates} | {r['means']['sample']['survival']*100:.2f}% | {r['means']['sample']['seconds']:.3f}秒 |")
lines+=['','| 入力 | 基準からの完走率差 | 差の95%再標本化区間 | 改善 / 悪化した初期値 |',
        '|---|---:|---:|---:|']
for v,name in list(names.items())[1:]:
    r=summary[v];lo,hi=np.array(r['paired_seed_case_bootstrap95'])*100
    lines.append(f"| {name} | {100*r['paired_mean_difference']:+.2f}ポイント | {lo:+.2f}〜{hi:+.2f} | {r['improved_seeds']} / {r['worse_seeds']} |")
lines+=['','区間は対応する初期値3組と、共通の評価弾幕96条件を再標本化した10,000回のbootstrap。',
        '初期値3組の探索的比較であり、安定性や他の弾幕への一般化を確定するものではない。','',
        '### 観測固定との比較','','| 入力 | 通常・確率選択 | 固定・確率選択 | 通常・最大確率 | 固定・最大確率 |',
        '|---|---:|---:|---:|---:|']
for v,name in names.items():
    r=summary[v]['means'];values=' | '.join(f"{r[mode]['survival']*100:.2f}%" for mode in ('sample','frozen_sample','greedy','frozen_greedy'))
    lines.append(f'| {name} | {values} |')
lines+=['','観測固定は開始時の全観測を保ったまま実際の弾幕を進める診断。弾だけを取り除く比較ではない。',
        '同じ完走率でも全操作が同一であるとは限らず、通常−固定の差だけで回避能力を証明しない。','',
        '### 入力を変えたときの操作反応','',
        '全モデル共通の診断64場面で、空間グリッドを別場面のものに置換した際の操作分布変化を測定。',
        'TVは確率質量が動いた量（0〜1）。独立9＋2の確率を掛け合わせた18組で比較しており、18出力への変更ではない。','',
        '| 入力 | グリッド置換の平均TV（3モデル平均） | 追加入力を0にした平均TV |',
        '|---|---:|---:|']
for v,name in names.items():
    rows=response[v];tv=np.mean([r['grid_permutation']['mean_policy_tv'] for r in rows])
    extra=f"{np.mean([r['added_input_zero']['mean_policy_tv'] for r in rows]):.6f}" if 'added_input_zero' in rows[0] else '—'
    lines.append(f'| {name} | {tv:.6f} | {extra} |')
lines+=['','この置換は入力への反応を測る診断であり、操作の正しさ・回避成績の改善を直接測るものではない。',
        '追加入力を0にする診断も、学習した状態分布から外れる可能性がある。重みの更新量等はinput-response.jsonに保存。','',
        'さらに分岐実行済み59場面のうち安全・衝突操作が混在する場面で、実際に衝突する操作の選択確率を計算。',
        '16Fは2F移動後に14F停止する仮定であり、その間に方針が再判断し続けたときの衝突率ではない。','',
        '| 入力 | 追加情報あり−追加情報0：2F衝突操作の確率差 | 同16F |',
        '|---|---:|---:|']
for v in ('action_grid','geometry'):
    rows=response[v]
    delta={h:100*np.mean([r['actual_collision_probability'][h]['mixed_mean_probability']-r['added_input_zero']['actual_collision_probability'][h]['mixed_mean_probability'] for r in rows]) for h in ('2f','16f')}
    lines.append(f"| {names[v]} | {delta['2f']:+.4f}ポイント | {delta['16f']:+.4f}ポイント |")
lines+=['','負の差は追加情報を渡した方が衝突操作を選びにくかったことを示す。診断場面内の小さな変化であり、完走率の因果的な改善を保証しない。','',
        '### 処理負荷・容量','','同じPC、CPU推論1スレッド、64場面×3反復。比較学習プロセスの終了後に測定。',
        '実機の等倍プレイでの総遅延測定ではなく、入力生成とactor単体推論の測定。','',
        '| 入力 | 入力生成 中央値 / p95 | actor推論 中央値 / p95 | 512判断の観測保存量 | パラメーター数 |',
        '|---|---:|---:|---:|---:|']
for v,name in names.items():
    b=bench[v]
    lines.append(f"| {name} | {b['encode_median_ms']:.2f} / {b['encode_p95_ms']:.2f} ms | {b['actor_median_ms']:.2f} / {b['actor_p95_ms']:.2f} ms | {b['rollout512_observation_mib']:.2f} MiB | {b['parameters']:,} |")
lines+=['','学習16,384判断の平均実時間：'+ '、'.join(f"{name} {summary[v]['training_seconds_mean']/60:.2f}分" for v,name in names.items())+'。',
        '学習時間は3本並列実行中の値で、純粋な単体性能比ではない。保存量は観測のみで中間活性や勾配等を含まない。',
        '①の現実装は基準エンコーダーの局所入力も一度作ってから1px版で置き換えるため、その分の重複処理を含む。','']
path=pathlib.Path('docs/spatial-input-candidates.md')
base=path.read_text(encoding='utf-8').partition('\n## 実測結果\n')[0].rstrip()
path.write_text(base+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
print(path)
