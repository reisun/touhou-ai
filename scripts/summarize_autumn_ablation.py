import sys,pathlib,json
out=pathlib.Path(sys.argv[1]);rows=json.loads((out/'results.json').read_text())
lines=['# オータムスカイ比較結果','', '| 入力 | 速度 | seed | 確率選択 生存率 | 平均秒 | 最大確率 生存率 | 観測固定・確率選択 | 観測固定・最大確率 |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 e=r['evaluations'];lines.append(f"| {r['representation']} | {r['speed']} | {r['seed']} | {e['sample']['survival']:.1%} | {e['sample']['mean_frames']/60:.2f} | {e['greedy']['survival']:.1%} | {e['frozen_sample']['survival']:.1%} | {e['frozen_greedy']['survival']:.1%} |")
for speed in ['switch','fast']:
 lines+=['',f'## {speed} 条件の座標比較（学習seed単位）','']
 for seed in [7,17,27]:
  a=next((r for r in rows if r['key']==f'absolute-{speed}-{seed}'),None);b=next((r for r in rows if r['key']==f'relative-{speed}-{seed}'),None)
  if not a or not b:continue
  x=a['evaluations'];y=b['evaluations']
  lines.append(f"- seed {seed}: 相対−絶対、確率選択の生存率差 {(y['sample']['survival']-x['sample']['survival'])*100:+.1f}ポイント、平均生存差 {(y['sample']['mean_frames']-x['sample']['mean_frames'])/60:+.2f}秒。最大確率の生存率差 {(y['greedy']['survival']-x['greedy']['survival'])*100:+.1f}ポイント。")
 for rep in ['absolute','relative']:
  selected=[r for r in rows if r['representation']==rep and r['speed']==speed]
  for mode in ['sample','greedy','frozen_sample','frozen_greedy']:
   values=[r['evaluations'][mode] for r in selected]
   if values:lines.append(f"- {rep} {mode}: {len(values)}学習seed平均、生存率 {sum(v['survival'] for v in values)/len(values):.1%}、平均生存 {sum(v['mean_frames'] for v in values)/len(values)/60:.2f}秒。")
lines+=['','各24評価、1回差は4.17ポイント。CNN/低速固定はseed7の予備比較。数値の座標切替/高速固定は3学習seed。','本番全弾幕や長期学習の優劣は未判定。','']
s='\n'.join(lines)
(out/'summary.md').write_text(s,encoding='utf-8');print(s)
