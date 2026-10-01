import json,pathlib,numpy as np,hashlib
from scripts.compare_joint_adopted import OUT
s=json.loads((OUT/'summary.json').read_text());assert all(s[r]['n_seeds']==3 for r in ['relative','cnn'])
lines=['\n## 最終結果\n','各欄は3初期値の平均。各モデルを96弾幕で評価。','', '|入力|確率完走率 9+2→18|平均生存秒 9+2→18|最大確率完走率 9+2→18|18の観測固定・確率完走率|','|---|---:|---:|---:|---:|']
for rep in ['relative','cnn']:
 b=s[rep]['means']['independent'];a=s[rep]['means']['joint']
 lines.append(f"|{rep}|{b['sample']['survival']*100:.2f}→{a['sample']['survival']*100:.2f}%|{b['sample']['seconds']:.3f}→{a['sample']['seconds']:.3f}|{b['greedy']['survival']*100:.2f}→{a['greedy']['survival']*100:.2f}%|{a['frozen_sample']['survival']*100:.2f}%|")
lines += ['','### 初期値別の確率選択完走率','']
for rep in ['relative','cnn']:
 for p in s[rep]['pairs']:lines.append(f"- {rep} seed{p['seed']}: {p['independent']['sample']['survival']*100:.2f}→{p['joint']['sample']['survival']*100:.2f}%")
dep=json.loads((OUT/'joint-dependence.json').read_text())
lines += ['','### 結合出力の利用状況','', '過去の危険場面プローブ108場面を学習済みモデルへ入力し、18分布とその周辺分布の積を比較。','独立出力なら両者の差はゼロになる。これは出力の結合度の診断であり、未使用の最終汎化評価ではない。']
for rep in ['relative','cnn']:
 rows=[r for r in dep if r['representation']==rep];tv=np.mean([r['mean_total_variation_from_independent_marginals'] for r in rows]);mi=np.mean([r['mean_mutual_information_nats'] for r in rows])
 lines.append(f"- {rep}: 平均全変動距離{tv:.4f}、相互情報量{mi:.4f} nats。")
lines += ['','方向に応じて速度確率が異なる分布は学習できている。','ただし方向との結び付きは、弾幕を観測して適切に使い分けたことの証明ではない。','', '### 判断','', '今回の16384判断×3初期値では、数値入力は悪化、CNNは平均で小幅改善するが初期値間で不安定。','18択の優位を採用できるだけの一貫した証拠は得られず、両モデルとも独立9+2を基準として維持する。','18択一般の無効性を意味しない。CNNの候補として結果を保持するが、実機へは反映しない。','分布、CNN別勾配上限との組合せ、保存再開、評価方法を含む7テスト成功。','設定configは開始時manifestと一致。実機設定未変更。']
config=json.loads(pathlib.Path('configs/autumn-learning-reference-v2.json').read_text());physics=hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest()
for rep in ['relative','cnn']:
 for seed in [7,17,27]:
  m=json.loads((OUT/rep/str(seed)/'manifest.json').read_text());assert m['config']==config and m['physics_sha256']==physics
(OUT/'status.json').write_text(json.dumps({'status':'complete','adopted':False,'baseline_unchanged':True,'reason':'numerical regresses; CNN stochastic mean increases modestly but seed27 regresses; insufficient consistent evidence to replace independent9+2'},indent=2))
p=pathlib.Path('docs/joint-adopted-comparison.md');p.write_text(p.read_text(encoding='utf-8-sig').replace('実行中。','完了。基準の独立9+2を維持。')+'\n'.join(lines)+'\n',encoding='utf-8')
entry='\n## 2026-09-29 採用済み設定で独立9+2対結合18を再比較\n\ndocs/joint-adopted-comparison.md参照。数値は共通勾配上限、CNNは別上限。未採用の危険入力なし。\n各3初期値16384判断、同じ96弾幕評価。確率完走率: 数値12.15→7.29%、CNN13.54→15.97%。\nCNNは2初期値改善・1悪化で不安定。数値悪化。両モデルの基準は独立9+2のまま維持。\n18分布は方向・速度の結び付きを持つが、それだけで有利な回避学習へ繋がるとは確認できない。\n7テスト成功、6モデルと対照評価を保存、実機未変更。\n'
for name in ['model-learning-history.md','fixed-dodge.md']:
 with (pathlib.Path('docs')/name).open('a',encoding='utf-8') as f:f.write(entry)
print('\n'.join(lines))
