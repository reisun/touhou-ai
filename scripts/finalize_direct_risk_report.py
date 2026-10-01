import json,pathlib,numpy as np
out=pathlib.Path('artifacts/autumn-direct-risk-20260928')
s=json.loads((out/'summary.json').read_text())
assert all(s[r]['seed_count']==3 for r in ['relative','cnn'])
lines=['\n## 最終結果（3初期値×各96弾幕）\n','|入力|確率選択の完走率 基準→追加|生存秒 基準→追加|危険入力ゼロの完走率|観測固定の完走率|','|---|---:|---:|---:|---:|']
for rep in ['relative','cnn']:
 r=s[rep];b=r['means']['before']['sample'];a=r['means']['after']['sample'];z=r['means']['after']['zero_risk_sample'];f=r['means']['after']['frozen_sample']
 lines.append(f"|{rep}|{b['survival']*100:.2f}→{a['survival']*100:.2f}%|{b['seconds']:.3f}→{a['seconds']:.3f}|{z['survival']*100:.2f}%|{f['survival']*100:.2f}%|")
 lines.append('') if False else None
 (out/rep/'status.json').write_text(json.dumps({'status':'complete','adopted':False,'reason':'no stable stochastic survival/time improvement across seeds; risk-removal controls show no consistent benefit'}))
lines+=['','CNNのseed別確率完走率（基準→追加）:']
for p in s['cnn']['pairs']:lines.append(f"- seed{p['seed']}: {p['before']['sample']['survival']*100:.2f}→{p['after']['sample']['survival']*100:.2f}%")
a=s['cnn']['means']['after'];pm=s['cnn']['probe_danger_mass']
lines += ['',f"CNN最大確率選択の平均完走率は{a['greedy']['survival']*100:.2f}%だが、全3初期値で観測固定と一致。",f"CNN危険場面プローブの危険操作確率質量:通常{pm['normal']*100:.2f}%、ゼロ{pm['zero']*100:.2f}%、入替{pm['shuffled']*100:.2f}%。",'','両入力とも未採用。短期危険度を直接渡しても、今回の学習予算では安定した回避改善は得られなかった。','入力の認識だけでなく、死亡罰からその情報を操作へ反映する学習過程に課題が残ることを示す。','ただし入力経路、予測近似、学習量にも依存するため原因を一つには限定しない。','操作分布の変化量診断は input-influence.json、数値の追加最大確率対照は relative/zero-risk-greedy.json。','実装と既存勾配上限の計6テスト成功。基準configと実験開始manifestの一致を確認。実機未変更。']
p=pathlib.Path('docs/direct-risk-comparison.md');text=p.read_text(encoding='utf-8-sig').replace('実行中。','完了。両入力とも未採用。');p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
entry='\n## 2026-09-29 短期危険度の操作入力への直接追加\n\ndocs/direct-risk-comparison.md参照。移動9+速度2独立出力を維持し、18危険指標をactor最終層へ追加。\n両入力3seed×16384判断・各96弾幕評価。確率完走率:数値12.15→7.29%、CNN13.54→14.58%。\n数値は悪化、CNNは不安定で平均生存も悪化。推論時の危険入力除去で成績が下がる傾向もなく両方未採用。\n数値最大確率選択は改善するが危険入力除去でも維持。CNN最大確率選択は全3seedで観測固定と一致。\n入力認識を計算で助けても、死亡罰から操作へ反映する過程の改善は不十分。6テスト成功、標準設定維持、実機未変更。\n'
for name in ['model-learning-history.md','fixed-dodge.md']:
 p=pathlib.Path('docs')/name
 with p.open('a',encoding='utf-8') as f:f.write(entry)
config=json.loads(pathlib.Path('configs/autumn-learning-reference-v2.json').read_text())
assert all(json.loads((out/r/'manifest.json').read_text(encoding='utf-8'))['baseline_config']==config for r in ['relative','cnn'])
print('\n'.join(lines))
