import json,pathlib,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.replicate_cnn_output_stability import OUT
s=json.loads((OUT/'summary.json').read_text());p=s['primary_fresh'];b=p['two_axis_bootstrap'];ratio=p['observed_sd_ratio']
if s['stability_criterion_passed']:verdict='追加7初期値では、18択のばらつき縮小を支持する結果。'
elif ratio<1:verdict='追加7初期値でも縮小傾向はあるが、不確かさを含めて安定性向上が確認できたとは言えない。'
else:verdict='追加7初期値では、18択のばらつき縮小は再現しなかった。'
lines=['\n## 最終結果\n',verdict,'','### 主評価: 追加7初期値・各192回','', '|指標|独立9+2|結合18|','|---|---:|---:|']
a=p['independent'];j=p['joint']
for name,key in [('平均完走率','mean'),('初期値間の標準偏差','sd'),('中央値','median'),('四分位範囲','iqr')]:
 unit='pt' if key in ['sd','iqr'] else '%'
 lines.append(f"|{name}|{a[key]*100:.2f}{unit}|{j[key]*100:.2f}{unit}|")
lines += [f"|最低〜最高|{a['min']*100:.2f}〜{a['max']*100:.2f}%|{j['min']*100:.2f}〜{j['max']*100:.2f}%|",f"|完走率10%未満|{a['below_10_percent']}/7|{j['below_10_percent']}/7|",'',f"標準偏差比（18 / 9+2）: {ratio:.3f}。二軸bootstrap95%区間: {b['sd_ratio_joint_over_independent_95'][0]:.3f}〜{b['sd_ratio_joint_over_independent_95'][1]:.3f}。",f"平均完走率差（18−9+2）: {p['mean_difference']*100:+.2f}ポイント。95%区間: {b['mean_difference_95'][0]*100:+.2f}〜{b['mean_difference_95'][1]*100:+.2f}ポイント。",'', 'この区間は7組という小標本からの近似。学習seedと評価ケースの二軸を対応させて20000回再標本化した。','1未満なら標準偏差縮小、1超なら拡大。平均が低下して低い成績へ揃うケースは別に評価する。','', '### 発見に使った3組と全10組（補助評価）','', '|集計|9+2 平均/標準偏差|18 平均/標準偏差|','|---|---:|---:|']
for title,key in [('当初3組','exploratory_original'),('全10組','secondary_all')]:
 r=s[key];lines.append(f"|{title}|{r['independent']['mean']*100:.2f}% / {r['independent']['sd']*100:.2f}pt|{r['joint']['mean']*100:.2f}% / {r['joint']['sd']*100:.2f}pt|")
lines += ['','### 追加7組の初期値別結果','', '|初期値|9+2 完走率|18 完走率|','|---|---:|---:|']
for r in p['pairs']:lines.append(f"|{r['seed']}|{r['independent']['survival']*100:.2f}%|{r['joint']['survival']*100:.2f}%|")
lines += ['','### 判断の範囲','', '対象は、この弾幕・学習量における初期値間の安定性。学習途中の成績の揺れや実機の安定性を示すものではない。','前半96回・後半96回、学習seedだけを再標本化した区間、生存時間の結果もsummary.jsonへ保存。','実機設定は変更していない。評価実装は既存の両方式それぞれ8回と完全一致を確認。','初期値対応bootstrapは同一入力と既知の倍率で動作確認。','', '![初期値別の比較](../artifacts/cnn-output-stability-20260929/stability.png)']
from scripts.plot_cnn_output_stability import make_plot
make_plot(s)
path=pathlib.Path('docs/cnn-output-stability.md');path.write_text(path.read_text(encoding='utf-8-sig').replace('実行中。','完了。')+'\n'.join(lines)+'\n',encoding='utf-8')
manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['config']==json.loads(pathlib.Path('configs/autumn-learning-reference-v2.json').read_text());assert manifest['physics_sha256']==hashlib.sha256(pathlib.Path('touhou_ai/autumn_sky.py').read_bytes()).hexdigest()
status={'status':'complete','stability_supported':s['stability_criterion_passed'],'mean_noninferiority_supported':s['mean_noninferiority_criterion_passed'],'adoption_criterion_passed':s['adoption_criterion_passed'],'baseline_changed':False}
(OUT/'status.json').write_text(json.dumps(status,indent=2))
entry=f"\n## 2026-09-29 小型CNNの9+2対18・安定性追試\n\ndocs/cnn-output-stability.md参照。新規7初期値×両方式、16384判断、各192弾幕。旧3組も評価を192へ拡張。\n{verdict}\n追加分の標準偏差:9+2 {a['sd']*100:.2f}pt、18 {j['sd']*100:.2f}pt。比{ratio:.3f}、95%区間{b['sd_ratio_joint_over_independent_95'][0]:.3f}〜{b['sd_ratio_joint_over_independent_95'][1]:.3f}。\n既存設定・実機設定未変更。モデル14個と全20モデル分の評価を保存。\n"
for name in ['model-learning-history.md','fixed-dodge.md']:
 with (pathlib.Path('docs')/name).open('a',encoding='utf-8') as f:f.write(entry)
print(verdict);print(json.dumps(status))
