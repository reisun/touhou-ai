"""Paired aggregate and same-scene policy diagnostics for completed arms."""
import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scripts.validate_spatial_inputs import scenes
from scripts.audit_cnn_layers import outputs,inspect
from touhou_ai.risk_refinements import forecast
from touhou_ai.spatial_input_candidates import CandidateEncoder,TILES
from touhou_ai.separate_clip_ppo import SeparateClipPPO

ROOT=pathlib.Path('artifacts/risk-refinements-20260929')
def main():
    torch.set_num_threads(1)
    variants=[v for v in ('control','urgency','continuous','scale') if all((ROOT/v/str(s)/'result.json').exists() for s in (7,17,27))]
    assert 'control' in variants
    data=scenes();raws=[r for r,_ in data];valid=[i for i,r in enumerate(raws) if r['stage_frame']<=584]
    records=json.loads(pathlib.Path('artifacts/spatial-inputs-20260929/predictor-fidelity.json').read_text())['scenes']
    assert [raws[i]['stage_frame'] for i in valid]==[r['frame'] for r in records]
    labels=np.array([[[r['actual'][f][y][x] for f in (0,1)] for y,x in TILES] for r in records]).reshape(-1,18)
    mixed=(labels.sum(1)>0)&(labels.sum(1)<18)
    summary={};control=None
    for variant in variants:
        rows=[json.loads((ROOT/variant/str(s)/'result.json').read_text()) for s in (7,17,27)]
        scores=np.array([[r['success'] for r in sorted(x['evaluations']['sample']['episodes'],key=lambda r:r['seed'])] for x in rows],float)
        means={mode:dict(survival=float(np.mean([x['evaluations'][mode]['survival'] for x in rows])),seconds=float(np.mean([x['evaluations'][mode]['mean_frames']/60 for x in rows]))) for mode in ('sample','greedy','frozen_sample','frozen_greedy')}
        result=dict(means=means,seed_survival=scores.mean(1).tolist())
        recent=[row for seed in (7,17,27) for row in json.loads((ROOT/variant/str(seed)/'learning.json').read_text())[-4:]]
        result['training_diagnostics']=dict(last_four_rollouts_per_seed=True,
            mean_explained_variance=float(np.mean([r['explained_variance'] for r in recent])),
            rmse_in_death_units=float(np.mean([r['rmse'] for r in recent]))/(1 if variant=='scale' else 60))
        if control is None:control=scores
        else:
            delta=scores-control;rng=np.random.default_rng(381)
            boot=[delta[rng.integers(3,size=3)][:,rng.integers(96,size=96)].mean() for _ in range(10000)]
            result.update(difference=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
        obs=[]
        for raw in raws:
            o=CandidateEncoder('action_grid').encode(raw)
            if variant in ('urgency','continuous'):o['action_grid']=forecast(raw,continuous=variant=='continuous',graded=variant=='urgency').astype(np.float16).astype(np.float32)
            obs.append(o)
        batch={k:np.stack([o[k] for o in obs]) for k in obs[0]}
        diagnoses=[]
        for seed in (7,17,27):
            m=SeparateClipPPO.load(ROOT/variant/str(seed)/'model.zip',device='cpu')
            p,v=outputs(m.policy,batch)
            other={k:(np.roll(a,len(a)//2,axis=0) if k.endswith('_grid') else a) for k,a in batch.items()}
            q,_=outputs(m.policy,other)
            zero={k:a.copy() for k,a in batch.items()};zero['action_grid'].fill(0)
            z,_=outputs(m.policy,zero)
            layer=inspect(m.policy,batch)
            row=dict(seed=seed,grid_permutation_tv=float(np.abs(p-q).sum(1).mean()/2),
                collision_probability=float((p[valid]*labels).sum(1)[mixed].mean()),
                collision_probability_swapped=float((q[valid]*labels).sum(1)[mixed].mean()),
                collision_probability_zero=float((z[valid]*labels).sum(1)[mixed].mean()),
                critic_saturation=layer['layers']['critic.mlp.3']['tanh_abs_above_099'],value_range=float(np.ptp(v)))
            diagnoses.append(row)
        result['diagnostics']=diagnoses;summary[variant]=result
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    names={'control':'採用済み②（対照）','urgency':'① 接触までの時間','continuous':'② 移動継続予測','scale':'③ 死亡罰−1'}
    lines=['## 実測結果','','完了した条件のみ掲載。各3初期値×96弾幕。','',
      '| 条件 | 初期値7 | 初期値17 | 初期値27 | 平均完走率 | 平均生存時間 |','|---|---:|---:|---:|---:|---:|']
    for variant,r in summary.items():
        rates=' | '.join(f'{v*100:.2f}%' for v in r['seed_survival'])
        lines.append(f"| {names[variant]} | {rates} | {r['means']['sample']['survival']*100:.2f}% | {r['means']['sample']['seconds']:.3f}秒 |")
    lines+=['','| 条件 | 対照との差 | 対応bootstrap95%区間 |','|---|---:|---:|']
    for variant,r in summary.items():
        if variant=='control':continue
        lo,hi=np.array(r['bootstrap95'])*100
        lines.append(f"| {names[variant]} | {r['difference']*100:+.2f}ポイント | {lo:+.2f}〜{hi:+.2f} |")
    lines+=['','初期値3組と共通評価96条件を対応させた10,000回の再標本化。探索的な区間であり、安定性の確証ではない。','',
      '| 条件 | 通常・確率 | 観測固定・確率 | 通常・最大確率 | 観測固定・最大確率 |','|---|---:|---:|---:|---:|']
    for variant,r in summary.items():
        values=' | '.join(f"{r['means'][m]['survival']*100:.2f}%" for m in ('sample','frozen_sample','greedy','frozen_greedy'))
        lines.append(f'| {names[variant]} | {values} |')
    lines+=['','### 共通場面の操作反応と価値予測','',
      '| 条件 | グリッド置換TV | 危険操作確率：通常−置換 | 通常−追加入力0 | critic最終Tanh飽和率 |',
      '|---|---:|---:|---:|---:|']
    for variant,r in summary.items():
        ds=r['diagnostics'];avg=lambda k:float(np.mean([d[k] for d in ds]))
        lines.append(f"| {names[variant]} | {avg('grid_permutation_tv'):.6f} | {(avg('collision_probability')-avg('collision_probability_swapped'))*100:+.3f}ポイント | {(avg('collision_probability')-avg('collision_probability_zero'))*100:+.3f}ポイント | {avg('critic_saturation')*100:.2f}% |")
    lines+=['','危険操作確率はシミュレーターで分岐確認した2F衝突を使用。安全/衝突操作が混在する20場面の診断。',
      '負の差は正しい入力の方が危険操作を選びにくい方向。入力置換/ゼロは分布外となり得るため、それだけで改善の因果関係を証明しない。',
      'TVは独立9＋2の積で求めた18組の分布変化量。18択出力へ変更したものではない。',
      '飽和率は共通64場面で最終Tanhの絶対値が0.99を超える割合。','']
    lines+=['### 学習終盤の価値予測','','各初期値の最後4 rolloutの平均。各方針が訪れた場面での診断で、共通場面の精度比較ではない。',
      'RMSEは死亡罰の絶対値で割り、報酬単位の違いを除く。対象は実際の将来死亡そのものではなく、学習用GAE return。','',
      '| 条件 | 説明分散 | RMSE / 死亡罰 |','|---|---:|---:|']
    for variant,r in summary.items():
        d=r['training_diagnostics']
        lines.append(f"| {names[variant]} | {d['mean_explained_variance']:.4f} | {d['rmse_in_death_units']:.4f} |")
    doc=pathlib.Path('docs/risk-refinements-20260929.md')
    header=doc.read_text(encoding='utf-8').partition('## 実測結果')[0]
    doc.write_text(header+'\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
