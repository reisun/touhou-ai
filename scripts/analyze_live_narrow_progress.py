"""Read-only comparison of completed live games; never loads or controls the game."""
import json,pathlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path('artifacts');OUT=ROOT/'live-narrow-progress-535'
RUNS={'current':'live-learning-20260929-071910-9a4a78','previous':'live-learning-20260928-124148-6c9ab8'}
OUT.mkdir(exist_ok=True)
data={}
for key,name in RUNS.items():
    source=ROOT/name/'status.json'
    snapshot=OUT/(key+'-status.json')
    raw=(snapshot if snapshot.exists() else source).read_text(encoding='utf-8');report=json.loads(raw)
    if not snapshot.exists():snapshot.write_text(raw,encoding='utf-8')
    cap=535 if key=='current' else 918
    episodes=[dict(e) for e in report['episodes'] if e['episode']<=cap]
    assert len(episodes)==cap
    initial_counts=collections.Counter()
    for e in episodes:
        initial=json.loads((ROOT/name/f"initial-state-{e['episode']}.json").read_text(encoding='utf-8'))
        initial_counts[tuple(initial[k] for k in ('difficulty','character','shot','lives_raw','stage','stage_frame'))]+=1
        e['seconds']=(e['final_frame']-initial['stage_frame'])/60
        e['rank']=(e.get('max_progress') or {}).get('rank',0)
    data[key]=episodes
    assert set(initial_counts)=={(1,0,1,2,1,1)}
    print(key,'initial conditions',dict(initial_counts))

def describe(rows):
    seconds=np.array([e['seconds'] for e in rows])
    return dict(first=rows[0]['episode'],last=rows[-1]['episode'],n=len(rows),
        mean_seconds=float(seconds.mean()),median_seconds=float(np.median(seconds)),
        p10_seconds=float(np.quantile(seconds,.1)),p90_seconds=float(np.quantile(seconds,.9)),max_seconds=float(seconds.max()),
        midboss_or_later=sum(e['rank']>=1 for e in rows),boss_or_later=sum(e['rank']>=3 for e in rows),
        truncated_or_recovery=sum(e['truncated'] or e['recovery_update'] for e in rows),
        hit_counts=dict(collections.Counter(e['hits'] for e in rows)),
        kl_stops=sum(e['optimization'].get('kl_early_stopped',False) for e in rows),
        end_training_steps=rows[-1]['total_steps'])

limit=data['current'][-1]['total_steps']
windows={'current_last100':data['current'][-100:],
         'previous_same_games':data['previous'][435:535],
         'previous_same_steps':[e for e in data['previous'] if e['total_steps']<=limit][-100:],
         'previous_final100':data['previous'][-100:]}
summary={k:describe(v) for k,v in windows.items()}
blocks={k:[describe([e for e in rows if lo<=e['episode']<lo+100]) for lo in range(1,len(rows)+1,100)] for k,rows in data.items()}
result=dict(runs=RUNS,current_cap=535,previous_cap=918,windows=summary,blocks=blocks,
    same_steps_excluding_recovery=describe([e for e in windows['previous_same_steps'] if not(e['truncated'] or e['recovery_update'])]),
    caveats=['Training trajectories, not frozen-policy paired evaluations.',
             'Time means start to game over across all three lives, including invulnerability; not first-hit survival.',
             'One run per configuration; small CNN and separate gradient limits changed together.',
             'Current live input does not include adopted offline action_grid.'])
(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
for key,rows in data.items():
    x=np.array([e['episode'] for e in rows]);seconds=np.array([e['seconds'] for e in rows]);boss=np.array([e['rank']>=3 for e in rows])
    label='Current: narrow CNN + separate clipping' if key=='current' else 'Previous: larger CNN + shared clipping'
    axes[0].plot(x[99:],np.convolve(seconds,np.ones(100)/100,mode='valid'),label=label)
    axes[1].plot(x[99:],100*np.convolve(boss,np.ones(100)/100,mode='valid'),label=label)
axes[0].set(ylabel='Game-over time (game seconds)',title='Live evasion training: trailing 100 completed games')
axes[1].set(xlabel='Completed games / updates',ylabel='Reached stage 1 boss (%)')
for ax in axes:ax.grid(alpha=.25);ax.axvline(535,color='gray',ls=':',lw=1)
axes[0].legend();fig.savefig(OUT/'comparison.png',dpi=150)
lines=['# 実機回避学習：535更新時点の比較','','現在：小型CNN＋操作/価値の別勾配上限。前回：大型CNN＋共通勾配上限。',
       '両方ともNormal/霊夢B、初期残機2（自機込み3機）、射撃・ボム無効、死亡罰−60。',
       '現在の実機に②の操作別予測入力は未導入。学習やゲームには操作していない。','',
       '| 比較窓 | ゲーム番号 | 平均ゲーム内時間 | 中央値 | 中ボス以降 | 1面ボス到達 |',
       '|---|---|---:|---:|---:|---:|']
names={'current_last100':'現在・直近100','previous_same_games':'前回・同じ回数','previous_same_steps':'前回・同程度の判断数','previous_final100':'前回・最終100'}
for key,s in summary.items():lines.append(f"| {names[key]} | {s['first']}〜{s['last']} | {s['mean_seconds']:.2f}秒 | {s['median_seconds']:.2f}秒 | {s['midboss_or_later']}% | {s['boss_or_later']}% |")
lines+=['','時間は各ゲームの開始フレームからゲームオーバーまでを60F/秒で換算。初被弾までの時間ではない。',
        '比較窓はすべて100ゲーム、各ゲーム3被弾。同判断数の旧窓にゲームオーバー後の復旧更新が1回含まれる。',
        'その1回を除いても旧窓は平均60.53秒・ボス到達8/99（8.08%）で、比較の方向は変わらない。他の3窓に途中打ち切り・復旧更新はない。',
        f"現在535更新は{limit:,}判断。前回535更新は{data['previous'][534]['total_steps']:,}判断。長く生き残るほど1更新の学習量が増えるため、判断数を揃えた窓も併記。",'',
        '## 推移','','| ゲーム区間 | 前回・平均時間 | 現在・平均時間 |', '|---|---:|---:|']
for lo,hi in [(1,100),(101,200),(201,300),(301,400),(401,500),(486,535)]:
    a=describe([e for e in data['previous'] if lo<=e['episode']<=hi]);b=describe([e for e in data['current'] if lo<=e['episode']<=hi])
    lines.append(f"| {lo}〜{hi} | {a['mean_seconds']:.2f}秒 | {b['mean_seconds']:.2f}秒 |")
lines+=['','![100ゲーム移動平均](../artifacts/live-narrow-progress-535/comparison.png)','',
        '## 評価','',
        '同じ回数でも、同程度の累積判断数でも、現在の方が平均ゲーム時間とボス到達率が良い。前回より速く改善した実績がある。',
        'ただし前回918更新時点の最終100ゲームには届かない。現在は300回台以降、平均66〜68秒付近で伸びが小さい。',
        '直近100ではKL上限による更新早期終了が86回（1epoch25回、2epoch61回、3epoch14回）。更新停止ではなく、方針変化が大きい更新を途中で止める機構が働いている。横ばいの原因をこれだけに限定しない。',
        '学習中の方針で行った実績比較であり、固定した保存モデルを同一条件で再評価した比較ではない。各構成1回の学習で、CNN小型化と別勾配上限の効果も分離していない。','']
pathlib.Path('docs/live-narrow-progress-535.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,indent=2))
