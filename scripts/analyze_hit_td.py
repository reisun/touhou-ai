import pathlib,json,statistics
root=pathlib.Path('artifacts');report={}
sources={'older':root/'boss-reward-tanh-20260930/capture.json','current':root/'hit-td-20261001/capture.json'}
for name,path in sources.items():
    if not path.exists():continue
    data=json.loads(path.read_text());rows=data['rows'] if isinstance(data,dict) else data
    rows=sorted(rows,key=lambda r:(str(r['episode']),r['time']))
    hits=[];normal=[]
    for i,r in enumerate(rows[:-1]):
        n=rows[i+1]
        if r['terminal'] or n['episode']!=r['episode'] or not 0<n['time']-r['time']<.2:continue
        change=r['gamma']*n['value']-r['value'];td=r['reward']+change
        if r['components'].get('hit',0)<0:
            pre=[a for a in rows[max(0,i-180):i] if a['episode']==r['episode'] and r['time']-a['time']<=2]
            hits.append(dict(time=r['time'],episode=r['episode'],value=r['value'],next_value=n['value'],reward=r['reward'],hit_reward=r['components']['hit'],td=td,value_change=change,damage_reward_last2s=sum(a['components'].get('damage',0) for a in pre),preceding_rows=len(pre)))
        else:normal.append(td)
    report[name]=dict(rows=len(rows),nonterminal_hits=hits,normal_td_mean=statistics.mean(normal) if normal else None,
                      normal_td_min=min(normal) if normal else None)
print(json.dumps(report,indent=2))
dest=root/'hit-td-20261001';dest.mkdir(exist_ok=True)
(dest/'analysis.json').write_text(json.dumps(report,indent=2))
