"""Sequential bounded real-game arms, then resume the unchanged-reward arm."""
import hashlib,json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parents[1]
source=json.loads(pathlib.Path(sys.argv[1]).read_text())
out=root/'artifacts/shot-reward-study-20260930';out.mkdir(exist_ok=True)
base=source['checkpoint'];report=dict(source=base,arms={})
def launch(checkpoint,game_status,name,scale=None):
 args=['pwsh','-NoProfile','-File','scripts/live-learning.ps1','rehearse','-Continuous','-MaxSteps','18000','-DualGrid','-DirectML','-NoUI','-ResumeCheckpoint',checkpoint]
 if game_status.get('game_left_open'):
  args.append('-ContinueManaged')
  if game_status.get('paused_on_exit'):args.append('-ResumePaused')
 if scale is not None:args.extend(['-ShotStudyGames','3','-ShotRewardScale',str(scale)])
 stdout=(out/f'{name}.stdout.log').open('w');stderr=(out/f'{name}.stderr.log').open('w')
 process=subprocess.Popen(args,cwd=root,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
 stdout.close();stderr.close()
 return process
game_status=source['status']
for name,scale in [('unchanged',1.),('half',.5)]:
 process=launch(base,game_status,name,scale)
 code=process.wait()
 if code:raise RuntimeError(f'{name} failed; inspect logs')
 record=json.loads((root/'.runtime/live-learning.json').read_text());folder=pathlib.Path(record['Output'])
 status=json.loads((folder/'status.json').read_text())
 assert status['stop_reason']=='shot_study_complete' and len(status['episodes'])==3
 for episode in status['episodes']:
  assert episode['reload_verified'] and episode['reload_rng_verified']
  assert hashlib.sha256((folder/episode['checkpoint']).read_bytes()).hexdigest()==episode['checkpoint_sha256']
 report['arms'][name]=dict(record=record,status=status)
 (out/'study.json').write_text(json.dumps(report,indent=2))
 print(name,record['RunId'],'complete',flush=True)
 game_status=status
a=report['arms']['unchanged'];b=report['arms']['half']
assert a['status']['initial_policy_sha256']==b['status']['initial_policy_sha256']
checkpoint=str(pathlib.Path(a['record']['Output'])/a['status']['episodes'][-1]['checkpoint'])
process=launch(checkpoint,game_status,'resumed-unchanged')
report['resumed_checkpoint']=checkpoint;report['resume_launcher_pid']=process.pid
(out/'study.json').write_text(json.dumps(report,indent=2))
print('Standard reward continuation launched',flush=True)
