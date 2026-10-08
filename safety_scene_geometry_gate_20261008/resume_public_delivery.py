"""Authorized unattended completion; fail closed before remote mutations."""
import os,json,time,hashlib,subprocess,shutil,urllib.request
from pathlib import Path
from datetime import datetime,timezone
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008');main=Path('/home/liyufeng/safeduo-dashboard-benchmark-protocol-20261006');archive=Path('/home/liyufeng/safeduo-dashboard-object-media-20261006');py='/home/liyufeng/miniforge3/envs/safeduo/bin/python';base_main='b12ca5794151a57816a1f18eb3029c2bf60ab3c4';base_archive='eae404bb32e6fef9c43d02b2cba0e28e26cb4362';sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();read=lambda f:json.loads(Path(f).read_text());start=time.monotonic()
def state(stage,**extra):
 r=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**extra);(p/'DELIVERY_STATE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
def run(args,cwd=p,env=None,log=None):
 print('COMMAND',args,flush=True)
 if log:
  with (p/log).open('w') as f:subprocess.run(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 else:subprocess.run(args,cwd=cwd,env=env,check=True)
def output(args,cwd=p):return subprocess.check_output(args,cwd=cwd,text=True).strip()
def wait_native(name,pid):
 while not (raw/name/'recording_receipt.json').exists():
  if (raw/name/'failure.txt').exists():raise RuntimeError('Native failure: '+name)
  if not Path(f'/proc/{pid}').exists():raise RuntimeError('Native stopped without final receipt: '+name)
  if time.monotonic()-start>6*3600:raise TimeoutError('No complete native receipt within six hours: '+name)
  progress=read(raw/name/'progress.json') if (raw/name/'progress.json').exists() else {}
  state('WAIT_NATIVE',run=name,completed_steps=progress.get('steps',0),planned_steps=8640);time.sleep(30)
 assert not (raw/name/'failure.txt').exists()
def expected_changes(repo,prefixes):
 lines=output(['git','status','--porcelain','--untracked-files=all'],repo).splitlines()
 for line in lines:
  path=line[3:];assert any(path==prefix or path.startswith(prefix+'/') for prefix in prefixes),('unrelated_git_change',repo,path)
 assert lines,('no_changes',repo)
def api_json(path):
 req=urllib.request.Request('https://api.github.com/'+path,headers={'Accept':'application/vnd.github+json','User-Agent':'SafeDuo-evidence-verification'});return json.load(urllib.request.urlopen(req,timeout=30))
def check_protection():
 assert api_json('repos/asimfish/safeduo-dashboard/branches/main')['protected'] is False,'Main protected; preserve local artifacts and stop'
 assert not any(x['enforcement']=='active' for x in api_json('repos/asimfish/safeduo-dashboard/rulesets')),'Active branch rules require explicit applicability review before commit'
try:
 state('RESUME_PUBLICATION_FROM_VERIFIED_LOCAL_ARTIFACTS')
 summary=read(p/'SUMMARY.json');assert summary['total_native_env_states']==276480 and summary['original_native_images']==700
 assert read(p/'browser_local/receipt.json')['status']=='PASS_LOCAL_BROWSER'
 for criterion in read(p/'LOCAL_ACCEPTANCE_MATRIX.json')['requirements']:assert sha(p/criterion['evidence'])==criterion['sha256'],criterion
 assert read(p/'FRESH_SCENE_BATCH_FULL_RESULTS.json')['recording_receipt_sha256']==sha(raw/'fresh_scene_batch/recording_receipt.json')
 browser_env=dict(os.environ,PYTHONPATH='/tmp/safeduo_dashboard_browser_tools',PLAYWRIGHT_BROWSERS_PATH='/tmp/safeduo_dashboard_browser')
 assert output(['git','rev-parse','HEAD'],archive)==base_archive;expected_changes(archive,[p.name]);run(['git','add','--',p.name],archive);run(['git','diff','--cached','--check','--',p.name]+[':(exclude)'+x['file'] for x in read(p/'ARCHIVE_WHITESPACE_SCOPE_GATE.json')['preserved_raw_files']],archive);run(['git','diff','--cached','--stat'],archive,log='ARCHIVE_STAGED_REVIEW.log')
 # Manifest and independent source seals establish exact intended binary content.
 check_protection();run(['git','commit','-m','[evidence/test]: archive scene admission and native qualification'],archive);archsha=output(['git','rev-parse','HEAD'],archive);run(['git','push','origin','HEAD:exp/scene-geometry-media-20261008'],archive)
 state('VERIFY_PUBLIC_IMMUTABLE_ARCHIVE',archive_commit=archsha);run([py,'verify_archive.py','--commit',archsha],log='ARCHIVE_PUBLIC_VERIFY.log')
 run([py,'install_main.py','--archive-commit',archsha],log='MAIN_FINAL_INSTALL.log');expected_changes(main,['index.html','docs/'+p.name]);run(['git','fetch','origin','main'],main);assert output(['git','rev-parse','origin/main'],main)==base_main,'Remote main advanced; preserve artifact and stop for integration'
 check_protection();run(['git','add','--','index.html','docs/'+p.name],main);run(['git','diff','--cached','--check'],main);run(['git','diff','--cached','--stat'],main,log='MAIN_STAGED_REVIEW.log');run(['git','diff','--cached','--','index.html','docs/'+p.name+'/SUMMARY.json'],main,log='MAIN_CONTENT_REVIEW.log')
 run(['git','commit','-m','[dashboard/test]: publish scene admission evidence and scope'],main);commit=output(['git','rev-parse','HEAD'],main);run(['git','push','origin','HEAD:main'],main)
 state('WAIT_EXACT_PAGES_DEPLOYMENT',main_commit=commit,archive_commit=archsha)
 deadline=time.monotonic()+900
 while True:
  response=api_json(f'repos/asimfish/safeduo-dashboard/actions/runs?head_sha={commit}&per_page=10')
  runs=[x for x in response['workflow_runs'] if x['head_sha']==commit and x['name']=='pages build and deployment']
  if any(x['status']=='completed' and x['conclusion']=='success' for x in runs):break
  if any(x['status']=='completed' and x['conclusion']!='success' for x in runs):raise RuntimeError('Pages workflow failed on exact commit')
  if time.monotonic()>deadline:raise TimeoutError('Pages exact commit still pending after 15 minutes')
  time.sleep(30)
 run([py,'verify_public.py'],log='PUBLIC_MAIN_VERIFY.log');run(['/usr/bin/python3','check_browser.py','--live'],env=browser_env,log='BROWSER_PUBLIC.log')
 assert not output(['git','status','--porcelain'],main) and not output(['git','status','--porcelain'],archive)
 final=dict(status='PASS_PUBLIC_DELIVERY',scientific_candidate_pass=summary['limited_empty_hand_candidate_pass'],full_system0_acceptance=False,main_commit=commit,archive_commit=archsha,url='https://asimfish.github.io/safeduo-dashboard/docs/'+p.name+'/',root_url='https://asimfish.github.io/safeduo-dashboard/#sceneGeometryEvidence',native_environment_states=276480,original_complete_run_images=700,video_clips=6,gates={file:sha(p/file) for file in ['LOCAL_ACCEPTANCE_MATRIX.json','ARCHIVE_DOWNLOAD.json','PUBLIC_DELIVERY.json','browser_public/receipt.json']},completed_utc=datetime.now(timezone.utc).isoformat())
 (p/'FINAL_DELIVERY_GATE.json').write_text(json.dumps(final,indent=2)+'\n');state('COMPLETE',**final)
except BaseException as e:
 import traceback
 state('STOPPED_NEEDS_DIAGNOSIS',error=str(e),traceback=traceback.format_exc(),no_failed_run_promoted=True);raise
