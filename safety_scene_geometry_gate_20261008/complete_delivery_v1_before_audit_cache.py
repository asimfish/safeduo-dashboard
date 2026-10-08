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
 state('WAITING_FOR_MATCHED_AND_FIRST_FRESH_CYCLE')
 assert output(['git','rev-parse','HEAD'],main)==base_main and output(['git','rev-parse','HEAD'],archive)==base_archive
 assert not output(['git','status','--porcelain'],main) and not output(['git','status','--porcelain'],archive)
 wait_native('matched_scene',3488705)
 if not (p/'MATCHED_SCENE_FULL_RESULTS.json').exists():run([py,'analyze_scene.py','--name','matched_scene'],log='MATCHED_ANALYSIS.log')
 assert read(p/'MATCHED_SCENE_FULL_RESULTS.json')['recording_receipt_sha256']==sha(raw/'matched_scene'/'recording_receipt.json')
 run([py,'close_camera_source.py'])
 while not (raw/'fresh_scene_batch'/'progress.json').exists() or read(raw/'fresh_scene_batch'/'progress.json')['steps']<720:
  assert Path('/proc/3936298').exists() and not (raw/'fresh_scene_batch'/'failure.txt').exists();state('WAIT_FIRST_COMPLETE_FRESH_CYCLE');time.sleep(30)
 if not (p/'REGISTRATION_CAMERA_UR.json').exists():run([py,'register_ur_replay.py'],log='CAMERA_UR_REGISTRATION.log')
 for path,h in read(p/'REGISTRATION_CAMERA_UR.json')['source_sha256'].items():assert sha(path)==h,('render source drift',path)
 env=dict(os.environ,LD_PRELOAD='/home/liyufeng/miniforge3/envs/safeduo/lib/libstdc++.so.6',OMNI_KIT_ACCEPT_EULA='YES',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0,1',PYTHONPATH='/home/liyufeng/safeduo/src')
 state('RENDER_ARCHIVED_UR_STATES')
 if not (raw/'camera_replay_ur'/'recording_receipt.json').exists():
  if (raw/'camera_replay_ur').exists():
   running=[]
   for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:argv=(proc/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if str(p/'runtime_camera_ur/native_camera_replay.py').encode() in argv:running.append(proc)
   assert len(running)==1,('incomplete replay without exactly one running owner',running)
   while not (raw/'camera_replay_ur'/'recording_receipt.json').exists():
    assert running[0].exists() and not (raw/'camera_replay_ur'/'failure.txt').exists();state('WAIT_EXISTING_UR_REPLAY',pid=int(running[0].name));time.sleep(20)
  else:run([py,str(p/'runtime_camera_ur/native_camera_replay.py'),'--registration',str(p/'REGISTRATION_CAMERA_UR.json'),'--out',str(raw/'camera_replay_ur'),'--headless','--device','cuda:0','--kit_args=--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'],env=env,log='CAMERA_UR_NATIVE.log')
 assert (raw/'camera_replay_ur'/'recording_receipt.json').exists() and not (raw/'camera_replay_ur'/'failure.txt').exists()
 wait_native('fresh_scene_batch',3936298)
 state('INDEPENDENT_FULL_NATIVE_SCORING')
 run([py,'analyze_scene.py','--name','fresh_scene_batch'],log='FRESH_BATCH_ANALYSIS.log');run([py,'audit_batch_mapping.py','--name','fresh_scene_batch'],log='FRESH_BATCH_MAPPING.log');run([py,'close_ur_camera_source.py']);run([py,'characterize_workspace.py'],log='WORKSPACE_CHARACTERIZATION.log')
 run([py,'encode_videos.py'],log='ENCODE_VIDEOS.log');run([py,'build_panel_data.py'],log='BUILD_PANEL_DATA.log');run([py,'assemble_source_closure.py'],log='SOURCE_CLOSURE.log');run(['node','--check','panel/panel.js'])
 summary=read(p/'SUMMARY.json');assert summary['total_native_env_states']==276480 and summary['original_native_images']==700;assert read(p/'VIDEO_RECEIPT.json')['native_clips']==3 and read(p/'VIDEO_RECEIPT.json')['supplementary_render_clips']==3
 assert read(p/'FRESH_SCENE_BATCH_MAPPING_GATE.json')['steps']==8640
 state('ASSEMBLE_AND_VERIFY_LOCAL_DASHBOARD',scientific_candidate_pass=summary['limited_empty_hand_candidate_pass'])
 run([py,'prepare_archive.py'],log='ARCHIVE_PREPARE.log');run([py,'install_main.py','--archive-commit','LOCAL_PREFLIGHT'],log='MAIN_PREFLIGHT.log')
 browser_env=dict(os.environ,PYTHONPATH='/tmp/safeduo_dashboard_browser_tools',PLAYWRIGHT_BROWSERS_PATH='/tmp/safeduo_dashboard_browser')
 run(['/usr/bin/python3','check_browser.py'],env=browser_env,log='BROWSER_LOCAL.log')
 browser=read(p/'browser_local/receipt.json');assert browser['status']=='PASS_LOCAL_BROWSER';assert all(x['images_checked']==256 and len(x['video_checks'])==6 for x in browser['checks'])
 matrix=dict(delivery_local_verdict='PASS',scientific_verdict='PASS_LIMITED_EMPTY_HAND' if summary['limited_empty_hand_candidate_pass'] else 'REJECT_LIMITED_EMPTY_HAND',full_system0_acceptance=False,requirements=[dict(criterion=label,status='PASS',evidence=file,sha256=sha(p/file)) for label,file in [('Full fresh native reconstruction','FRESH_SCENE_BATCH_FULL_RESULTS.json'),('Matched negative controls and scene gate','MATCHED_SCENE_FULL_RESULTS.json'),('All batch identity and original native point mappings','FRESH_SCENE_BATCH_MAPPING_GATE.json'),('Recorder qualification with expected unsafe cases','BATCH_PILOT_READY_RECORDER_GATE.json'),('Six decoded native/reconstructed video clips','VIDEO_RECEIPT.json'),('Matched overhead source closure','CAMERA_SOURCE_CLOSURE.json'),('Fresh right-hand overhead source closure','CAMERA_UR_SOURCE_CLOSURE.json'),('Registered source bytes available','SOURCE_CLOSURE.json'),('Desktop/mobile local browser and native images','browser_local/receipt.json'),('Main payload and prior root preserved','MAIN_PAYLOAD_GATE.json')]],unqualified=['Full System0/four-arm manipulation','Real grasp/load/release','Full 26-dimensional operating domain','Constructor contact forces','Full USD/material/simulator dependency seal','End-to-end performance speedup'])
 (p/'LOCAL_ACCEPTANCE_MATRIX.json').write_text(json.dumps(matrix,indent=2)+'\n');run([py,'seal_archive.py'],log='ARCHIVE_SEAL.log')
 assert output(['git','rev-parse','HEAD'],archive)==base_archive;expected_changes(archive,[p.name]);run(['git','add','--',p.name],archive);run(['git','diff','--cached','--check'],archive);run(['git','diff','--cached','--stat'],archive,log='ARCHIVE_STAGED_REVIEW.log')
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
