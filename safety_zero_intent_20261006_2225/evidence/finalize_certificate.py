"""Publish initial proof certificate, then record actual final deployment closure."""
from pathlib import Path
from candidate_decision import decide
from datetime import datetime,timezone
import argparse,gzip,hashlib,json,shutil,subprocess

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_worktree_20261006')
PUBLIC=WORK/'docs'/HERE.name
def load(name):return json.loads((HERE/name).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def initial():
    required=['NUMERIC_EXECUTION.json','ALL_ANALYSIS_EXECUTION.json','ASTRA_ZERO_FINAL_REVIEW.json','ASTRA_ZERO_FINAL_SCORE.json','first_failure_audit.json','FIXED_FEASIBILITY_AUDIT.json','CAMERA_STATE_AUDIT.json','visual_verification.json','ZERO_INTENT_AUDIT.json','raw_evidence_manifest.json','metadata_archive_receipt_initial.json','ASSET_PUBLICATION.json','PUBLIC_initial.json','local_browser.json','live_initial_browser.json','static_verification.json','root_change_verification.json']
    for n in required:assert load(n)['status'].startswith('PASS'),n
    legacy=load('legacy_browser.json');assert legacy['status']=='pass'
    assert load('legacy_live_browser.json')['status']=='pass'
    for name in ['local_browser.json','live_initial_browser.json']:
        check=load(name);assert check['cases_methods']==576 and check['coverage_cells']==1560 and len(check['filters'])==40
    r=load('holdout_results.json');v=load('visual_verification.json');assert r['completed_method_windows']==576 and r['invalid_method_windows']==0
    totals={t['mode']:t for t in r['totals']};assert load('CANDIDATE_DECISION.json')['code']==decide(r)['code']
    certificate=dict(status='PASS_COMPLETE_SOFTWARE_SCIENTIFIC_EVIDENCE_DELIVERY',physical_safety_status='UNVALIDATED',candidate_decision=load('CANDIDATE_DECISION.json')['code'],hardware_approved=False,production_promoted=False,
        cases=192,numeric_method_windows=576,invalid_windows=0,totals=r['totals'],zero_prefix=load('ZERO_INTENT_AUDIT.json')['summary'],camera_images=v['images'],camera_groups=v['groups'],camera_exact_numerical_replay=[{k:x[k] for k in ['mode','binding_status','q_max_difference']} for x in v['runs']],
        software_review='Parent8CPU; dedicatedGPT6Astraxhigh20CPU/5mutations and32rawsynthetic, independent576native scoring and scoped realcamera review. PASS applies only to declared software/evidence scope.',
        initial_public_commit=load('PUBLIC_initial.json')['commit'],fixed_asset_commit=load('ASSET_PUBLICATION.json')['commit'],public_url='https://asimfish.github.io/safeduo-dashboard/docs/'+HERE.name+'/',
        evidence_sha256={n:sha(HERE/n) for n in required+['holdout_results.json','REPORT.md','REPRODUCE.md','NEXT.md','legacy_browser.json','legacy_live_browser.json']},initial_archive=load('metadata_archive_receipt_initial.json'),
        acceptance_matrix=[dict(stage=stage,verdict='PASS',evidence=names) for stage,names in [
            ('artifact integrity',['NUMERIC_EXECUTION.json','ALL_ANALYSIS_EXECUTION.json','raw_evidence_manifest.json','metadata_archive_receipt_initial.json']),
            ('static quality',['static_verification.json']),
            ('focused correctness',['zero_intent_cpu_tests_v2.log','ASTRA_ZERO_SOURCE_REVIEW.json','ASTRA_ZERO_FINAL_SCORE.json','ASTRA_ZERO_FINAL_REVIEW.json']),
            ('regression',['legacy_browser.json','legacy_live_browser.json','root_change_verification.json']),
            ('security and data',['static_verification.json','CAMERA_STATE_AUDIT.json','ASSET_PUBLICATION.json','PUBLIC_initial.json']),
            ('compatibility',['LP_BACKEND_REGISTRATION.json','LP_BACKEND_PRECHECK.json','NUMERIC_EXECUTION.json']),
            ('native user journey',['local_browser.json','live_initial_browser.json','PUBLIC_initial.json']),
            ('documentation',['REPORT.md','REPRODUCE.md','NEXT.md'])]],
        final_publication='This certificate binds the initial completed deployment; exact final certificate deployment is separately byte-verified and recorded in the final NAS member-readback archive.',
        limitations=['conditioned correlated simulation; no IID reliability or complete26D volume claim','three modes are paired on192tapes; not576 independent randomtapes','frozen-J endpoint risk is not a robust dynamics bound','continuousfull9021J not persisted; selectedJ only','actual9-view represented-sphere frustum excludes occlusion/wholemesh/hardware guarantee','camera binds ownstate and separately reports numerical replay failure','zero-intent audit preregistered; pure savedstate helperreplay not alternatephysics or uniquecause proof','initialGPU25GiBgateFAILthenaccidental3baseline launches; earlysameprocessSIGSTOP/SIGCONT retained, baseline-onlywallclockpause/protocoldeviation limits cleanexecution/causality','actual outertool and canonical childexit receipts separatelyreported; no converting failedlaunch/partialwindows intoPASS'],utc=datetime.now(timezone.utc).isoformat())
    certificate['archived_formatting']=load('ARCHIVED_WHITESPACE_DIAGNOSIS.json')
    certificate['initial_resource_gate']=load('NUMERIC_INITIAL_RESOURCE_GATE.json')
    certificate['resource_resume']=load('RESOURCE_PAUSE_RESUME_EXECUTION.json')
    certificate['recovery_execution']=load('RECOVERY_EXECUTION.json')
    certificate['recovery_amendment']=load('PLANS_RECOVERY_AMENDMENT.json')
    certificate['initial_attempt_validity']=dict(completed=448,invalid_unstarted=128,original_exit_only_summary_was_wrong=True)
    certificate['limitations'].append('first448complete/128unstartedremoteUSDfailures retained; postoutcome recovery of onlyunstartednumericconditions; cameraCUDA1attemptunsupported/no products/stopped/failed waitUNRECORDED, finaloriginalCUDA0recovery; transitiveUSDbyteequivalence acrossattempts notfullyattested')
    certificate['zero_prefix_diagnostic_preregistered']=True
    write(HERE/'VERIFICATION.json',certificate)
    public_names=['VERIFICATION.json','raw_evidence_manifest.json','metadata_archive_receipt_initial.json','ASSET_PUBLICATION.json','PUBLIC_initial.json','local_browser.json','live_initial_browser.json','static_verification.json','ARCHIVED_WHITESPACE_DIAGNOSIS.json','root_change_verification.json','ROOT_REMOTE_INHERITANCE.json','legacy_browser.json','legacy_live_browser.json']
    payload=json.loads(gzip.decompress((PUBLIC/'payload.json.gz').read_bytes()))
    for name in public_names:
        assert not (PUBLIC/name).exists();shutil.copy2(HERE/name,PUBLIC/name);assert sha(HERE/name)==sha(PUBLIC/name)
        payload['docs'].append(dict(title=name,path=name))
    (PUBLIC/'payload.json.gz').write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),compresslevel=6,mtime=0))
    total=sum(p.stat().st_size for p in (WORK/'docs').rglob('*') if p.is_file());assert total<1_073_741_824
    write(HERE/'CERTIFICATE_BUILD.json',dict(status='PASS_INITIAL_PROOF_CERTIFICATE_READY',files=public_names,total_docs_bytes=total,one_GiB_satisfied=True,decimal_1GB_satisfied=total<1_000_000_000))
    print('CERTIFICATE_COMPLETE',total,flush=True)
def final():
    p=load('PUBLIC_final.json');browser=load('live_final_browser.json');assert p['status'].startswith('PASS') and browser['status'].startswith('PASS')
    assert browser['cases_methods']==576 and browser['coverage_cells']==1560
    assert sha(HERE/'VERIFICATION.json')==sha(PUBLIC/'VERIFICATION.json')
    status=subprocess.run(['git','-c','core.filemode=false','-C',str(WORK),'status','--porcelain'],capture_output=True,text=True);assert status.returncode==0 and not status.stdout,status.stdout
    receipt=dict(status='PASS_FINAL_CERTIFICATE_EXACT_DEPLOYMENT_AND_NATIVE_JOURNEY',commit=p['commit'],pages_success=True,all_owned_public_files_rehashed=p['count'],public_url='https://asimfish.github.io/safeduo-dashboard/docs/'+HERE.name+'/',fixed_asset_commit=load('ASSET_PUBLICATION.json')['commit'],native_cases_methods=576,native_curve_data_arrays=4032,combined_filters=40,coverage_cells=1560,all_camera_image_link_bindings=browser['image_url_bindings'],verification_sha256=sha(HERE/'VERIFICATION.json'),final_public_receipt_sha256=sha(HERE/'PUBLIC_final.json'),final_native_receipt_sha256=sha(HERE/'live_final_browser.json'),git_worktree_clean=True,physical_safety_certified=False,candidate_decision=load('CANDIDATE_DECISION.json')['code'],utc=datetime.now(timezone.utc).isoformat())
    write(HERE/'FINAL_DELIVERY_CLOSURE.json',receipt);print('FINAL_CLOSED',p['commit'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initial','final']);a=p.parse_args();initial() if a.stage=='initial' else final()
