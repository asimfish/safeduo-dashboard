"""Compact Pages data plus full unaltered evidence in an isolated asset branch."""
from pathlib import Path
import gzip,hashlib,json,shutil,sys
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
PUBLIC=Path('/home/liyufeng/safeduo-dashboard-feasible-guard-20261005/docs')/HERE.name
ASSETS=Path('/home/liyufeng/safeduo-dashboard-fifo-assets-20261006')/HERE.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def copy_bound(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);before=sha(src);shutil.copy2(src,dest)
    assert sha(src)==sha(dest)==before
    return dict(path=str(dest.relative_to(ASSETS)),sha256=before,bytes=dest.stat().st_size)

def assets():
    assert not ASSETS.exists()
    result=load(HERE/'holdout_results.json');visual=load(HERE/'visual_verification.json')
    assert result['completed_method_windows']==768 and result['invalid_method_windows']==0
    assert visual['status']=='PASS_BOUNDED_ACTUAL_SIX_PLANE_CAMERA'
    visual_roots=load(HERE/'VISUAL_EFFECTIVE_PLAN.json')['actual_visual_roots']
    files={};failures=[]
    for row in result['rows']:
        root=Path(row['path']);rec=load(root/'first_failure_receipts.json')
        for cap in rec['receipts']:
            src=root/cap['path'];relative=str(src.relative_to(RAW));dest=ASSETS/relative
            assert sha(src)==cap['sha256']
            if relative not in files:files[relative]=copy_bound(src,dest)
            failures.append(dict(mode=row['mode'],seed=row['seed'],step=cap['step'],env_ids=cap['env_ids'],path=relative,sha256=cap['sha256']))
        src=root/'input_recipe.npz';relative=str(src.relative_to(RAW));files[relative]=copy_bound(src,ASSETS/relative)
    for g in visual['galleries']:
        for rel,digest in [(g['state'],g['state_sha256']),(g['before'],g['before_sha256']),
                           (g['after'],g['after_sha256']),*[(im['path'],im['sha256']) for im in g['images']]]:
            relative='visual/'+g['mode']+'/'+rel;src=Path(visual_roots[g['mode']])/rel;assert sha(src)==digest
            if relative not in files:files[relative]=copy_bound(src,ASSETS/relative)
    design=load(HERE/'RANDOM_EXPERIMENT_DESIGN.json')
    for r in design['rows']:
        for name in ('bank.npz','metadata.json'):
            rel=f'banks/{r["initial_seed"]}/{name}';files[rel]=copy_bound(RAW/rel,ASSETS/rel)
    for mode in ('joint_reference','motion_admission'):
        src=Path(visual_roots[mode])/'input_recipe.npz';rel='visual/'+mode+'/input_recipe.npz';files[rel]=copy_bound(src,ASSETS/rel)
    # One lossless complete result object serves downloads and interactive curves.
    for name in ('holdout_results.json','H6_PREDICTION_AUDIT.json'):
        original=(HERE/name).read_bytes();packed=gzip.compress(original,compresslevel=6,mtime=0)
        dest=ASSETS/'data'/(name+'.gz');dest.parent.mkdir(exist_ok=True);dest.write_bytes(packed)
        assert gzip.decompress(dest.read_bytes())==original
        files['data/'+name+'.gz']=dict(path='data/'+name+'.gz',sha256=sha(dest),bytes=dest.stat().st_size,
            source_uncompressed_sha256=hashlib.sha256(original).hexdigest(),codec='gzip_lossless_original_JSON_bytes')
    receipt=dict(status='PASS_UNALTERED_ASSET_COPY',files=list(files.values()),count=len(files),
          bytes=sum(x['bytes'] for x in files.values()),failures=failures,
          scope='original PNG, original own-state JSON/nativeNPZ, numericfirstfailNPZ, actualinputrecipes, qualifiedbanks; no alteredpixels or synthesizedimagery')
    (ASSETS/'asset_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    root=ASSETS.parent
    (root/'README.md').write_text('# SafeDuo frozen experiment evidence\n\nUnaltered original evidence for the FIFO motion-admission simulation experiment. The interactive panel is at https://asimfish.github.io/safeduo-dashboard/docs/'+HERE.name+'/ .\n\nThis branch stores PNG images, their own actual states, native PhysX snapshots, paired input recipes and exact first-failure causal snapshots. It does not replace numerical scoring or establish physical safety. The complete dense trajectories remain in the named NAS experiment archive; reproduction instructions are in the panel.\n')
    (HERE/'ASSET_BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('ASSETS_BOUND',receipt['count'],receipt['bytes'],flush=True)

def pages():
    publication=load(HERE/'ASSET_PUBLICATION.json');assert publication['status'].startswith('PASS')
    result=load(HERE/'holdout_results.json');visual=load(HERE/'visual_verification.json')
    independent=load(HERE/'ASTRA_FINAL_SCORE.json')
    assert independent['status'].startswith('PASS')
    base='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+publication['commit']+'/'+HERE.name+'/'
    h6=load(HERE/'H6_PREDICTION_AUDIT.json');pred=[]
    for mode in load(HERE/'RANDOM_EXPERIMENT_DESIGN.json')['modes']:
        rows=[r for r in h6['rows'] if r['mode']==mode]
        for kind in ('cv','pd'):
            joint_n=sum(r['joints'][kind]['n'] for r in rows);distance_n=sum(r['geometry'][kind]['n'] for r in rows)
            pred.append(dict(mode=mode,predictor=kind,
               joint_rms=(sum(r['joints'][kind]['rms_rad']**2*r['joints'][kind]['n'] for r in rows)/joint_n)**.5,
               joint_max=max(r['joints'][kind]['max_abs_rad'] for r in rows),
               distance_rms=(sum(r['geometry'][kind]['rms_m']**2*r['geometry'][kind]['n'] for r in rows)/distance_n)**.5,
               optimistic_negative=sum(r['geometry'][kind]['optimistic_raw_negative_row_endpoints'] for r in rows)))
    required=['REPORT.md','REPRODUCE.md','NEXT.md','RANDOM_EXPERIMENT_DESIGN.json','MODEL_FIT.json',
        'MODEL_CALIBRATION.json','PREDICTOR_RESULTS.json','PREDICTOR_SCOPE_REGISTRATION.json','DESIGN.json','SOURCE_BASELINE.json','identify_dynamics.py','bank_audit.json','command_audit.json','audit_banks.py','audit_commands.py','execute_bank.py',
        'ASTRA_FIFO_REVIEW.md','ASTRA_FIFO_REVIEW.json','ASTRA_MOTION_RUNNER_REVIEW.md','ASTRA_MOTION_RUNNER_REVIEW.json',
        'ASTRA_CAMERA_REVIEW.md','ASTRA_CAMERA_REVIEW.json','CAMERA_VISION_SELECTION.json','CAMERA_DEVICE_TRACE.json','CAMERA_FAILED_PROCESS_EXIT.json','close_camera_attempt.py','camera_parallel_v3_driver.log','visual_motion_admission.log','CAMERA_MITIGATION_REGISTRATION.json','VISUAL_EFFECTIVE_PLAN.json','VISUAL_EFFECTIVE_EXECUTION.json','visual_retry_plan.json','visual_retry_execution.json','execute_camera_retry.py','camera_launch_contract.py','test_camera_launch_contract.py','camera_launch_contract_tests.log','verify_visual_v2.py','audit_camera_state_v2.py','CAMERA_STATE_AUDIT.json','audit_camera_state.py','CAMERA_STATE_AUDIT_REGISTRATION.json','ASTRA_FINAL_REVIEW.md','ASTRA_FINAL_REVIEW.json','ASTRA_FINAL_SCORE.json','visual_verification.json',
        'ASTRA_FINAL_NUMERIC_PREFIX_REVIEW.md','ASTRA_FINAL_NUMERIC_PREFIX_REVIEW.json','ASTRA_FINAL_LP_BACKEND_REVIEW.md','ASTRA_FINAL_LP_BACKEND_REVIEW.json','ASTRA_FINAL_BRIDGE_REVIEW.md','ASTRA_FINAL_BRIDGE_REVIEW.json','ASTRA_FINAL_BRIDGE_REREVIEW.md','ASTRA_FINAL_BRIDGE_REREVIEW.json','astra_bridge_rereview_tests.py','astra_bridge_rereview_tests.log','ANALYSIS_ORIGINAL_OUTER_EXIT.json',
        'ASTRA_FINAL_SCORER_OUTER_EXIT.json','ASTRA_FINAL_INTERRUPTED_ATTEMPT1_ASTRA_FINAL_SCORE_PROGRESS.json','astra_final_score_interrupted_attempt1.log','ASTRA_FINAL_RECOVERY_PLAN.json','ASTRA_FINAL_RECOVERY_LAUNCH.json','ASTRA_FINAL_RECOVERY_EXECUTION.json','astra_score_recovery.py','astra_score_recovery_tests.py','astra_score_recovery_tests.log',
        'NUMERIC_REGISTRATION.json','NUMERIC_PREFIX_AUDIT.json','NUMERIC_LANE1_OUTER_EXIT.json','NUMERIC_LANE0_OUTER_EXIT.json','NUMERIC_BLOCK2_OUTER_EXIT.json','ALL_NUMERIC_CLOSED_READBACK.json','numeric_block2_parallel.log','numeric_lane_0.log','numeric_lane_1.log','visual_plan.json','visual_plan_v2.json','visual_plan_v3.json','CAMERA_EARLY_SCHEDULING.json','CAMERA_FOLLOW_SUPERSESSION.json','SCHEDULING_SUPERSESSION.json','VERIFICATION.json','guard_runner.py','motion_forecast.py',
        'target_forecast.py','reference_envelope.py','projection_diagnostics.py','analyze.py','dense_audit.py',
        'audit_prediction_h6.py','test_motion_forecast.py','motion_tests.log','astra_fifo_oracle.py',
        'astra_fifo_tests.py','astra_motion_runner_tests.py','astra_motion_runner_tests.log','astra_final_score.py','astra_score_core.py','astra_score_tests.py','ASTRA_FINAL_SCORING_PLAN.json','astra_score_tests.log','CAMERA_ORIGINAL_ATTEMPT_CLOSURE.json','CAMERA_MITIGATION_RESULT.json',
        'ENVIRONMENT.json','ANALYSIS_ENVIRONMENT.json','first_failure_audit.json','failure_audit.py','run_analysis.py','execute_readback_parallel.py','ANALYSIS_PARALLEL_REGISTRATION.json','ANALYSIS_CPU_SUPERSESSION.json','ANALYSIS_CPU_ORIGINAL_EXIT.json','ANALYSIS_PARALLEL_ATTEMPT1_CLOSURE.json','READBACK_LOAD_PROFILE.json','H6_PARALLEL_EXECUTION.json','ASTRA_CPU_PARALLEL_REVIEW.md','ASTRA_CPU_PARALLEL_REVIEW.json','offline_analysis_execution.json','ANALYSIS_EFFECTIVE_EXECUTION.json','ANALYSIS_FOLLOW_EXECUTION.json','finalize_analysis_effective.py','run_lp_compatible.py','LP_BACKEND_REGISTRATION.json','LP_COMPATIBLE_EXECUTION.json','LP_COMPATIBLE_PROCESS_EXIT.json','BRIDGE_PROVENANCE_CORRECTION.json','LP_SYSTEM_COMPATIBILITY_CHECK.log','LP_SIMENV_COMPATIBILITY_CHECK.log','ANALYSIS_REGISTRATION.json','visual_runner.py','verify_visual.py','register_numeric.py','execute_numeric.py','fresh_bank.py',
        'register_visual.py','execute_visual.py','execute_visual_parallel.py','execute_visual_parallel_v3.py','build_delivery.py','panel_template.html','check_panel.py','check_legacy.py','check_static.py','verify_closed.py','verify_remote.py','verify_asset_publication.py','verify_final_delivery.py','finalize_verification.py','seal_evidence.py','update_root.py','plot_results.py','PLOT_BACKEND_SELECTION.json','comparison.png','comparison.pdf','comparison.svg']
    PUBLIC.mkdir(parents=True,exist_ok=False)
    docs=[]
    for name in required:
        src=HERE/name;assert src.is_file(),name
        shutil.copy2(src,PUBLIC/name);assert sha(src)==sha(PUBLIC/name)
        docs.append(dict(title=name,path=name))
    for name in ('holdout_results.json','H6_PREDICTION_AUDIT.json'):
        docs.insert(0,dict(title=name+' · gzip 无损下载',path=base+'data/'+name+'.gz'))
    for p in sorted((HERE/'plans').glob('*.json')):
        (PUBLIC/'plans').mkdir(exist_ok=True);shutil.copy2(p,PUBLIC/'plans'/p.name)
        docs.append(dict(title=p.name,path='plans/'+p.name))
    modes=load(HERE/'RANDOM_EXPERIMENT_DESIGN.json')['modes'];totals={r['mode']:r for r in result['totals']}
    candidate=totals['motion_admission'];reference=totals['joint_reference']
    if candidate['violations']<reference['violations']:
        verdict='本轮严格违规减少；仍存在失败，安全机制尚未通过物理安全验证。'
    elif candidate['violations']==reference['violations']:
        verdict='本轮联合预测未减少严格违规；仍存在失败，安全机制尚未通过物理安全验证。'
    else:verdict='本轮联合预测增加了严格违规；候选不升级为生产安全策略。'
    dt=load(Path(result['rows'][0]['path'])/'protocol.json')['dt']
    assert all(load(Path(row['path'])/'protocol.json')['dt']==dt for row in result['rows'])
    native_duration={m:dict(strict=independent['counts'][m]['strict']['env_steps'],deep=independent['counts'][m]['deep']['env_steps']) for m in modes}
    payload=dict(dt=dt,nativeDuration=native_duration,result={k:result[k] for k in ['registered_method_windows','completed_method_windows','invalid_method_windows',
         'totals','coverage','exposure','paired']},modes=modes,
         seedBlocks={r['command_seed']:r['block'] for r in load(HERE/'RANDOM_EXPERIMENT_DESIGN.json')['rows']},
         bank=load(HERE/'bank_audit.json'),commands=load(HERE/'command_audit.json'),visual=visual,
         failures=load(HERE/'ASSET_BUILD.json')['failures'],failureDetails=load(HERE/'first_failure_audit.json')['records'],assetBase=base,docs=docs,predictionSummary=pred,
         resultDataPath='data/holdout_results.json.gz',verdict=verdict,reviewText='GPT‑6 Astra xhigh 已独立完成 FIFO 时序检查、14 项接入测试及新实验原始结果复算；父任务 9 项接入测试也已通过。独立原始计分结果和逐案例配对结果均可下载核对。')
    packed=gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),compresslevel=6,mtime=0)
    (PUBLIC/'payload.json.gz').write_bytes(packed)
    assert json.loads(gzip.decompress(packed))==json.loads(json.dumps(payload))
    shutil.copy2(HERE/'panel_template.html',PUBLIC/'index.html')
    total=sum(p.stat().st_size for p in PUBLIC.parent.rglob('*') if p.is_file())
    assert total<1_000_000_000,('Pages existing+new bytes above explicit budget',total)
    (HERE/'PANEL_BUILD.json').write_text(json.dumps(dict(status='PASS_BOUND_COMPACT_PAGES_BUILD',page=sha(PUBLIC/'index.html'),payload=sha(PUBLIC/'payload.json.gz'),
       cases=len(result['cases']),windows=result['completed_method_windows'],images=visual['images'],public_added_bytes=sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file()),
       total_pages_bytes=total,asset_commit=publication['commit'],asset_bytes=publication['bytes']),indent=2)+'\n')
    print('PAGES_BOUND',total,'bytes total',flush=True)
if __name__=='__main__':assets() if sys.argv[1]=='assets' else pages()
