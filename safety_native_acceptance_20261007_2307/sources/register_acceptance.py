"""Freeze finite holdout protocol once known-data evidence and peer CPU checks close."""
from pathlib import Path
import json, hashlib, datetime, copy, ast
H = Path(__file__).resolve().parent
R = Path('/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307')
P = Path('/home/liyufeng/safeduo')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()
def j(p):
    return json.loads(Path(p).read_text())
if __name__ == '__main__':
    assert not (H/'ACCEPTANCE_REGISTRATION.json').exists()
    development = j(H/'native_development06_fullview_plan.json')
    assert j(H/'native_development06_fullview_execution.json')['status']=='complete'
    assert j(H/'native_development06_fullview_supervisor_execution.json')['actual_exit']==0
    assert j(H/'DEVELOPMENT_AUDIT_MATRIX_RESULT.json')['status']=='complete'
    assert j(H/'development_audit_matrix_execution.json')['actual_exit']==0
    assert j(H/'bank_qualification_execution.json')['status']=='complete'
    assert j(H/'bank_qualification_supervisor_execution.json')['actual_exit']==0
    peer = j(H/'ASTRA_ACCEPTANCE_DECISION_TEST_RECEIPT.json')
    assert peer['status']=='PASS_CPU_ONLY' and peer['last_actual_exit']==0
    assert peer['attempts'][-1]['checker_sha256']==sha(H/'astra_acceptance_decision.py')
    assert 'hand_camera_binding' in (H/'astra_acceptance_decision.py').read_text(), 'supplemental hand-view evidence not integrated'
    for root in ['native_smoke_06_fullview','adaptive_development_06_fullview']:
        for kind in ['native','geometry','contacts','opening','hands']:
            assert j(H/(root+'_'+kind+'.json'))['status'].startswith('PASS_')
        assert j(H/('ASTRA_NATIVE_REAL_'+root+'.json'))['status']=='PASS_RAW_BINDING_AND_ORIGINAL_SCORING_ONLY'
    sources = dict(development['sources'])
    for p,d in sources.items():
        assert sha(p)==d, 'development-frozen source changed '+p
    extra = ['ACCEPTANCE_CRITERIA.json','PAIRING_REGISTRATION.json','INITIAL_STATE_ACCEPTANCE_REGISTRATION.json',
             'CAMERA_COVERAGE_REGISTRATION.json','audit_geometry_v2.py','audit_native.py','audit_contacts.py',
             'audit_opening.py','audit_hand_views.py','astra_native_audit.py','astra_acceptance_decision.py',
             'analyse_acceptance.py','run_step.py','run_audit_matrix.py','monitor_acceptance_audits.py','register_acceptance.py','build_coverage.py',
             'fresh_bank.py','execute_bank.py','bank_qualification_plan.json',
             'ASTRA_ACCEPTANCE_DECISION_TEST_RECEIPT.json','ASTRA_ACCEPTANCE_DECISION_REVIEW.json',
             'ASTRA_HAND_VIEWS_TEST_RECEIPT.json','ASTRA_HAND_VIEWS_REVIEW.json']
    for name in extra:
        sources[str(H/name)] = sha(H/name)
        if name.endswith('.py'):
            ast.parse((H/name).read_text())
    local_assets=[p for p in (P/'assets_real/usd').rglob('*') if p.is_file()]
    local_assets += [p for p in (P/'assets_src/real').rglob('*.yaml') if p.is_file()]
    for p in local_assets:
        sources[str(p)] = sha(p)
    criteria=j(H/'ACCEPTANCE_CRITERIA.json');jobs=[]
    for row in criteria['initial_bank_rows']:
        bank=R/'banks'/str(row['initial_seed'])/'bank.npz'
        for p in [bank,bank.with_name('metadata.json')]:
            sources[str(p)]=sha(p)
        assert j(bank.with_name('metadata.json'))['bank_sha256']==sha(bank)
        for mode in criteria['modes']:
            job=copy.deepcopy(development['jobs'][0]);name=f"b{row['block']}_{mode}";out=R/'acceptance'/f"block_{row['block']}"/mode
            job.update(id=name,out=str(out),steps=960,scheduled_groups=21,max_wall_seconds=10800)
            assert not out.exists()
            argv=job['argv'];argv[argv.index('--out')+1]=str(out);argv[argv.index('--seeds')+1]=str(row['command_seed']);argv[argv.index('--duration-s')+1]='16'
            job['env'].update(SAFEDUO_INITIAL_BANK_NPZ=str(bank),SAFEDUO_JOINT_MODE=mode,SAFEDUO_CAPTURE_STEPS='[75,480,959]')
            jobs.append(job)
    plan=dict(tag='native_acceptance_01',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              stage='frozen fresh paired full-window acceptance',cwd=development['cwd'],sources=sources,jobs=jobs,
              resource=dict(min_initial_free_MiB=20480,max_owned_process_MiB=12288,min_runtime_free_MiB=6144,poll_seconds=1,
                            scope='frozen resource bounds established by closed known-data dev04/05/06; score criteria unchanged; own-process identity verified stop and actual wait; no foreign process action'),
              local_asset_source_files=len(local_assets),external_asset_dependency_scope='Isaac5.1 remote FR3 references remain external; local USD+YAML frozen, no whole-mesh or supply-chain attestation',
              criteria_sha256=sha(H/'ACCEPTANCE_CRITERIA.json'),camera_registration_sha256=sha(H/'CAMERA_COVERAGE_REGISTRATION.json'),
              future_queue_status='UNKNOWN',hardware_approved=False,production_promoted=False)
    with (H/'ACCEPTANCE_REGISTRATION.json').open('x') as f:
        json.dump(plan,f,indent=2);f.write('\n')
    print('REGISTERED',len(jobs),'jobs',512,'windows',len(sources),'frozen sources',flush=True)
