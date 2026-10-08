"""Canonical byte-preserved camera evidence and full-resolution public case curves."""
from pathlib import Path
import hashlib, json, shutil, datetime, gzip, time
import numpy as np
H=Path(__file__).resolve().parent
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_assets_20261007')
DEST=W/H.name
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def j(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,separators=(',',':'),allow_nan=False);f.write('\n')
def main():
    result=j(H/'ACCEPTANCE_RESULT.json')
    assert result['status']=='COMPLETE_PREREGISTERED_FINITE_ACCEPTANCE'
    assert j(H/'ACCEPTANCE_AUDIT_MATRIX_RESULT.json')['status']=='complete'
    assert not DEST.exists();DEST.mkdir()
    reg=j(H/'ACCEPTANCE_REGISTRATION.json');mapping=[];galleries=[];rawmanifest=[];curve_rows=[]
    def blob(p,job_id,root):
        digest=sha(p);name='blobs/'+digest+p.suffix;target=DEST/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert sha(target)==digest
        mapping.append(dict(job_id=job_id,raw_path=str(p.relative_to(root)),public_path=name,bytes=p.stat().st_size,sha256=digest))
        return name
    for job in reg['jobs']:
        root=Path(job['out']);name=job['id'];mode=job['env']['SAFEDUO_JOINT_MODE'];block=int(name[1])
        assert j(root/'visual_protocol.json')['status']=='complete'
        known={}
        for manifest_name in ['forecast_receipts.json','native_receipts.json','native_contact_receipts.json']:
            known.update({v['path']:v['sha256'] for v in j(root/manifest_name)['chunks']})
        for p in sorted(root.rglob('*')):
            if p.is_file():
                relative=str(p.relative_to(root));digest=known[relative] if relative in known else sha(p)
                rawmanifest.append(dict(job_id=name,path=relative,bytes=p.stat().st_size,sha256=digest,hash_provenance='closed producer chunk hash independently reread by parent and Astra actual0 audits' if relative in known else 'exporter actual file byte hash'))
        # Preserve original camera JSON/PNG/native snapshots; canonical aliases avoid redundant identical all64 snapshots.
        camera_files=[p for folder in ['multiview','hand_views'] for p in (root/folder).rglob('*') if p.is_file()]
        camera_files += list(root.glob('native_render*.npz'))
        camera_files += [root/n for n in ['native_initial.npz','native_contact_identity.json','full_row_identity.json','camera_receipts.json','hand_camera_receipts.json','native_receipts.json','native_contact_receipts.json','forecast_receipts.json','visual_protocol.json','open_hand_initialization.json','input_recipe.npz','mechanism.npz','project_diagnostics.npz','bank_assignment.npz','guard_metadata.json','random_manifest.json']]
        aliases={str(p.relative_to(root)):blob(p,name,root) for p in camera_files}
        overview=j(root/'camera_receipts.json');hands=j(root/'hand_camera_receipts.json')
        for cap in overview['receipts']:
            matched=next(c for c in hands['receipts'] if (c['env_id'],c['step'],c['capture_kind'])==(cap['env_id'],cap['step'],cap['capture_kind']))
            state=j(root/cap['state']);hand_state=j(root/matched['state']);images=[]
            for im in state['images']:
                images.append(dict(path=aliases[im['path']],label=Path(im['path']).stem,raw_path=im['path'],sha256=im['sha256'],type='overview'))
            for im in hand_state['images']:
                images.append(dict(path=aliases[im['path']],label=im['arm']+' '+im['view'],raw_path=im['path'],sha256=im['sha256'],type='hand'))
            assert len(images)==21
            galleries.append(dict(job_id=name,mode=mode,block=block,env_id=cap['env_id'],step=cap['step'],capture_kind=cap['capture_kind'],state_url=aliases[cap['state']],hand_state_url=aliases[matched['state']],state_time_s=state['state_time_s'],physical_simulation_time_s=hand_state['simulation_time_s'],physical_simulation_time_step_index=hand_state['simulation_time_step_index'],native_contact_scope='macro post image, microstep peak retained separately',images=images))
        z=load(root/'cell_001.npz');T=len(z['q']);assert T==960
        c=j(H/(name+'_contacts.json'));m=load(c['metric_file'])['partner_normal_max_N'];assert m.shape==(960,2,64,4)
        dt=j(root/'visual_protocol.json')['dt'] if 'dt' in j(root/'visual_protocol.json') else json.loads(str(z['meta_json']))['dt']
        curves=dict(job_id=name,mode=mode,block=block,times=((np.arange(T)+1)*dt).tolist(),margin_by_env=z['official_margins'].tolist(),native_hand_micro_times=((np.arange(2*T)+1)*dt/2).tolist(),native_hand_max_by_env=m.reshape(2*T,64,4).tolist(),native_hand_threshold_N=.1,represented_sphere_threshold_m=0,macro_time_scope='window elapsed post-step time; actual renderer simulation clock is stored separately in each original state',measured_q_npz='measured_q/'+name+'.npz',raw_cell_sha256=sha(root/'cell_001.npz'),raw_native_contact_metrics_sha256=c['metric_sha256'])
        qp=DEST/'measured_q'/f'{name}.npz';qp.parent.mkdir(exist_ok=True);np.savez_compressed(qp,q=z['q'],q_initial=z['q_initial'],raw_cell_sha256=np.asarray(sha(root/'cell_001.npz')))
        cp=DEST/'curves'/f'{name}.json';write(cp,curves)
        with cp.open('rb') as src,gzip.GzipFile(str(cp)+'.gz',mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
        curve_rows.append(dict(job_id=name,path=str(cp.relative_to(DEST)),frames=T,micro_frames=2*T,cases=64,downsampled=False,raw_cell_sha256=curves['raw_cell_sha256']))
        for suffix in ['native','geometry','contacts','opening','hands']:
            shutil.copyfile(H/(name+'_'+suffix+'.json'),DEST/(name+'_'+suffix+'.json'))
        shutil.copyfile(H/('ASTRA_NATIVE_REAL_'+name+'.json'),DEST/('ASTRA_NATIVE_REAL_'+name+'.json'))
        print('ASSET_JOB',name,'galleries',len(overview['receipts']),flush=True)
    write(DEST/'galleries.json',galleries);write(DEST/'canonical_raw_camera_mapping.json',dict(schema='original_bytes_to_canonical_public_blobs.v1',rows=mapping,no_raw_image_editing=True,original_state_bytes_preserved=True,alias_usage='resolve raw_relative path in original camera state using matching job_id and raw_path',full_forecast_scope='full9021 distance/Jacobian/forecast and complete native/contact streams remain in localNAS archive; public camera+960/1920 curves do not replace those',curve_rows=curve_rows))
    write(DEST/'raw_experiment_manifest.json',dict(roots=[dict(job_id=j['id'],local_root=j['out']) for j in reg['jobs']],files=rawmanifest,bytes=sum(r['bytes'] for r in rawmanifest),scope='all8closed raw products; fullforecast files remain localNAS, not publicGit payload'))
    wait_receipt=H/'acceptance_astra_decision_execution.json'
    while not wait_receipt.is_file():
        print('ASSETS_DRAFT_COMPLETE_WAITING_INDEPENDENT_ACTUAL_EXIT',flush=True);time.sleep(20)
    assert j(wait_receipt)['actual_exit']==0,'independent verifier actualnonzero; draft cannot be published'
    peer=j(H/'ASTRA_ACCEPTANCE_REAL.json');assert peer['status']=='COMPLETE_INDEPENDENT_FINITE_DECISION' and peer['candidate_decision']==result['candidate_decision']
    for name in ['ACCEPTANCE_RESULT.json','ASTRA_ACCEPTANCE_REAL.json','CASE_TABLE.json','COVERAGE_RESULT.json','MECHANISM_RESULT.json','ACCEPTANCE_AUDIT_MATRIX_RESULT.json','ACCEPTANCE_CRITERIA.json','PAIRING_REGISTRATION.json','INITIAL_STATE_ACCEPTANCE_REGISTRATION.json','CAMERA_COVERAGE_REGISTRATION.json','COMMON_HAND_CONTEXT_REGISTRATION.json','ACCEPTANCE_REGISTRATION.json','ASTRA_ACCEPTANCE_DECISION_REVIEW.json','ASTRA_ACCEPTANCE_DECISION_TEST_RECEIPT.json','ASTRA_HAND_VIEWS_REVIEW.json','ASTRA_HAND_VIEWS_TEST_RECEIPT.json']:
        shutil.copyfile(H/name,DEST/name)
    shutil.copyfile(H/'CASE_TABLE.json',DEST/'case_table.json')
    for p in H.glob('*.py'):
        target=DEST/'sources'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    figure_root=Path('/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/figures')
    for p in figure_root.iterdir():
        target=DEST/'figures'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    for name in ['native_acceptance_supervisor_execution.json','acceptance_audit_supervisor_execution.json','acceptance_parent_analysis_execution.json','acceptance_astra_decision_execution.json','acceptance_coverage_execution.json','acceptance_mechanism_execution.json','acceptance_plots_execution.json']:
        shutil.copyfile(H/name,DEST/name)
    rows=[dict(path=str(p.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(DEST.rglob('*')) if p.is_file()]
    assert max(r['bytes'] for r in rows)<95*1024*1024
    manifest=dict(status='COMPLETE_CANONICAL_PUBLIC_ASSET_MANIFEST',files=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows),gallery_groups=len(galleries),original_PNG=sum(len(x['images']) for x in galleries),source_aliases=len(mapping),unique_blobs=len(list((DEST/'blobs').iterdir())),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(DEST/'asset_manifest.json',manifest);write(H/'PANEL_ASSET_BUILD.json',{k:v for k,v in manifest.items() if k!='files'})
    print('ASSETS_COMPLETE',len(rows),manifest['bytes'],manifest['original_PNG'],flush=True)
if __name__=='__main__':main()
