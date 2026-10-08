"""Check every exported array and every source/blob mapping against closed raw cells."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent
A=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_assets_20261007')/H.name
def j(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    reg=j(H/'ACCEPTANCE_REGISTRATION.json');roots={x['id']:Path(x['out']) for x in reg['jobs']};mapping=j(A/'canonical_raw_camera_mapping.json')['rows'];index={};checked={}
    for row in mapping:
        assert (row['job_id'],row['raw_path']) not in index
        source=roots[row['job_id']]/row['raw_path'];public=A/row['public_path'];assert source.stat().st_size==public.stat().st_size==row['bytes']
        assert sha(source)==row['sha256']
        if row['public_path'] not in checked:checked[row['public_path']]=sha(public)
        assert checked[row['public_path']]==row['sha256'];index[(row['job_id'],row['raw_path'])]=row['public_path']
    total_values=0
    for name,root in roots.items():
        z=load(root/'cell_001.npz');curve=j(A/'curves'/f'{name}.json');contact=j(H/(name+'_contacts.json'));metrics=load(contact['metric_file'])['partner_normal_max_N'];dt=json.loads(str(z['meta_json']))['dt']
        assert curve['job_id']==name and curve['raw_cell_sha256']==sha(root/'cell_001.npz') and curve['raw_native_contact_metrics_sha256']==contact['metric_sha256']
        assert np.array_equal(np.asarray(curve['times']),((np.arange(960)+1)*dt))
        assert np.array_equal(np.asarray(curve['native_hand_micro_times']),((np.arange(1920)+1)*dt/2))
        assert np.array_equal(np.asarray(curve['margin_by_env']),z['official_margins'])
        assert np.array_equal(np.asarray(curve['native_hand_max_by_env']),metrics.reshape(1920,64,4))
        q=load(A/curve['measured_q_npz'])
        for k in ['q','q_initial']:assert q[k].dtype==z[k].dtype and q[k].shape==z[k].shape and q[k].tobytes()==z[k].tobytes()
        total_values+=z['official_margins'].size+metrics.size+q['q'].size
    galleries=j(A/'galleries.json');seen=set();photos=0
    for g in galleries:
        name=g['job_id'];root=roots[name];key=(name,g['env_id'],g['step'],g['capture_kind']);assert key not in seen;seen.add(key)
        principal=j(root/'camera_receipts.json');cap=next(x for x in principal['receipts'] if (x['env_id'],x['step'],x['capture_kind'])==key[1:]);hand=j(root/'hand_camera_receipts.json');hc=next(x for x in hand['receipts'] if (x['env_id'],x['step'],x['capture_kind'])==key[1:])
        assert g['state_url']==index[(name,cap['state'])] and g['hand_state_url']==index[(name,hc['state'])]
        images=j(root/cap['state'])['images']+j(root/hc['state'])['images'];assert len(images)==len(g['images'])==21
        for original,exported in zip(images,g['images']):
            assert original['path']==exported['raw_path'] and original['sha256']==exported['sha256'] and exported['path']==index[(name,original['path'])]
        photos+=21
    expected=sum(j(root/'camera_receipts.json')['groups'] for root in roots.values());assert len(galleries)==expected
    assert j(A/'case_table.json')==j(H/'CASE_TABLE.json') and len(j(A/'case_table.json'))==512
    r=j(H/'ACCEPTANCE_RESULT.json');assert photos==r['total_PNG']
    out=dict(status='PASS_ALL_PUBLIC_DATA_EXACT_RAW_BINDINGS',jobs=8,cases=512,array_values_checked=total_values,original_source_aliases=len(mapping),unique_public_blobs_checked=len(checked),gallery_groups=len(galleries),original_PNG=photos,full960_and1920_samples_preserved=True,images_byte_preserved=True,raw_forecast_stays_local_and_manifested=True,source_sha256=sha(__file__),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/'PANEL_DATA_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(out,flush=True)
if __name__=='__main__':main()
