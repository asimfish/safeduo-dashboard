"""Bind duplicate camera execution to the measured pilot, not new trials."""
from pathlib import Path
import hashlib,json,numpy as np
H=Path(__file__).resolve().parent
R=Path('/mnt/nas/data/lyf/double_hand/safety_response_multirow_pilot_20261009_0220')
A=R/'multirow_pilot_v3';B=R/'multirow_camera_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
fields=0
for name in ['response_stream.npz','dynamics_stream.npz','initial_geometry.npz','resolved_native_parameters.npz']:
    with np.load(A/name) as a,np.load(B/name) as b:
        assert set(a.files)==set(b.files)
        for k in a.files:
            assert np.array_equal(a[k],b[k]),(name,k)
            fields+=1
camera=json.loads((B/'pilot_camera_receipts.json').read_text())
assert camera['status']=='complete' and camera['frames']==17 and camera['images']==408
with np.load(B/'response_stream.npz') as stream,np.load(B/'initial_geometry.npz') as initial:
    for event in camera['receipts']:
        p=B/event['native_state_path'];assert sha(p)==event['native_state_sha256']
        with np.load(p) as state:
            for arm in ['F_L','F_R','U_L','U_R']:
                for f in ['q','qd']:
                    expected=initial[arm+'_'+f] if event['step']==-1 else stream['post_'+arm+'_'+f][event['step']]
                    assert np.array_equal(state[arm+'_'+f],expected)
        assert event['render_did_not_advance_physics']
        for image in event['images']:
            assert sha(B/image['path'])==image['sha256']
            assert image['all_selected_spheres_contained'] and image['minimum_plane_margin_m']>=.05
receipt=dict(status='PASS_MATCHED_CAMERA_NATIVE_BINDING',identical_native_fields=fields,
    maximum_native_difference=0,independent_new_trials=0,images_verified=408,frames=17,
    explicit_detail_scope='U_L only; top/front/side contain represented four arms',
    camera_receipt_sha256=sha(B/'pilot_camera_receipts.json'),safety_acceptance=False)
(H/'CAMERA_V4_RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
