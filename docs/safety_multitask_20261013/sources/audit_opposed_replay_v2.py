"""Recompute known native105 contact opposition; preserve all old evidence."""
from pathlib import Path
import hashlib,json
import numpy as np
from opposed_contact_v1 import opposed_contact

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
T=Path('/mnt/nas/data/lyf/double_hand/safeduo_task_foundation_20261012')
ARMS=('F_L','F_R','U_L','U_R')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import sys
    sys.path.insert(0,str(T))
    from episode_judge import continuous_window
    raw=Path(json.loads((H/'fourhand_native_probe_v105/RESOURCE_EXECUTION_V1.json').read_text())['out'])
    identity=json.loads((raw/'native_task_identity.json').read_text());partners=identity['partners']
    totals={a:0 for a in ARMS};masks=[];times=[];sources={}
    for rec in json.loads((raw/'physics_chunks.json').read_text()):
        p=raw/rec['path'];assert sha(p)==rec['sha256'];sources[str(p)]=rec['sha256']
        with np.load(p,allow_pickle=False) as z:
            offsets=z['object_normal_event_offsets'];flags=np.zeros((len(z['state_time_s']),4),bool)
            for i in range(len(flags)):
                lo,hi=map(int,offsets[i:i+2]);owners=z['object_normal_partner_indices'][lo:hi]
                for ai,a in enumerate(ARMS):
                    ids=[lo+j for j,v in enumerate(owners) if '/'+a+'/' in partners[int(v)]
                        and any(s in partners[int(v)].lower() for s in ['thumb','index','middle','ring','little','pinky','hand_base'])]
                    flags[i,ai]=opposed_contact([partners[int(z['object_normal_partner_indices'][j])] for j in ids],
                        z['object_normal_normals'][ids].reshape(-1,3),z['object_normal_points'][ids].reshape(-1,3),
                        z['object_normal_normal_forces'][ids,0])
            masks.append(flags);times.append(z['state_time_s'].copy())
    masks=np.concatenate(masks);times=np.concatenate(times);dt=float(np.diff(times)[0])
    report={'status':'CORRECTED_CONTACT_OPPOSITION_OBSERVER_REPLAY','native_source_version':105,'new_native_episodes':0,
        'physics_events':len(times),'observer_error':'Full center-relative p0 dot p1 incorrectly depends on beam length and longitudinal grasp location',
        'observer_correction':'Opposite outward normals plus positive contact separation along the first normal and negative along the second',
        'per_arm':{a:{'opposed_events':int(masks[:,i].sum()),'longest_continuous_opposition_s':continuous_window(times,masks[:,i],dt)} for i,a in enumerate(ARMS)},
        'native_safety_threshold_changes':0,'original_task_completion':False,'new_complete_task_successes':0,
        'force_closure_certified':False,'fullSystem0_accepted':False,
        'source_sha256':{**sources,str(K/'opposed_contact_v1.py'):sha(K/'opposed_contact_v1.py'),str(Path(__file__)):sha(__file__)}}
    output=K/'NATIVE105_CORRECTED_OPPOSITION_V2.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(K/'NATIVE105_CORRECTED_OPPOSITION_TIMELINE_V2.npz',time_s=times,opposed=masks)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
