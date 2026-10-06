"""Development-only reanalysis; original outcome gates are immutable."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=Path('/home/liyufeng/safeduo/artifacts/forensics/20261004_randomized_safety_campaign_v4')
ARMS=('F_L','F_R','U_L','U_R');OBJECTS=('beam700','beam300')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bottom_z(states,sizes):
    q=states[...,3:7].astype(np.float64);q/=np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    row=np.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1)
    return states[...,2]-np.sum(np.abs(row)*np.asarray(sizes)/2,axis=-1)

def main():
    schema=json.loads((OLD/'block_00/dense_schema.json').read_text())
    hand_ids={a:[i for i,v in enumerate(schema['pair_sensor_items']) if v['arm']==a and any(k in v['body'] for k in ('thumb','index','middle','ring','pinky','little'))] for a in ARMS}
    assert all(len(v)==12 for v in hand_ids.values())
    sizes=np.asarray([[.04,.7,.08],[.04,.3,.08]])
    rows=[];sources=[];sampled={}
    for b in (0,1):
        folder=OLD/f'block_{b:02d}';receipt=json.loads((folder/'recording_receipt.json').read_text())
        arrays={};steps=[]
        for c in receipt['chunks']:
            path=folder/c['file'];digest=sha(path);assert digest==c['sha256']
            sources.append(dict(path=str(path),sha256=digest))
            with np.load(path,allow_pickle=False) as z:
                keys=['time','step','objects','table_normal','pair_normal','violation']+[a+':arm_q' for a in ARMS]+[a+':arm_target' for a in ARMS]
                for k in keys:arrays.setdefault(k,[]).append(z[k])
        d={k:np.concatenate(v) for k,v in arrays.items()}
        assert np.array_equal(d['step'],np.arange(1572))
        origins=np.asarray(json.loads((folder/'native_initial_readback.json').read_text())['env_origins_world_m'])
        states=d['objects'].copy();states[...,:3]-=origins[None,:,None,:]
        bottom=bottom_z(states,sizes);clearance=bottom-.8
        time=d['time'][:,0];scheduled=(d['step']>=480)&(d['step']<=720)
        force=np.linalg.norm(d['table_normal'],axis=-1)
        native_initial=json.loads((folder/'native_initial_readback.json').read_text())
        for e in range(64):
            case=dict(block=b,env=e,objects={})
            for oi,o in enumerate(OBJECTS):
                assigned=ARMS[oi*2:oi*2+2]
                hands=np.stack([np.linalg.norm(d['pair_normal'][:,e,hand_ids[a],oi,:],axis=-1).sum(-1) for a in assigned],-1)
                own=force[:,e,oi,oi]
                sep=clearance[:,e,oi]>.003
                lift=states[:,e,oi,2]-native_initial['objects'][o]['initial_pose_world_wxyz'][e][2]
                release=(time>=15.5)&(time<18.7)
                after=(time>=18.7)
                obj=dict(scheduled_positive_support_steps=int(((own>1e-6)&scheduled).sum()),
                    scheduled_positive_support_and_clear_steps=int(((own>1e-6)&scheduled&sep).sum()),
                    actual_bbox_clear_steps=int(sep.sum()),actual_bbox_clear_positive_support_steps=int(((own>1e-6)&sep).sum()),
                    max_bbox_clear_support_n=float(own[sep].max()) if sep.any() else None,
                    lifted_50mm_steps=int((lift>.05).sum()),
                    lifted_50mm_both_assigned_hands_gt_0_1n_steps=int(((lift>.05)&(hands>.1).all(-1)).sum()),
                    released_period_hand_contact_steps=int(((hands>.1).any(-1)&release).sum()),
                    after_clearance_hand_contact_steps=int(((hands>.1).any(-1)&after).sum()),
                    release_to_final_displacement_m=float(np.linalg.norm(states[-1,e,oi,:3]-states[np.flatnonzero(time>=15.5)[0],e,oi,:3])))
                case['objects'][o]=obj
            rows.append(case)
        for e in (0,1):
            sampled[f'b{b}_e{e}']=dict(time_s=d['time'][:,1].tolist(),object_z_m=states[:,e,:,2].tolist(),bbox_clearance_m=clearance[:,e].tolist(),
                own_table_normal_n=np.stack([force[:,e,i,i] for i in range(2)],-1).tolist(),
                hand_normal_n=[[float(np.linalg.norm(d['pair_normal'][t,e,hand_ids[a],oi],axis=-1).sum()) for oi in range(2) for a in ARMS[oi*2:oi*2+2]] for t in range(1572)])
    totals={o:{k:sum(r['objects'][o][k] for r in rows) for k in ['scheduled_positive_support_steps','scheduled_positive_support_and_clear_steps','actual_bbox_clear_steps','actual_bbox_clear_positive_support_steps','lifted_50mm_steps','lifted_50mm_both_assigned_hands_gt_0_1n_steps','released_period_hand_contact_steps','after_clearance_hand_contact_steps']} for o in OBJECTS}
    result=dict(status='PASS_DEVELOPMENT_TRACE_AUDIT',tasks=128,blocks=[0,1],new_physical_trials=0,original_task_and_sensor_gates_unchanged=True,
        scope='authored rotated bounding-box clearance and last-substep native filtered normal; not validated mesh collision or total force',table_top_m=.8,bbox_clear_threshold_m=.003,contact_description_threshold_n=.1,
        sources=sources,source_sha256=sha(__file__),totals=totals,cases=rows,examples=sampled)
    (ROOT/'retrospective.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],tasks=128,totals=totals)))

if __name__=='__main__':main()
