"""Independent offline scoring of complete native trials and unchanged gates."""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np
from PIL import Image
from retrospective import bottom_z
from object_binding import ARMS
R=Path(__file__).resolve().parent;OBJECTS=('beam700','beam300')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();folder=a.out
    rec=json.loads((folder/'recording_receipt.json').read_text());initial=json.loads((folder/'native_initial.json').read_text())
    assert rec['status']=='PASS_COMPLETE_NATIVE_RECORDING' and rec['steps']==1572 and rec['cases']==8
    assert sha(folder/'native_initial.json')==rec['native_initial_sha256']
    reg_path=R/('REGISTRATION_V2.json' if (folder/'effective_registration.json').exists() else 'REGISTRATION.json')
    if (folder/'effective_registration.json').exists():reg_path=Path(json.loads((folder/'effective_registration.json').read_text())['path'])
    reg=json.loads(reg_path.read_text());assert sha(reg_path)==rec['registration_sha256']==initial['registration_sha256']
    arrays={}
    for c in rec['chunks']:
        path=folder/c['file'];assert sha(path)==c['sha256']
        with np.load(path,allow_pickle=False) as z:
            for k in z.files:
                assert np.isfinite(z[k]).all(),k;arrays.setdefault(k,[]).append(z[k])
    d={k:np.concatenate(v) for k,v in arrays.items()};assert np.array_equal(d['step'],np.arange(1572))
    origins=np.asarray(initial['origins']);states=d['objects'].astype(np.float64);states[...,:3]-=origins[None,:,None,:]
    bottom=bottom_z(states,reg['sizes_m']);clearance=bottom-.8
    tableids=[next(i for i,v in enumerate(initial['partners']) if v.get('table')==t) for t in ('TableF','TableU')]
    assert all(abs(initial['native_table_world_max'][t][2]-.8)<1e-6 for t in ('TableF','TableU'))
    forces=np.linalg.norm(d['object_partner_normal'],axis=-1)
    own=np.stack([forces[:,:,i,tableids[i]] for i in range(2)],-1)
    hand=np.stack([np.stack([forces[:,:,oi,[i for i,v in enumerate(initial['partners']) if v['arm']==arm and v['hand']]].sum(-1) for arm in ARMS[2*oi:2*oi+2]],-1) for oi in range(2)],2)
    t=d['time'][:,0];initial_z=np.asarray(initial['task_input'])[:,:,2];lift=states[:,:,:,2]-initial_z[None]
    q=states[:,:,:,3:7];q/=np.linalg.norm(q,axis=-1,keepdims=True)
    tilt=np.degrees(np.arccos(np.clip(1-2*(q[...,1]**2+q[...,2]**2),-1,1)))
    cases=[];exports=[]
    for e,c in enumerate(reg['cases']):
        objects={};viols=int(d['violation'][:,e].sum());resets=int((d['termination'][:,e]|d['truncation'][:,e]).sum())
        for oi,o in enumerate(OBJECTS):
            xy=float(np.linalg.norm(states[-1,e,oi,:2]-np.asarray(reg['goals_local_m'][oi][:2])))
            dz=float(states[-1,e,oi,2]-initial_z[e,oi]);ml=float(lift[:,e,oi].max());ft=float(tilt[-1,e,oi])
            gates=dict(xy=xy<=.08,tilt=ft<=10,lift=ml>.05,height=-.006<dz<.012,official_violation=viols==0,resets=resets==0)
            above=lift[:,e,oi]>.05;both=(hand[:,e,oi]>.1).all(-1)
            clear=clearance[:,e,oi]>.003;release=t>=15.5
            rest=np.median(d['object_partner_normal'][60:180,e,oi,tableids[oi],2]);mg=np.asarray(initial['native_mass_kg'][o])[e,0]*9.81
            objects[o]=dict(pass_task=all(gates.values()),gates=gates,final_xy_error_m=xy,final_tilt_deg=ft,max_lift_m=ml,final_delta_z_m=dz,
                lifted_50mm_steps=int(above.sum()),lifted_50mm_both_hands_steps=int((above&both).sum()),lifted_50mm_both_hands_time_s=float((above&both).sum()*reg['steps']/reg['steps']*(d['time'][0,1]-d['time'][0,0])),
                actual_bbox_clear_steps=int(clear.sum()),bbox_clear_positive_support_steps=int((clear&(own[:,e,oi]>1e-6)).sum()),
                wrong_table_max_normal_n=float(forces[:,e,oi,tableids[1-oi]].max()),initial_weight_n=float(mg),initial_table_median_z_n=float(rest),initial_weight_relative_error=float(abs(rest-mg)/mg),
                post_open_any_hand_contact_steps=int((release&(hand[:,e,oi]>.1).any(-1)).sum()),
                after_clearance_any_hand_contact_steps=int(((t>=18.7)&(hand[:,e,oi]>.1).any(-1)).sum()))
        cases.append(dict(**{k:v for k,v in c.items() if k!='objects'},layout_objects=c['objects'],pass_task=all(v['pass_task'] for v in objects.values()),official_violation_steps=viols,resets=resets,objects=objects))
        for step in range(1572):
            row=dict(env=e,step=step,state_time_s=float(d['time'][step,1]))
            for oi,o in enumerate(OBJECTS):
                for name,value in zip(('x_m','y_m','z_m'),states[step,e,oi,:3]):row[o+'_'+name]=float(value)
                row[o+'_lift_m']=float(lift[step,e,oi]);row[o+'_table_normal_n']=float(own[step,e,oi]);row[o+'_left_hand_normal_n']=float(hand[step,e,oi,0]);row[o+'_right_hand_normal_n']=float(hand[step,e,oi,1])
            exports.append(row)
    with np.load(folder/'bound_reference.npz',allow_pickle=False) as z:
        reference_contrast={a:[float(np.abs(z['q_'+a][i+4]-z['q_'+a][i]).max()) for i in range(4)] for a in ARMS}
        nominal_exact=all(np.array_equal(z['q_'+a][0],z['q_'+a][4]) for a in ARMS)
        input_pose_max_diff=float(np.abs(z['task_input'][:4,:,:7]-z['task_input'][4:,:,:7]).max())
    image_readbacks=[]
    for item in rec['images']:
        path=folder/item['file'];assert sha(path)==item['sha256']
        with Image.open(path) as image:assert image.size==(1280,720);image.verify()
        assert np.array_equal(np.asarray(item['objects'],dtype=np.float32),d['objects'][item['step'],item['env']])
        image_readbacks.append({k:item[k] for k in ('file','env','step','view','sha256','state_time_s')})
    totals={mode:dict(cases=4,passed=sum(c['pass_task'] for c in cases if c['binding']==flag),violating=sum(c['official_violation_steps']>0 for c in cases if c['binding']==flag)) for mode,flag in [('fixed_replay',False),('object_pose_binding',True)]}
    result=dict(status='PASS_NATIVE_TRACE_AUDIT',cases=cases,totals=totals,complete_task_windows=8,new_final_trials=0,
        nominal_reference_bitexact=nominal_exact,paired_local_input_max_difference=input_pose_max_diff,reference_max_difference_rad=reference_contrast,
        native_images=image_readbacks,native_source_directory=str(folder),registration_sha256=sha(reg_path),recording_receipt_sha256=sha(folder/'recording_receipt.json'),
        contact_capacity_saturation=False,full_g0_pass=False,task_reliability_accepted=False,
        scope='explicit development contrast; fixed original task gates; normals at last substep, authored bbox, no full-scene collision or friction certification',
        source_sha256=sha(__file__))
    (R/'native_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (R/'native_state_timeseries.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exports[0]),lineterminator='\n');writer.writeheader();writer.writerows(exports)
    np.savez_compressed(R/'native_curves.npz',time=d['time'][:,1],states=states,lift=lift,tilt=tilt,hand_normal=hand,own_table_normal=own,bbox_clearance=clearance,violation=d['violation'])
    print(json.dumps(dict(status=result['status'],totals=totals,nominal_reference_bitexact=nominal_exact,paired_local_input_max_difference=input_pose_max_diff,images=len(image_readbacks)),indent=2))

if __name__=='__main__':main()
