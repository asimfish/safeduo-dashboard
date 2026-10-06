"""Frozen numerical gates; offline scorer does not call the interlock."""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ARMS=('F_L','F_R','U_L','U_R');OBJECTS=('beam700','beam300')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(folder,reg):
    rec=json.loads((folder/'recording_receipt.json').read_text());init=json.loads((folder/'native_initial.json').read_text())
    assert not (folder/'failure.txt').exists()
    assert rec['status']=='PASS_COMPLETE_NATIVE_RECORDING' and rec['steps']==reg['steps'] and rec['cases']==12
    assert rec['registration_sha256']==sha(R/'REGISTRATION.json') and rec['native_initial_sha256']==sha(folder/'native_initial.json')
    arrays={}
    for c in rec['chunks']:
        assert sha(folder/c['file'])==c['sha256']
        with np.load(folder/c['file'],allow_pickle=False) as z:
            for k in z.files:
                assert np.isfinite(z[k]).all();arrays.setdefault(k,[]).append(z[k])
    d={k:np.concatenate(v) for k,v in arrays.items()};assert np.array_equal(d['step'],np.arange(reg['steps']))
    for image in rec['images']:
        assert sha(folder/image['file'])==image['sha256']
        assert np.array_equal(np.array(image['objects'],dtype=np.float32),d['objects'][image['step'],image['env']])
    assert all(all(v==0 for v in check['native_readback_max_abs_differences'].values()) for check in rec['native_render_checks'])
    return d,rec,init
def score(folder,reg):
    d,rec,init=load(folder,reg);block=rec['block'];cases=[];rows=[]
    states=d['objects'].astype(np.float64);states[...,:3]-=np.asarray(init['origins'])[None,:,None,:]
    lift=states[:,:,:,2]-np.asarray(init['task_input'])[None,:,:,2]
    q=states[...,3:7];q/=np.linalg.norm(q,axis=-1,keepdims=True)
    tilt=np.degrees(np.arccos(np.clip(1-2*(q[...,1]**2+q[...,2]**2),-1,1)))
    forces=np.linalg.norm(d['object_partner_normal'],axis=-1);partners=init['partners']
    ti=[next(j for j,p in enumerate(partners) if p.get('table')==t) for t in ('TableF','TableU')]
    hands=np.stack([np.stack([forces[:,:,i,[j for j,p in enumerate(partners) if p['arm']==a and p['hand']]].sum(-1) for a in ARMS[2*i:2*i+2]],-1) for i in range(2)],2)
    own=np.stack([forces[:,:,i,ti[i]] for i in range(2)],-1)
    residual=np.linalg.norm(d['object_net_contact']-d['object_partner_normal'].sum(-2),axis=-1)
    for e,c in enumerate(reg['blocks'][block]['cases']):
        violations=int(d['violation'][:,e].sum());resets=int((d['termination'][:,e]|d['truncation'][:,e]).sum())
        phase=d['phase'][:,e];aborted=bool((d['gate_stage'][:,e]==3).any());complete=phase[-1]>=24.2
        objects={}
        for i,o in enumerate(OBJECTS):
            xy=float(np.linalg.norm(states[-1,e,i,:2]-np.asarray(reg['goals_local_m'][i][:2])));dz=float(lift[-1,e,i]);ml=float(lift[:,e,i].max());ft=float(tilt[-1,e,i])
            gates=dict(xy=xy<=.08,tilt=ft<=10,lift=ml>.05,height=-.006<dz<.012,official_violation=violations==0,resets=resets==0)
            early=(phase<15.2)&(lift[:,e,i]>.05)&(hands[:,e,i]>.1).all(-1)
            closed=(phase>=15.2)&(phase<18.7)
            clear=phase>=18.7
            contacts=(hands[:,e,i]>.1).any(-1)
            # A recontact event requires 6 consecutive contact-free states first.
            free=0;armed=False;recontacts=0;first=None
            for s in np.flatnonzero(clear):
                if contacts[s]:
                    if armed:recontacts+=1;armed=False;first=float(d['time'][s,1]) if first is None else first
                    free=0
                else:
                    free+=1
                    if free>=6:armed=True
            mg=float(np.asarray(init['native_mass_kg'][o])[e,0]*9.81)
            rest=float(np.median(d['object_partner_normal'][60:180,e,i,ti[i],2]))
            objects[o]=dict(pass_original_task=all(gates.values()),gates=gates,final_xy_error_m=xy,final_tilt_deg=ft,max_lift_m=ml,final_delta_z_m=dz,
                planned_lift_both_hands_states=int(early.sum()),planned_lift_both_hands_time_s=float(early.sum()*reg['native_control_dt_s']),
                tilt_over_80_states=int((tilt[:,e,i]>80).sum()),max_tilt_deg=float(tilt[:,e,i].max()),
                release_contact_states=int((closed&contacts).sum()),clearance_contact_states=int((clear&contacts).sum()),recontact_events=recontacts,first_recontact_s=first,
                release_max_speed_m_s=float(np.linalg.norm(states[closed,e,i,7:10],axis=-1).max()),
                initial_weight_relative_error=abs(rest-mg)/mg,initial_table_z_n=rest,weight_n=mg,wrong_table_max_n=float(forces[:,e,i,ti[1-i]].max()),
                reported_net_minus_filtered_max_n=float(residual[:,e,i].max()),terminal_table_n=float(own[-1,e,i]))
        hold=d['command_held'][:,e];cmdmax=max(float(np.abs(d[a+':cmd'][hold,e]).max()) if hold.any() else 0 for a in ARMS)
        assert cmdmax==0,(block,e,'held command was not zero',cmdmax)
        # Actual motion is retained: a held target does not zero physical velocity.
        path=sum(float(np.linalg.norm(np.diff(d[a+':q'][:,e],axis=0),axis=-1).sum()) for a in ARMS)
        held_motion=sum(float((np.linalg.norm(d[a+':qd'][hold,e],axis=-1)*reg['native_control_dt_s']).sum()) for a in ARMS)
        debt_motion=sum(float(np.linalg.norm(d[a+':cmd'][(phase>=15.2)&(phase<18.7),e],axis=-1).sum()) for a in ARMS)
        cases.append(dict(block=block,env=e,layout=c['layout'],method=c['method'],physics_seed=reg['blocks'][block]['physics_seed'],objects=objects,
            pass_original_task=all(x['pass_original_task'] for x in objects.values()),complete_reference=bool(complete),aborted=aborted,abort_reason=int(d['gate_reason'][-1,e]),
            accepted_task=all(x['pass_original_task'] for x in objects.values()) and complete and not aborted,
            official_violation_steps=violations,resets=resets,final_phase_s=float(phase[-1]),held_command_max_rad=cmdmax,held_steps=int(hold.sum()),
            held_actual_joint_motion_rad=held_motion,actual_joint_path_rad=path,release_command_path_rad=debt_motion))
        for s in range(reg['steps']):
            row=dict(block=block,env=e,layout=c['layout'],method=c['method'],time_s=float(d['time'][s,1]),phase_s=float(phase[s]),stage=int(d['gate_stage'][s,e]),held=bool(hold[s]))
            for i,o in enumerate(OBJECTS):
                row.update({o+'_lift_m':float(lift[s,e,i]),o+'_tilt_deg':float(tilt[s,e,i]),o+'_hand_n':float(hands[s,e,i].sum()),o+'_table_n':float(own[s,e,i])})
            rows.append(row)
    return cases,rows,rec
def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);args=p.parse_args()
    reg=json.loads((R/'REGISTRATION.json').read_text());cases=[];rows=[];receipts=[]
    for b in range(2):
        c,r,rec=score(args.raw_root/f'block_{b}',reg);cases+=c;rows+=r;receipts.append(dict(block=b,receipt_sha256=sha(args.raw_root/f'block_{b}'/'recording_receipt.json'),images=len(rec['images']),video_frames=len(rec['video_frames']),render_checks=len(rec['native_render_checks'])))
    totals={m:dict(windows=8,pass_original_task=sum(c['pass_original_task'] for c in cases if c['method']==m),accepted_task=sum(c['accepted_task'] for c in cases if c['method']==m),aborted=sum(c['aborted'] for c in cases if c['method']==m),violating=sum(c['official_violation_steps']>0 for c in cases if c['method']==m),either_object_tilt80=sum(any(o['tilt_over_80_states']>0 for o in c['objects'].values()) for c in cases if c['method']==m),total_actual_joint_path_rad=sum(c['actual_joint_path_rad'] for c in cases if c['method']==m)) for m in reg['methods']}
    result=dict(status='PASS_COMPLETE_24_WINDOW_NATIVE_AUDIT',totals=totals,cases=cases,receipts=receipts,registration_sha256=sha(R/'REGISTRATION.json'),scorer_sha256=sha(__file__),new_final_trials=0,full_safety_accepted=False,scope=reg['scope'])
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (R/'timeseries.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(status=result['status'],totals=totals,receipts=receipts),indent=2))
if __name__=='__main__':main()
