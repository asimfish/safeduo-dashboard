"""Offline development-only admission timing audit; no controller replay or tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'safety_mechanism_20261005'
NAS=Path('/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005')

def audit(mode,root,events_file):
    with np.load(root/'cell_001.npz',allow_pickle=False) as z:
        keys=['pre_sphere_centers','sphere_centers','pre_row_id','pre_row_valid',
              'pre_row_d','pre_row_full_rate','pre_row_dmin','full_table_distance',
              'full_table_exempt','kinematic_meta_json']
        d={k:z[k] for k in keys}
    km=json.loads(str(d['kinematic_meta_json']));rm=json.loads((root/'mechanism_metadata.json').read_text())
    ids=np.array(km['pair_sphere_idx']);radii=np.array(km['sphere_radii_m']);nr=km['robot_pair_count']
    classes=np.array(rm['class_index']);nominal=np.array(rm['nominal_dmin'])
    horizon=np.array([.16,.4,.4,.3])[classes]
    table=d['full_table_distance'];before=np.concatenate([table[:1],table[:-1]])
    events=json.loads((OLD/events_file).read_text())['events'];out=[]
    for ev in events:
        e,t,p=ev['env'],ev['first_step'],ev['row_id'];start=max(1,t-60)
        records=[]
        for k in range(start,t+1):
            where=np.flatnonzero(d['pre_row_valid'][k,e]&(d['pre_row_id'][k,e]==p))
            if len(where):
                row=where[0];distance=float(d['pre_row_d'][k,e,row]);rate=float(d['pre_row_full_rate'][k,e,row]);dm=float(d['pre_row_dmin'][k,e,row]);exact=True
            else:
                i,j=ids[p]
                if p<nr:
                    distance=float(np.linalg.norm(d['pre_sphere_centers'][k,e,i]-d['pre_sphere_centers'][k,e,j])-radii[i]-radii[j])
                    post=float(np.linalg.norm(d['sphere_centers'][k,e,i]-d['sphere_centers'][k,e,j])-radii[i]-radii[j])
                else:
                    distance=float(before[k,e,p-nr]);post=float(table[k,e,p-nr])
                rate=(post-distance)*60;dm=float(nominal[p]);exact=False
            records.append(dict(step=k,admitted=bool(len(where)),distance_mm=distance*1000,
                                closing_mm_s=max(0,-rate)*1000,rate_from_exact_pre_state=exact,
                                predicted_critical=bool(distance+horizon[p]*min(rate,0)<=dm+.010)))
        # First membership can be intermittent; continuous membership is separate.
        actual=[x['step'] for x in records if x['admitted']]
        pred=[x['step'] for x in records if x['predicted_critical']]
        uninterrupted=t
        for x in reversed(records):
            if not x['admitted']:break
            uninterrupted=x['step']
        out.append(dict(env=e,channel=ev['channel'],row_id=p,first_step=t,
                        earliest_actual_lead_steps=t-min(actual) if actual else None,
                        uninterrupted_actual_lead_steps=t-uninterrupted,
                        earliest_predicted_lead_steps=t-min(pred) if pred else None,
                        missed_predicted_frames=sum(x['predicted_critical'] and not x['admitted'] for x in records),
                        context=records))
    # Coarse development capacity estimate uses forward finite differences only;
    # it cannot replace online exact-closing admission or guarantee capacity.
    widths=[];maximum_distance_error=0.
    for k in range(1,960,16):
        pre=d['pre_sphere_centers'][k];post=d['sphere_centers'][k]
        rd=np.linalg.norm(pre[:,ids[:nr,0]]-pre[:,ids[:nr,1]],axis=-1)-radii[ids[:nr,0]]-radii[ids[:nr,1]]
        rp=np.linalg.norm(post[:,ids[:nr,0]]-post[:,ids[:nr,1]],axis=-1)-radii[ids[:nr,0]]-radii[ids[:nr,1]]
        dist=np.concatenate([rd,before[k]],1);after=np.concatenate([rp,table[k]],1)
        pred=dist-horizon*np.maximum((dist-after)*60,0)
        union=(dist<=nominal+.010)|(pred<=nominal+.010)
        for e in range(64):
            valid=d['pre_row_valid'][k,e];rowids=d['pre_row_id'][k,e,valid]
            union[e,rowids]=True
            if len(rowids):maximum_distance_error=max(maximum_distance_error,float(np.max(np.abs(dist[e,rowids]-d['pre_row_d'][k,e,valid]))))
        widths.extend(union.sum(-1).tolist())
    return dict(mode=mode,trace_sha256=hashlib.sha256((root/'cell_001.npz').read_bytes()).hexdigest(),
                first_class_events=len(out),events_with_missed_predicted_frames=sum(x['missed_predicted_frames']>0 for x in out),
                events=out,coarse_capacity_estimate_max=max(widths),coarse_capacity_estimate_p95=float(np.percentile(widths,95)),
                reconstructed_distance_max_error_m=maximum_distance_error,
                warning='Unadmitted row rates and coarse capacity use observed next-state finite differences (hindsight), not deployable signals. Admitted rows use exact pre-state full-body distance rate. Only old development seed60317411; no new independent trajectories.')

def main():
    r=dict(schema='safeduo.admission_timing_audit.v1',development_only=True,policy_outcomes_used_for_new_bank=False,
           traces=[audit('baseline',NAS/'trace_cells/causal_baseline_60317411','causal_results.json'),
                   audit('envelope_050',NAS/'candidate_trace_cells_v2/causal_envelope_60317411_v2','candidate_causal_results.json')])
    (HERE/'admission_audit.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k not in ('events','warning')} for x in r['traces']]))

if __name__=='__main__':main()
