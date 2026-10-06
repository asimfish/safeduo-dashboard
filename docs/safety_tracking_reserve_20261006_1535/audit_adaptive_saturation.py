"""Post-observation saturation and recorded-trajectory equality diagnostics."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert json.loads((HERE/'NUMERIC_EXECUTION.json').read_text())['status']=='PASS_ALL_NUMERIC_CLOSED'
    rows=[];initial=[]
    for plan_path in sorted((HERE/'plans').glob('holdout_*_plan.json')):
        plan=json.loads(plan_path.read_text());root=Path(plan['output_root'])
        jobs={j['env']['SAFEDUO_JOINT_MODE']:j for j in plan['jobs']};a=root/jobs['delay_reserve']['id'];b=root/jobs['tight_reference']['id']
        assert json.loads((root/'campaign.json').read_text())['status']=='complete'
        rec=json.loads((a/'forecast_receipts.json').read_text());lo=np.inf;hi=-np.inf;floor=count=0
        for c in rec['chunks']:
            file=a/c['path'];assert sha(file)==c['sha256']
            with np.load(file,allow_pickle=False) as z:g=z['reserve_gap'].copy()
            lo=min(lo,float(g.min()));hi=max(hi,float(g.max()));floor+=int((g==np.float32(.010)).sum());count+=g.size
        equality={}
        with np.load(a/'cell_001.npz',allow_pickle=False) as x,np.load(b/'cell_001.npz',allow_pickle=False) as y:
            for k in ['q','cmd','exec','controller_target','actuator_target','official_margins','pair_margin']:
                equality[k]=bool(np.array_equal(x[k],y[k]))
        rows.append(dict(seed=jobs['delay_reserve']['expected']['seed'],min_gap=lo,max_gap=hi,tight_floor_env_steps=floor,total_env_steps=count,recorded_arrays_bitwise_equal=equality))
        first=a/rec['chunks'][0]['path']
        with np.load(first,allow_pickle=False) as source:
            z={k:source[k][0].copy() for k in ['reserve_forecast','measured_d','dmin','exempt']}
        identity=json.loads((a/'full_row_identity.json').read_text());exempt=np.unpackbits(z['exempt'],axis=-1)[:,:9021].astype(bool)
        risk=np.where(exempt,np.inf,z['reserve_forecast']-z['dmin']);ids=risk.argmin(-1)
        with np.load(a/'projection_snapshots/step_0000.npz',allow_pickle=False) as source:s={k:source[k].copy() for k in ['selected_ids','snapshot_J_F','snapshot_J_U']}
        for e,row in enumerate(ids):
            pos=np.flatnonzero(s['selected_ids'][e]==row);assert len(pos)==1
            jF=s['snapshot_J_F'][e,pos[0]];jU=s['snapshot_J_U'][e,pos[0]]
            pair=identity['pair_sphere_idx'][int(row)];cls=identity['class_id'][int(row)]
            names=[identity['sphere_names'][pair[0]]]
            names.append('table_'+str(pair[1]) if cls==2 else identity['sphere_names'][pair[1]])
            initial.append(dict(seed=jobs['delay_reserve']['expected']['seed'],env=e,row=int(row),raw_class=int(cls),sphere_names=names,
                distance_m=float(z['measured_d'][e,row]),risk_margin_m=float(risk[e,row]),J_F_norm=float(np.linalg.norm(jF.astype(np.float64))),J_U_norm=float(np.linalg.norm(jU.astype(np.float64))),
                selected_J_exact_zero=not bool(jF.any() or jU.any()),physical_distance_positive=bool(z['measured_d'][e,row]>0)))
    receipt=dict(status='PASS_CLOSED_EXPLORATORY_SATURATION_AUDIT',rows=rows,initial_min_rows=initial,initial_cases=192,
        initial_zero_gradient_min_rows=sum(r['selected_J_exact_zero'] for r in initial),initial_positive_min_distances=sum(r['physical_distance_positive'] for r in initial),
        adaptive_tight_floor_env_steps=sum(r['tight_floor_env_steps'] for r in rows),total_env_steps=sum(r['total_env_steps'] for r in rows),
        all_declared_recorded_arrays_equal=all(all(r['recorded_arrays_bitwise_equal'].values()) for r in rows),
        physics_global_uncontrollability_certified=False,scope='post-observation diagnostic, no policy tuning or new random denominator; exact zero selected J at initial state does not certify global immobility; equality limited to declared saved arrays')
    with (HERE/'ADAPTIVE_SATURATION_AUDIT.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print('SATURATION_DIAGNOSIS',receipt['adaptive_tight_floor_env_steps'],receipt['total_env_steps'],receipt['initial_zero_gradient_min_rows'],receipt['all_declared_recorded_arrays_equal'],flush=True)


if __name__=='__main__':main()
