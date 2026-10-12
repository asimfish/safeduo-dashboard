"""Read-only prospective-decision replay; never claims counterfactual physics."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from actual_state_arm_guard_v1 import ARMS,ActualStateArmGuard

P=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    result=P/'KNOWN108_ACTUAL_STATE_REPLAY_V1.json';assert not result.exists()
    sys.path.insert(0,str(H));sys.path.insert(0,'/home/liyufeng/safeduo/src')
    from prepare_random_arm_pair107_108_v1 import proxy_gaps
    from safeduo.delta.task_record_s9 import make_s9_provider
    torch.set_num_threads(1);provider=make_s9_provider(1,device='cpu',spheres='v7')
    reg=read(H/'fourhand_native_probe_v108/REGISTRATION_V1.json')
    raw=Path(read(H/'fourhand_native_probe_v108/RESOURCE_EXECUTION_V1.json')['out'])
    with np.load(raw/'native_parameters.npz') as z:params={k:z[k] for k in z.files}
    with np.load(reg['contact_reproducer']['fixture']) as z:seed={k:z[k] for k in z.files}
    with np.load(H/'motion_safety_benchmark_v1/RETREAT_COMMAND_BANK_V1.npz') as z:
        retreat={a:z[a+'_issued_target'][-1].copy() for a in ARMS}
    with np.load(reg['motion_test']['command_bank']) as z:issued={a:z[a+'_issued_target'] for a in ARMS}
    fields=['physics_event','state_time_s','all9021_raw_geometry']
    fields += [a+s for a in ARMS for s in ('_motor_control_q_before','_motor_control_qd_before','_native_q','_native_qd','_applied_target')]
    parts=[];sources={}
    for c in read(raw/'physics_chunks.json'):
        f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
        with np.load(f) as z:parts.append({k:z[k].copy() for k in fields})
    data={k:np.concatenate([part[k] for part in parts]) for k in fields}
    ids={a:params[a+'_controlled_indices'] for a in ARMS}
    initial={a:seed[a+'_actual_full_position_target_seed'][0,ids[a]] for a in ARMS}
    pending=[{a:initial[a].copy() for a in ARMS} for _ in range(6)]
    guard=ActualStateArmGuard(lambda q:proxy_gaps(provider,q),
        {a:params[a+'_hard_limits'][0] for a in ARMS},
        {a:params[a+'_max_velocity'][0] for a in ARMS},ids,retreat)
    decisions=[]
    for step in range(len(data['physics_event'])//16):
        pending.append({a:issued[a][step].copy() for a in ARMS});applied=pending.pop(0)
        for a in ARMS:np.testing.assert_array_equal(applied[a],data[a+'_applied_target'][step*16,0])
        if step<4:continue
        event=step*16
        fullq={a:data[a+'_motor_control_q_before'][event,0] for a in ARMS}
        fullv={a:data[a+'_motor_control_qd_before'][event,0] for a in ARMS}
        for a in ARMS:
            np.testing.assert_array_equal(fullq[a],data[a+'_native_q'][event-1,0])
            np.testing.assert_array_equal(fullv[a],data[a+'_native_qd'][event-1,0])
        d=guard.decide(fullq,fullv,applied,pending,data['all9021_raw_geometry'][event-1,0])
        decisions.append({'control':step,'before_physics_event':event,'state_time_s':step/60.,
                          'mode':d.mode,'alpha':d.alpha,'forecast_minimum_m':d.forecast_minimum_m,
                          'actual_minimum_m':d.actual_minimum_m,
                          'queue_maximum_departure_rad':d.queue_maximum_departure_rad,
                          'maximum_target_difference_from_old_command_rad':max(float(abs(d.target[a]-applied[a]).max()) for a in ARMS)})
    firstbad=int(np.flatnonzero(data['all9021_raw_geometry'].min((1,2))<0)[0])
    interventions=[d for d in decisions if d['mode']!='REQUEST']
    first=interventions[0] if interventions else None
    output={'scope':'OFFLINE_RECORDED_DECISION_REPLAY_ONLY','recorded_first_proxy_failure_event':firstbad,
            'recorded_first_proxy_failure_time_s':float(data['state_time_s'][firstbad]),
            'first_candidate_intervention':first,
            'candidate_intervenes_before_recorded_failure':bool(first and first['before_physics_event']<firstbad),
            'all_real_FIFO6_applied_commands_reconstructed':True,
            'actual_state_and_velocity_exactly_bound_to_previous_native_event':True,
            'decisions':decisions,'forecast_tracking_braking_bounds_qualified':False,
            'native_counterfactual_executed':False,'physical_safety_prevention_proven':False,
            'new_physical_success_count':0,'learnedSystem0_active':False,'fullSystem0_accepted':False,
            'source_sha256':{**sources,str(Path(__file__)):sha(__file__),str(P/'actual_state_arm_guard_v1.py'):sha(P/'actual_state_arm_guard_v1.py')}}
    result.write_text(json.dumps(output,indent=2)+'\n')
    print('RECORDED108_REPLAY',output['candidate_intervenes_before_recorded_failure'],
          'first',first,'recorded_failure',firstbad,'native_counterfactual0',flush=True)


if __name__=='__main__':main()
