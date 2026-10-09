"""Independent full74/contact/geometry admission; retain all512 original inputs."""
import json
from pathlib import Path
import sys

import numpy as np

P=Path(__file__).resolve().parent
H=P.parent
sys.path.insert(0,str(H))
from analyse_prefix512_v2 import oracle,sha

ARMS=('F_L','F_R','U_L','U_R')


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def main():
    reg=json.loads((P/'REGISTRATION_V1.json').read_text())
    execution=json.loads((H/'strong_prefix512_v3_execution.json').read_text())
    assert execution['status']=='complete' and len(execution['jobs'])==8
    assert all(j['actual_exit']==0 and j['status']=='complete' for j in execution['jobs'])
    bank=load(reg['bank']);recipe=load(reg['recipe']);fixture=load(reg['fixture'])
    rows=[];oracles=[];params=[]
    for batch,job in enumerate(execution['jobs']):
        leaf=Path(job['out']);p=load(leaf/'prefix_inputs.npz');r=load(leaf/'prefix_result.npz');ids=np.arange(batch*64,(batch+1)*64)
        assert np.array_equal(p['global_input_id'],bank['selected_indices'][ids]) and np.array_equal(p['bank_position'],ids)
        assert np.array_equal(p['cell_id'],recipe['cell_id'][ids])
        roots=json.loads((leaf/'actual_solver_roots.json').read_text());assert len(roots)==4 and all(x['position_iterations']==64 and x['velocity_iterations']==0 for x in roots)
        assert np.isfinite(r['all_raw_geometry']).all() and r['all_raw_geometry'].shape==(12,64,9021)
        offset=0
        for arm in ARMS:
            idx=p[arm+'_controlled_joint_indices'];n=len(idx);hand=np.ones(len(p[arm+'_joint_names']),bool);hand[idx]=False
            assert np.array_equal(p[arm+'_initial_q'],bank[arm+'_native_q'][ids])
            assert np.array_equal(p[arm+'_initial_qd'][:,idx],recipe['controlled_initial_qd'][ids,offset:offset+n])
            assert not p[arm+'_initial_qd'][:,hand].any()
            expected=np.broadcast_to(fixture[arm+'_full_hold_target'][0],p[arm+'_full_hold_target'].shape).copy();expected[:,idx]=p[arm+'_initial_q'][:,idx]
            assert np.array_equal(p[arm+'_full_hold_target'],expected)
            offset+=n
        audit=oracle(leaf,r,p);oracles.append(dict(batch=batch,**audit));params.append(p)
        initial=p['initial_all_raw_geometry'].min(-1)>=.0001
        gap=r['all_raw_geometry'].min((0,2));force=r['scalar_normal_max_N'].max(0);hard=r['supplementary_hard_limit_violation_rad'].max(0);speed=r['supplementary_velocity_limit_violation_rad_s'].max(0)
        primary=initial&(gap>=0)&(force<=.1);full=primary&(hard<=1e-5)&(speed<=1e-5)
        assert np.array_equal(full,r['admitted'])
        for lane in range(64):
            rows.append(dict(bank_position=int(ids[lane]),input_id=int(p['global_input_id'][lane]),cell=int(p['cell_id'][lane]),velocity_fraction=float(p['velocity_fraction'][lane]),command_mode=int(p['command_mode'][lane]),initial_replay_geometry_admitted=bool(initial[lane]),primary_prefix_pass=bool(primary[lane]),full74_prefix_pass=bool(full[lane]),minimum_raw_gap_m=float(gap[lane]),peak_normal_N=float(force[lane]),all74_hard_max_rad=float(hard[lane]),all74_speed_max_rad_s=float(speed[lane])))
    cells=[];selected=[]
    for cell in range(16):
        candidates=[r for r in rows if r['cell']==cell];qualified=[r for r in candidates if r['full74_prefix_pass']]
        assert len(candidates)==32
        chosen=qualified[:8];selected.extend(r['bank_position'] for r in chosen)
        cells.append(dict(cell_id=cell,total_original=32,primary_pass=sum(r['primary_prefix_pass'] for r in candidates),full74_pass=len(qualified),prospectively_selected=len(chosen),shortfall_from8=8-len(chosen)))
    result=dict(status='CLOSED_STRONG512_NATIVE_FULL74_WAITING_PREFIX_ONLY',source_states=512,controls_each=6,physics_steps_each=12,velocity_fractions=[0,.5,.75,1],all8_native_exit0=True,all8_rawpoint_full74_oracles_pass=True,all_original_denominators_retained=True,primary_pass=sum(r['primary_prefix_pass'] for r in rows),full74_pass=sum(r['full74_prefix_pass'] for r in rows),all74_hard_bad=sum(r['all74_hard_max_rad']>1e-5 for r in rows),all74_speed_bad=sum(r['all74_speed_max_rad_s']>1e-5 for r in rows),selected_for_future=len(selected),cells=cells,states=rows,oracles=oracles,random_reference_commands_actuated=False,complete_controller_or_object_task_evaluated=False,formal_holdout=False,fullSystem0_accepted=False,registration_sha256=sha(P/'REGISTRATION_V1.json'),analysis_source_sha256=sha(Path(__file__)))
    (P/'PREFIX512_RESULT_V1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    indices=np.array(selected,np.int64)
    np.savez_compressed(P/'prospective_selection_v1.npz',bank_position=indices,global_input_id=bank['selected_indices'][indices],cell_id=recipe['cell_id'][indices])
    print(result['status'],{k:result[k] for k in ['primary_pass','full74_pass','all74_hard_bad','all74_speed_bad','selected_for_future']},cells,flush=True)


if __name__=='__main__':main()
