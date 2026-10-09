"""Freeze stronger future random inputs before any new physical/controller outcomes."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

P=Path(__file__).resolve().parent
H=P.parent
OUT=Path('/mnt/nas/data/lyf/double_hand/safety_physical_qualified_random_20261009_1613/strong_random_recipe_v1')
ARMS=('F_L','F_R','U_L','U_R')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (P/'REGISTRATION_V1.json').exists(),'Preserve first frozen input preparation'
    reg=json.loads((H/'FIXEDHAND_TRIPLET_DEV_REG_V1.json').read_text())
    params=Path(reg['native_namespace'])/'paired_batch0_raw_v8/resolved_native_parameters.npz'
    with np.load(params) as z:
        boxes=[];speeds=[];names=[]
        for a in ARMS:
            assert np.array_equal(z[a+'_soft_limits'],np.broadcast_to(z[a+'_soft_limits'][0],z[a+'_soft_limits'].shape))
            idx=z[a+'_controlled_joint_indices'];boxes.append(z[a+'_soft_limits'][0]);speeds.append(z[a+'_max_velocity'][0,idx]);names.extend([a+'/'+str(v) for v in z[a+'_joint_names'][idx]])
        box=np.concatenate(boxes);vmax=np.concatenate(speeds)
    proposal_seed=1920573461;velocity_seed=740913527;command_seed=1046309821
    qunit=np.random.default_rng(proposal_seed).random((32000,26),dtype=np.float32)
    proposals=box[:,0]+qunit*(box[:,1]-box[:,0])
    np.savez_compressed(OUT/'initial_proposals.npz',proposal_id=np.arange(32000),unit_cube=qunit,controlled_q=proposals,soft_box_rad=box,joint_names=np.array(names))
    velocity_unit=np.random.default_rng(velocity_seed).uniform(-1,1,(512,26)).astype(np.float32)
    cells=np.arange(512)%16;fractions=np.array([0,.5,.75,1],np.float32)[cells//4];modes=cells%4
    qd=velocity_unit*fractions[:,None]*vmax
    cadence=np.array([8,8,1,32])[modes]
    # Reserve command units for every control before geometry qualification; no fresh RNG later.
    independent=np.random.default_rng(command_seed).random((480,512,26),dtype=np.float32)
    tape=np.empty_like(independent)
    for lane in range(512):
        steps=(np.arange(480)//cadence[lane])*cadence[lane]
        tape[:,lane]=independent[steps,lane]
    np.savez_compressed(OUT/'commands_velocity.npz',cell_id=cells,velocity_fraction=fractions,velocity_unit=velocity_unit,controlled_initial_qd=qd,native_vmax_rad_s=vmax,command_mode=modes,refresh_controls=cadence,command_unit=tape,soft_box_rad=box)
    cell_rows=[dict(cell_id=c,velocity_fraction=float(np.array([0,.5,.75,1])[c//4]),command_mode=['local_pm035_rad','full_soft_box_refresh8','full_soft_box_refresh1','full_soft_box_refresh32'][c%4],refresh_controls=int(np.array([8,8,1,32])[c%4]),geometry_bank_slots=32,maximum_long_window_slots=8) for c in range(16)]
    registration=dict(status='FROZEN_FUTURE_INPUTS_ONLY_NOT_NATIVE_TEST_RESULTS',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seeds=dict(initial_proposal=proposal_seed,velocity=velocity_seed,command=command_seed),proposal_count=32000,geometry_bank_max=512,steps=480,microsteps=960,cells=cell_rows,
        deterministic_target_rule='local mode: clip(initial_q+(2*unit-1)*.35,soft_box); full-box modes: lower+unit*(upper-lower); all26 coordinates independently sampled before cadence repetition',
        native_parameters_source=str(params),native_parameters_sha256=sha(params),source_sha256=sha(Path(__file__)),numpy_version=np.__version__,
        files={str(p):sha(p) for p in sorted(OUT.glob('*.npz'))},
        future_geometry_rule='Evaluate every one of32000 proposals under the frozen native0.03hand/64position physics; record all rejections; first up to512 with raw9021initial gap>=.0001; no regeneration on shortfall',
        future_prefix_rule='Run registered velocities and held initial targets for6controls/12physics on all up to512; joint+primary criterion, record all rejects; first up to8 within each cell before controller outcomes; never fill shortages after outcomes',
        future_long_rule='Matched raw/multirow/initial-hold on every selected state; fullreference tape and full74prefix identity required; retain replay rejects in original selected denominator',
        score_gates=dict(raw_geometry_m=0,point_normal_scalar_N=.1,all74_hard_rad=1e-5,all74_speed_rad_s=1e-5),
        trained_actor=False,objects_or_task_success=False,root_or_hand_randomized=False,formal_holdout=False,native_jobs_executed=0,coverage_claim='Proposal/command inputs only, no reachable/safe volume or native safety inference',
        required_before_execution='Bind a fresh native execution plan and actual limit/fixture readbacks; no config changes during this registered future challenge')
    (P/'REGISTRATION_V1.json').write_text(json.dumps(registration,indent=2,allow_nan=False)+'\n')
    span=(proposals.max(0)-proposals.min(0))/(box[:,1]-box[:,0])
    assertions=dict(status='PASS_FUTURE_RANDOM_INPUT_SUPPORT_ONLY',all_proposals_inside_native_soft_box=bool(((proposals>=box[:,0])&(proposals<=box[:,1])).all()),initial_marginal_span_fraction=[float(span.min()),float(span.max())],unit_target_shape=list(tape.shape),velocity_peak_fraction=float((abs(qd)/vmax).max()),registered_cells=16,velocity_fractions=[0,.5,.75,1],full_box_modes=3,maximum_refresh_every_control=True,native_tests_completed=0)
    assert assertions['all_proposals_inside_native_soft_box']
    assert set(cells)==set(range(16)) and all(int((cells==c).sum())==32 for c in range(16))
    for lane in range(512):
        ids=(np.arange(480)//cadence[lane])*cadence[lane]
        assert np.array_equal(tape[:,lane],independent[ids,lane])
    assert np.isfinite(qd).all() and (abs(qd)<=vmax).all()
    (P/'INPUT_PREPARATION_ORACLE_V1.json').write_text(json.dumps(assertions,indent=2)+'\n')
    print(assertions,flush=True)


if __name__=='__main__':main()
