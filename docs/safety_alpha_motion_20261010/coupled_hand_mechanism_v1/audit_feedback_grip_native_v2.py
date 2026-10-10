"""Independent raw-state reconstruction of prospective hand feedback decisions.

This does not import the candidate scheduler. Decision fidelity and physical
acceptance are separate outputs; a matched guard can still fail in physics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

H=Path(__file__).resolve().parent
ARMS=('F_L','F_R','U_L','U_R')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(version):
    P=H/f'fourhand_native_probe_v{version}'
    independent=json.loads((P/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text())
    assert independent['record_integrity_verified']
    reg=json.loads((P/'REGISTRATION_V1.json').read_text());settings=reg['feedback_grip_settings'];dt=reg['control_dt_s']
    raw=Path(json.loads((P/'RESOURCE_EXECUTION_V1.json').read_text())['out']);params=np.load(raw/'native_parameters.npz')
    poses=json.loads(Path(reg['calibrated_hand_poses']).read_text())
    keys=['physics_event','phase','four_hand_normal_N','feedback_grip_mode','feedback_grip_finger_fraction',
          'feedback_grip_thumb_fraction','feedback_grip_thumb_guard','feedback_grip_finger_guard']
    keys += [a+s for a in ARMS for s in ['_native_q','_native_qd','_actual_full_position_target','_hand_nominal_position_target','_hand_filtered_position_target']]
    pieces={k:[] for k in keys};chunks=sorted(raw.glob('physics_*.npz'))
    for path in chunks:
        with np.load(path) as z:
            for k in keys:pieces[k].append(z[k].copy())
    data={k:np.concatenate(v) for k,v in pieces.items()}
    assert np.array_equal(data['physics_event'],np.arange(independent['physics_events']))
    history={};last_modes={};records=[];stats={a:{m:0 for m in ['SEEK','HOLD','RELAX','JOINT_GUARD']} for a in ARMS}
    for event in range(2,len(data['physics_event']),2):
        if str(data['phase'][event]) not in ['CLOSE','LIFT','CARRY','PLACE']:continue
        assert data['phase'][event]==data['phase'][event+1]
        for ai,arm in enumerate(ARMS):
            names=params[arm+'_joint_names'];hand=np.flatnonzero(np.array(['thumb' in n or any(k in n for k in ['index','middle','ring','pinky','little']) for n in names]));assert len(hand)==12
            thumb=np.array(['thumb' in names[i] for i in hand]);q=data[arm+'_native_q'][event-1,0,hand];v=data[arm+'_native_qd'][event-1,0,hand]
            limits=params[arm+'_hard_limits'][0,hand];frame=poses[arm][str(reg['case'])+':'+str(reg['pose_variants'][arm])]['notes']['hand_frame']
            caps=[frame['f_squeeze'],frame['f_squeeze'] if frame['f_squeeze_thumb'] is None else frame['f_squeeze_thumb']]
            finger=float(data['feedback_grip_finger_fraction'][event-1,ai]);thumb_frac=float(data['feedback_grip_thumb_fraction'][event-1,ai])
            force=float(data['four_hand_normal_N'][event-1,ai])
            old=history.get(arm)
            if old is None:inward=v;outward=v
            else:
                previous_event,previous_q=old;assert event-previous_event==2
                finite_difference=(q-previous_q)/dt;inward=np.minimum(v,finite_difference);outward=np.maximum(v,finite_difference)
            lo=q-limits[:,0]+np.minimum(inward,0)*settings['prediction_s']
            hi=limits[:,1]-q-np.maximum(outward,0)*settings['prediction_s']
            danger=((lo<settings['joint_reserve_rad'])&(inward<-.02))|((hi<settings['joint_reserve_rad'])&(outward>.02))
            tg=bool(danger[thumb].any());fg=bool(danger[~thumb].any())
            if tg or fg:
                mode='JOINT_GUARD';fd=-settings['relax_rate']*dt if fg else 0.;td=-settings['relax_rate']*dt if tg else 0.
            elif force>=settings['relax_above_N']:
                mode='RELAX';fd=td=-settings['relax_rate']*dt
            elif force>=settings['hold_above_N'] or (last_modes.get(arm)=='HOLD' and force>=settings['resume_below_N']):
                mode='HOLD';fd=td=0.
            else:mode='SEEK';fd=td=settings['seek_rate']*dt
            expected=[min(caps[0],max(0.,finger+fd)),min(caps[1],max(0.,thumb_frac+td))]
            actual=[float(data['feedback_grip_finger_fraction'][event,ai]),float(data['feedback_grip_thumb_fraction'][event,ai])]
            assert np.allclose(actual,expected,atol=1.1e-6,rtol=0),(event,arm,actual,expected)
            for e in [event,event+1]:
                assert str(data['feedback_grip_mode'][e,ai])==mode,(e,arm,mode,data['feedback_grip_mode'][e,ai])
                assert bool(data['feedback_grip_thumb_guard'][e,ai])==tg and bool(data['feedback_grip_finger_guard'][e,ai])==fg
                openq=params[arm+'_initial_q'][0,hand];soft=params[arm+'_soft_joint_limits'][0,hand]
                far=np.where(abs(soft[:,1]-openq)>=abs(soft[:,0]-openq),soft[:,1],soft[:,0])
                progress=np.where(thumb,actual[1],actual[0]);targets=openq+progress*(far-openq)
                nominal=data[arm+'_hand_nominal_position_target'][e]
                assert np.max(abs(targets-nominal))<1e-6,(e,arm,'nominal hand target not bound to progress')
                previous=data[arm+'_actual_full_position_target'][event-1,0,hand]
                desired=np.clip(nominal,limits[:,0],limits[:,1])
                distal=np.array([('thumb_3_joint' in names[i] or 'thumb_4_joint' in names[i]) for i in hand])
                spec=reg['interior_hand_target_settings']
                floors=limits[:,0]+np.minimum(spec['distal_lower_reserve_rad'],.2*(limits[:,1]-limits[:,0]))
                desired=np.where(distal,np.maximum(desired,floors),desired)
                step=spec['max_target_rate_rad_s']*dt
                expected_target=previous+np.clip(desired-previous,-step,step)
                actual_filtered=data[arm+'_hand_filtered_position_target'][e]
                measured=data[arm+'_actual_full_position_target'][e,0,hand]
                assert np.max(abs(expected_target-actual_filtered))<1e-6,(e,arm,'independent target filter mismatch')
                assert np.max(abs(actual_filtered-measured))<1e-6,(e,arm,'filtered target not applied natively')
            history[arm]=(event,q.copy());last_modes[arm]=mode;stats[arm][mode]+=1
            records.append(dict(control=event//2,arm=arm,mode=mode,normal_N=force,thumb_guard=tg,finger_guard=fg,
                predicted_lower_margin_rad=float(lo.min()),predicted_upper_margin_rad=float(hi.min()),
                finger_fraction=actual[0],thumb_fraction=actual[1]))
    paths=[Path(__file__),P/'REGISTRATION_V1.json',P/'INDEPENDENT_CAPTURE_AUDIT_V1.json',raw/'native_parameters.npz',*chunks]
    report=dict(status='PASS_RECONSTRUCTED_NATIVE_FEEDBACK_AND_INTERIOR_TARGET_BINDING',version=version,
        decision_records=len(records),states_reconstructed_from_previous_actual_microstep=True,
        force_and_finite_position_drift_used=True,native_hand_position_targets_verified=True,mode_counts=stats,
        physical_gates_pass=independent['physical_gates_pass'],task_pass=independent['four_hand_task_development_pass'],
        fullSystem0_accepted=False,guard_output_does_not_prove_safe_stop=True,
        records=records,source_sha256={str(p):sha(p) for p in paths})
    target=P/'INDEPENDENT_FEEDBACK_GRIP_AUDIT_V2.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(records),'physical',report['physical_gates_pass'],'task',report['task_pass'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int);main(parser.parse_args().version)
