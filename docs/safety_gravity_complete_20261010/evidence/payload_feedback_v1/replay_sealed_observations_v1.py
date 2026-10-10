"""Retrospective captured-input task observer check; zero new physical trials."""
import hashlib
import json
from pathlib import Path

import numpy as np

from payload_phases_v1 import Observation, PayloadPhases

P = Path(__file__).resolve().parent
B = Path('/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006')
RAW = Path('/mnt/nas/data/lyf/double_hand/safety_object_binding_20261006/native_v5')
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
OBJECTS = ('beam700', 'beam300')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result_path = P / 'CAPTURED_REPLAY_RESULT_V1.json'
    assert not result_path.exists()
    rec = json.loads((RAW / 'recording_receipt.json').read_text())
    initial = json.loads((RAW / 'native_initial.json').read_text())
    reg = json.loads((B / 'REGISTRATION_V5.json').read_text())
    curves_path = B / 'native_curves.npz'
    with np.load(curves_path, allow_pickle=False) as z:
        curves = {key: z[key].copy() for key in z.files}
    postures = {a: [] for a in ARMS}
    source_sha = {str(curves_path): sha(curves_path), str(RAW / 'native_initial.json'): sha(RAW / 'native_initial.json'),
                  str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                  str(P / 'payload_phases_v1.py'): sha(P / 'payload_phases_v1.py')}
    for chunk in rec['chunks']:
        file = RAW / chunk['file']
        assert sha(file) == chunk['sha256']
        source_sha[str(file)] = chunk['sha256']
        with np.load(file, allow_pickle=False) as z:
            for arm in ARMS:
                postures[arm].append(z[arm + ':hand_q'].copy())
    postures = {a: np.concatenate(v) for a, v in postures.items()}
    time = curves['time']
    steps = np.flatnonzero(time >= 4.6)
    rows = []
    for env_id in range(8):
        for oi, obj in enumerate(OBJECTS):
            c = PayloadPhases(obj, initial['task_input'][env_id][oi][:3], reg['goals_local_m'][oi][:2],
                              reg['sizes_m'][oi], initial['native_mass_kg'][obj][env_id][0])
            first_failure = None
            for sequence, step in enumerate(steps):
                state = curves['states'][step, env_id, oi]
                open_error = tuple(float(abs(postures[a][step, env_id] - postures[a][0, env_id]).max())
                                   for a in ARMS[2*oi:2*oi+2])
                obs = Observation(obj, sequence, float(time[step]), float(time[step]), True,
                    tuple(state[:3]), tuple(state[3:7]), tuple(state[7:10]), tuple(state[10:13]),
                    tuple(reg['sizes_m'][oi]), tuple(curves['hand_normal'][step, env_id, oi]),
                    float(curves['own_table_normal'][step, env_id, oi]), open_error)
                intent = c.update(obs)
                if intent.phase == 'FAILED':
                    first_failure = dict(native_step=int(step), state_time_s=float(time[step]), reason=intent.failure)
                    break
                if intent.phase == 'DONE':
                    break
            rows.append(dict(env=env_id, object=obj, final_observer_phase=c.phase,
                             first_failure=first_failure, transitions=c.transitions))
    assert len(rows) == 16
    assert all(row['final_observer_phase'] != 'DONE' for row in rows), 'old records cannot establish full task success'
    result = dict(status='PASS_CAPTURED_INPUT_REJECTION_OF_INCOMPLETE_TASKS', objects=16,
        observer_done=0, new_physics_trials=0, old_task_verdicts_unchanged=True, cases=rows,
        source_sha256=source_sha, full_system0_accepted=False,
        scope='Known sealed last-substep observations; open posture measured relative to recorded initial pose '
              'is only a proxy. No full friction/geometry/task/vision or new native feedback outcome claim.')
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], cases=rows), indent=2))


if __name__ == '__main__':
    main()
