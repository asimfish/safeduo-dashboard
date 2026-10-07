"""The per-object acceptance CLI must reject absent target telemetry."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CHECKER = Path(__file__).resolve().parents[3] / 'tools/r35_multi_object_check.py'


@pytest.mark.parametrize('targets,seen,expected_rc', [
    (['a', 'b'], [], 1), (['a', 'b'], ['a'], 1),
    (['a', 'b'], ['a', 'b'], 0), ([], [], 1),
])
def test_every_target_requires_telemetry(tmp_path, targets, seen, expected_rc):
    objects = [{'name': n, 'place_xy': [0.2, 0.0], 'size': [0.1]*3} for n in targets]
    traj = tmp_path / 'task.npz'
    np.savez(traj, meta=json.dumps({'s9_task': {'objects': objects}}))
    log = tmp_path / 'probe.log'
    lines = []
    for name in seen:
        lines += [f'GRASP_TELE t=0.0 phase=descend obj={name} z=0.85 xy=(0.0,0.0)',
                  f'GRASP_TELE t=1.0 phase=lift obj={name} z=0.95 xy=(0.0,0.0)',
                  f'GRASP_TELE t=2.0 phase=release obj={name} z=0.85 xy=(0.2,0.0)']
    log.write_text('\n'.join(lines))
    result = subprocess.run([sys.executable, str(CHECKER), str(log), str(traj)], capture_output=True, text=True)
    assert result.returncode == expected_rc, result.stdout + result.stderr


@pytest.mark.parametrize('quat,expected_rc', [
    ('[1,0,0,0]', 0), ('[2,0,0,0]', 0),
    ('[0.707107,0.707107,0,0]', 1), ('[0,1,0,0]', 1),
    ('[0,0,0,0]', 1), ('[1,0,0]', 1), ('[NaN,0,0,0]', 1),
    ('broken', 1),
])
def test_pose_rejects_tipped_object_even_at_start_height(tmp_path, quat, expected_rc):
    traj = tmp_path / 'task.npz'
    np.savez(traj, meta=json.dumps({'s9_task': {'objects': [
        {'name': 'beam', 'place_xy': [0.2, 0.0]}]}}))
    log = tmp_path / 'pose.log'
    log.write_text('GRASP_TELE t=0 phase=descend obj=beam z=0.85 xy=(0,0)\n'
                   'GRASP_TELE t=1 phase=lift obj=beam z=0.95 xy=(0,0)\n'
                   f'GRASP_TELE t=2 phase=done obj=beam z=0.85 xy=(0.2,0) quat_wxyz={quat}\n')
    result = subprocess.run([sys.executable, str(CHECKER), str(log), str(traj)],
                            capture_output=True, text=True)
    assert result.returncode == expected_rc, result.stdout + result.stderr
    assert 'orientation=quaternion' in result.stdout


def test_final_pose_cannot_disappear(tmp_path):
    traj = tmp_path / 'task.npz'
    np.savez(traj, meta=json.dumps({'s9_task': {'objects': [
        {'name': 'beam', 'place_xy': [0.2, 0.0]}]}}))
    log = tmp_path / 'pose.log'
    log.write_text('GRASP_TELE t=0 phase=descend obj=beam z=0.85 xy=(0,0) quat_wxyz=[1,0,0,0]\n'
                   'GRASP_TELE t=1 phase=lift obj=beam z=0.95 xy=(0,0)\n'
                   'GRASP_TELE t=2 phase=done obj=beam z=0.85 xy=(0.2,0)\n')
    result = subprocess.run([sys.executable, str(CHECKER), str(log), str(traj)],
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'tilt_deg=missing' in result.stdout
