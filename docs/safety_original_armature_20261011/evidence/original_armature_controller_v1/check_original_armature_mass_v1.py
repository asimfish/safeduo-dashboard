"""Rotor kinetic energy and inverse-dynamics checks, not native safety evidence."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from original_armature_mass_v1 import original_armature_mass

P = Path(__file__).resolve().parent
H = P.parent.parent
sys.path.insert(0, str(H / 'spike/full_motor_implicit_effort_v1'))
from coupled_implicit_motor_effort_v1 import coupled_backward_euler_torque


def main():
    reg = json.loads((P/'REGISTRATION_V1.json').read_text())
    for file, digest in reg['source_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
    rng = np.random.default_rng(reg['seed'])
    cases = 500
    max_energy = max_balance = 0.
    for i in range(cases):
        n = int(rng.integers(2, 20)); k = int(rng.integers(1, n+1))
        Q, _ = np.linalg.qr(rng.normal(size=(n,n)))
        M = Q @ np.diag(10**rng.uniform(-7, 1, n)) @ Q.T
        T = np.eye(n)[:, :k] + rng.normal(size=(n,k)) * .02
        A = rng.uniform(0, .05, n)
        if i % 5 == 0: A[:] = 0
        before_M, before_A = M.copy(), A.copy()
        estimate = original_armature_mass(M, A)
        np.testing.assert_array_equal(M, before_M)
        np.testing.assert_array_equal(A, before_A)
        assert not np.shares_memory(estimate, M)
        np.testing.assert_array_equal(estimate - np.diag(np.diag(estimate)), M - np.diag(np.diag(M)))
        if not A.any(): np.testing.assert_array_equal(estimate, M)
        v = rng.normal(size=k)
        rotor_energy = .5 * np.sum(A * (T @ v)**2)
        reduced = T.T @ estimate @ T
        original_reduced = T.T @ M @ T
        energy = .5 * v @ (reduced-original_reduced) @ v
        max_energy = max(max_energy, abs(energy-rotor_energy))
        np.testing.assert_allclose(energy, rotor_energy, atol=2e-13, rtol=2e-12)
        K = rng.choice([5.,400.,1200.], k); D = rng.choice([.5,80.], k)
        q, target = rng.normal(size=(2,k)); dt = .00208325
        answer = coupled_backward_euler_torque(estimate,T,target,q,v,K,D,dt)
        # Independent impulse balance includes the rotor contribution separately.
        delta = answer.predicted_free_velocity-v
        impulse = original_reduced @ delta + T.T @ (A * (T @ delta))
        error = float(abs(impulse-dt*answer.torque).max())
        scale = max(float(abs(impulse).max()),float(abs(dt*answer.torque).max()),1.)
        max_balance = max(max_balance,error/scale)
        assert error/scale < 1e-11
    invalid = [np.ones(3), np.ones((2,1)), np.array([-1.,0.]),
               np.array([np.nan,0.]),np.array([np.inf,0.])]
    for A in invalid:
        try: original_armature_mass(np.eye(2),A)
        except ValueError: pass
        else: raise AssertionError('Invalid original armature accepted')
    result = dict(status='PASS_ORIGINAL_ROTOR_KINETIC_ENERGY_AND_INDEPENDENT_IMPULSE_BALANCE',
                  cases=cases, seed=reg['seed'], invalid_cases=len(invalid),
                  max_rotor_energy_identity_error_J=max_energy,
                  max_relative_independent_impulse_error=max_balance,
                  native_simulations=0, fullSystem0_accepted=False)
    out=P/'SOFTWARE_RESULT_V1.json'; assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n'); print(result,flush=True)


if __name__ == '__main__': main()
