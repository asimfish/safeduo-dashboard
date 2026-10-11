"""Controller estimate includes existing native rotor inertia, without any write."""
import numpy as np


def original_armature_mass(native_mass, native_armature):
    mass = np.asarray(native_mass, dtype=np.float64)
    armature = np.asarray(native_armature, dtype=np.float64)
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1]:
        raise ValueError('Exact square native link inertia required')
    if armature.shape != (len(mass),):
        raise ValueError('Native DOF armature readback required, without broadcasting')
    if not np.isfinite(mass).all() or not np.isfinite(armature).all():
        raise ValueError('Finite unmodified native parameters required')
    if (armature < 0).any():
        raise ValueError('Negative original armature is invalid')
    result = mass.copy()
    result[np.diag_indices(len(mass))] += armature
    return result
