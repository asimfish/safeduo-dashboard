"""Independent prospective CPU oracle, not an implemented producer repair.

The reference is the fixed AABB center of the recorded native framing-point set,
the same definition used by the immutable old independent frustum reader.
Adaptive optical targets do not move this reference between selected views.
"""
import itertools
import math
import numpy as np
from evidence_io import require


def circular_error(a,b):
    require(math.isfinite(a) and math.isfinite(b),'finite angles')
    return (float(a)-float(b)+180.)%360.-180.


def separated(angles):
    require(len(angles)==3 and all(math.isfinite(x) for x in angles),'three finite angles')
    return all(abs(circular_error(a,b))>=30. for a,b in itertools.combinations(angles,2))


def measured(camera,points):
    world=np.asarray(camera['actual_camera_to_world_row_matrix'],dtype=np.float64)
    points=np.asarray(points,dtype=np.float64)
    require(world.shape==(4,4) and np.isfinite(world).all(),'finite camera4x4')
    require(points.ndim==2 and points.shape[1]==3 and len(points)>0 and np.isfinite(points).all(),'finite native framing points')
    require(np.allclose(world[:,3],[0,0,0,1],atol=1e-9,rtol=0) and
        np.allclose(world[:3,:3]@world[:3,:3].T,np.eye(3),atol=1e-6,rtol=0) and np.linalg.det(world[:3,:3])>0,'proper row pose')
    center=(points.min(axis=0)+points.max(axis=0))/2
    vector=world[3,:3]-center;rho=math.hypot(float(vector[0]),float(vector[1]))
    require(rho>1e-9,'azimuth undefined at vertical; unchanged old guard')
    azimuth=math.degrees(math.atan2(float(vector[1]),float(vector[0])))%360
    return dict(actual_azimuth_deg=azimuth,azimuth_reference_world_m=center.tolist(),horizontal_baseline_m=rho,
        condition=np.linalg.norm(vector).item()/rho,angle_source='FINAL_ACTUAL_USD_MATRIX_FIXED_NATIVE_FRAMING_CENTER',
        original_reader_identity_tolerance_deg=1e-4,minimum_pairwise_separation_deg=30.)


def prospective_record(rec,points):
    """In-memory oracle prediction only; never writes an actual producer record."""
    expected=measured(rec['camera'],points)
    direction=np.asarray(rec['world_direction'],float)
    require(direction.shape==(3,) and np.isfinite(direction).all() and np.linalg.norm(direction[:2])>0,'finite requested direction')
    requested=math.degrees(math.atan2(float(direction[1]),float(direction[0])))%360
    return dict(expected,requested_azimuth_deg=requested,azimuth_deg=expected['actual_azimuth_deg'],
        prediction_only=True,native_pass=False,safety_acceptance=False)


def verify_candidate(rec,points):
    expected=measured(rec['camera'],points)
    require(abs(circular_error(rec['azimuth_deg'],expected['actual_azimuth_deg']))<1e-4,'unchanged actual-angle identity1e-4deg')
    require(np.array_equal(rec['azimuth_reference_world_m'],expected['azimuth_reference_world_m']),'native angle reference identity')
    return expected


def selected_actual(records,points):
    require(len(records)==3 and all(type(r['attempt']) is int for r in records) and len({r['attempt'] for r in records})==3,'three distinct attempts')
    expected=[verify_candidate(r,points) for r in records]
    # Qualification is based on independent actual angles, never nominal fields
    # or angles admitted by an identity comparison tolerance.
    return separated([r['actual_azimuth_deg'] for r in expected])
