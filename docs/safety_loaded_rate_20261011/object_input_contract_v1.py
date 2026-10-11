"""Candidate contract for fresh native object inputs, independent of policy ABI.

This module does not run a robot, authenticate a sensor, infer grasp contact,
or claim that requesting a stop is physically safe. It is not installed in
the frozen native60 worker. Camera estimation needs a separate adapter.
"""
from dataclasses import dataclass
import math
import numpy as np

SCHEMA = 'safeduo.native_object_input.v1'
FEATURE_ORDER = ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz',
                 'size_x', 'size_y', 'size_z', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz')


class ObjectInputUnavailable(ValueError):
    """Reject an incomplete, ambiguous, or stale object observation."""


@dataclass(frozen=True)
class ObjectFrame:
    names: tuple
    physics_event: int
    state_time_s: float
    root_state: np.ndarray  # K,13: position, wxyz quaternion, linear/angular velocity
    size_m: np.ndarray      # K,3 full local extents
    source: str = 'simulator_ground_truth'
    coordinate_frame: str = 'world'
    quaternion_order: str = 'wxyz'
    extent_convention: str = 'full'
    length_unit: str = 'metre'


@dataclass(frozen=True)
class ObjectInputDecision:
    features: np.ndarray | None
    lock_request: tuple
    reason: str | None
    schema: str = SCHEMA
    actor_input_modified: bool = False
    physical_stop_verified: bool = False


class FreshNativeObjectInput:
    """Every accepted packet must belong to the current physics event.

    Returned features own their data and use the same 16-value order as the
    existing ObjectContext.features. A rejected packet cannot advance history.
    """
    def __init__(self, expected_names, *, max_age_s=.05, clock_tolerance_s=1e-9):
        names = tuple(expected_names)
        if not names or len(names) != len(set(names)) or any(type(x) is not str or not x for x in names):
            raise ValueError('Expected a fixed nonempty inventory of unique names')
        if isinstance(max_age_s, bool) or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        if isinstance(clock_tolerance_s, bool) or not math.isfinite(clock_tolerance_s) or clock_tolerance_s < 0:
            raise ValueError('clock_tolerance_s must be finite and nonnegative')
        self.names = names
        self.max_age_s = max_age_s
        self.clock_tolerance_s = clock_tolerance_s
        self._last_event = None
        self._last_time = None

    def consume(self, packet, *, expected_event, now_s):
        if type(expected_event) is not int or expected_event < 0 or isinstance(now_s, bool) or not math.isfinite(now_s):
            raise ValueError('Consumer event and clock must be valid')
        if not isinstance(packet, ObjectFrame):
            raise ObjectInputUnavailable('MISSING_OBJECT_FRAME')
        if type(packet.names) is not tuple or packet.names != self.names:
            raise ObjectInputUnavailable('OBJECT_INVENTORY_MISMATCH')
        if (packet.source, packet.coordinate_frame, packet.quaternion_order,
            packet.extent_convention, packet.length_unit) != (
                'simulator_ground_truth', 'world', 'wxyz', 'full', 'metre'):
            raise ObjectInputUnavailable('UNSUPPORTED_SOURCE_FRAME_OR_UNITS')
        if type(packet.physics_event) is not int or packet.physics_event != expected_event:
            raise ObjectInputUnavailable('OBJECT_EVENT_MISMATCH')
        if isinstance(packet.state_time_s, bool) or not isinstance(packet.state_time_s, (float, int)) or not math.isfinite(packet.state_time_s):
            raise ObjectInputUnavailable('NONFINITE_OBJECT_TIMESTAMP')
        age = now_s - packet.state_time_s
        if age < -self.clock_tolerance_s:
            raise ObjectInputUnavailable('FUTURE_OBJECT_STATE')
        if age > self.max_age_s + self.clock_tolerance_s:
            raise ObjectInputUnavailable('STALE_OBJECT_STATE')
        if self._last_event is not None and packet.physics_event <= self._last_event:
            raise ObjectInputUnavailable('REORDERED_OR_REPEATED_OBJECT_EVENT')
        if self._last_time is not None and packet.state_time_s <= self._last_time:
            raise ObjectInputUnavailable('OBJECT_CLOCK_NOT_ADVANCING')
        k = len(self.names)
        arrays = []
        for value, shape in ((packet.root_state, (k, 13)), (packet.size_m, (k, 3))):
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind != 'f':
                raise ObjectInputUnavailable('OBJECT_SHAPE_OR_DTYPE_MISMATCH')
            copy = np.array(value, dtype=np.float64, copy=True)
            if not np.isfinite(copy).all():
                raise ObjectInputUnavailable('NONFINITE_OBJECT_VALUES')
            arrays.append(copy)
        state, size = arrays
        if np.any(size <= 0):
            raise ObjectInputUnavailable('INVALID_FULL_OBJECT_EXTENT')
        q = state[:, 3:7]
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        if not np.isfinite(norm).all() or np.any(norm < 1e-8):
            raise ObjectInputUnavailable('INVALID_OBJECT_QUATERNION')
        features = np.concatenate((state[:, :3], q / norm, size, state[:, 7:13]), axis=1)
        if not np.isfinite(features).all():
            raise ObjectInputUnavailable('NONFINITE_DERIVED_OBJECT_FEATURES')
        features.flags.writeable = False
        self._last_event, self._last_time = packet.physics_event, packet.state_time_s
        return features

    def assess(self, packet, *, expected_event, now_s):
        try:
            features = self.consume(packet, expected_event=expected_event, now_s=now_s)
        except ObjectInputUnavailable as error:
            return ObjectInputDecision(None, (True, True, True, True), str(error))
        return ObjectInputDecision(features, (False, False, False, False), None)
