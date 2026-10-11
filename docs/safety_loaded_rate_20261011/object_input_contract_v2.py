"""Candidate native-only size and event-clock binding, preserving v1 evidence."""
import math
from dataclasses import replace
import numpy as np
from object_input_contract_v1 import FreshNativeObjectInput, ObjectFrame, ObjectInputUnavailable

SCHEMA = 'safeduo.registered_native_object_input.v2'


class RegisteredNativeObjectInput(FreshNativeObjectInput):
    def __init__(self, expected_names, registered_sizes_m, *, physics_dt_s,
                 max_age_s=.05, clock_tolerance_s=1e-9):
        super().__init__(expected_names, max_age_s=max_age_s, clock_tolerance_s=clock_tolerance_s)
        sizes = np.asarray(registered_sizes_m)
        if sizes.shape != (len(self.names), 3) or sizes.dtype.kind not in 'fi':
            raise ValueError('Registered full metre extents must match the inventory')
        sizes = sizes.astype(np.float64, copy=True)
        if not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError('Registered extents must be finite and positive')
        if isinstance(physics_dt_s, bool) or not math.isfinite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError('A positive finite registered native physics step is required')
        self._sizes64 = sizes
        self._sizes32 = sizes.astype(np.float32).astype(np.float64)
        if not np.isfinite(self._sizes32).all() or np.any(self._sizes32 <= 0):
            raise ValueError('Registered extents are not representable by native float32')
        self._sizes64.flags.writeable = self._sizes32.flags.writeable = False
        self._physics_dt_s = float(physics_dt_s)

    def consume(self, packet, *, expected_event, now_s):
        # Check additional bindings before base consume can advance history.
        if isinstance(packet, ObjectFrame):
            sizes = packet.size_m
            if isinstance(sizes, np.ndarray) and sizes.shape == self._sizes64.shape and sizes.dtype.kind == 'f' and np.isfinite(sizes).all():
                observed = sizes.astype(np.float64)
                if not np.all((observed == self._sizes64) | (observed == self._sizes32)):
                    raise ObjectInputUnavailable('REGISTERED_OBJECT_SIZE_CHANGED')
            stamp = packet.state_time_s
            if type(expected_event) is int and expected_event >= 0 and not isinstance(stamp, bool) and isinstance(stamp, (int, float)) and math.isfinite(stamp):
                expected_time = (expected_event + 1) * self._physics_dt_s
                if abs(stamp - expected_time) > self.clock_tolerance_s:
                    raise ObjectInputUnavailable('OBJECT_TIMESTAMP_EVENT_MISMATCH')
        return super().consume(packet, expected_event=expected_event, now_s=now_s)

    def assess(self, packet, *, expected_event, now_s):
        return replace(super().assess(packet, expected_event=expected_event, now_s=now_s), schema=SCHEMA)
