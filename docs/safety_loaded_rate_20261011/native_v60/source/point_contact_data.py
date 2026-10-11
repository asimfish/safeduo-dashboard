"""CPU-only native contact point custody and cancellation-resistant summaries.

Fast integration (caller owns actual physics dt, frame/substep and native state)::

    data = view.get_contact_data(actual_physics_dt)
    arrays = [x.detach().clone().cpu().numpy() for x in data]
    event = pack_contact_event(*arrays, capacity=CAPACITY)
    # Keep separate streams for views with different sensor/filter identities.
    chunk = pack_contact_events(events_for_one_view)
    np.savez_compressed(path, **chunk)
    events = unpack_contact_events(chunk)
    restored = reconstruct_contact_event(events[0])

Argument order is the installed tensors API order: normal_forces, points,
normals, separations, counts, starts. Data is already converted from impulses
to forces by get_contact_data(dt); this helper does not scale it again.

Float buffers must be NumPy float32/float64 arrays with exact installed shapes:
(capacity,1), (capacity,3), (capacity,3), (capacity,1). Counts/starts must be
nonempty sensor-by-partner native-endian integer arrays (32/64-bit signed/unsigned), never
coerced from floats/bools. Zero-contact pairs keep their original starts, which
must lie in [0,capacity]. Only count>0 slices own points. Overlapping slices and
any reached active buffer boundary abort: truncation cannot then be excluded.

Each packed event retains original counts/starts, native point_indices and
sensor/partner owners; valid raw float values keep their dtype and exact bits.
Unreferenced holes/tails are NOT observations, need not be finite, and are NOT
saved. Reconstruction fills them with NaN and returns an explicit observed_mask.
Chunks use event_offsets for ragged point arrays; native indices remain local
to each event. All fields are ordinary npz-safe arrays; no object/pickle arrays.

partner_abs_normal_sum_N and sensor_abs_normal_sum_N accumulate abs(raw scalar
force) in float64. signed vectors are retained only as a cancellation diagnostic.
Totals are sensor/filter observations; duplicated physical contacts across
reciprocal sensors/overlapping filter identities are not deduplicated here.
Caller must bind exact native sensor/filter identities and actual state/time.

This is measurement infrastructure, not a safety certificate. Old/full128 runs
did not retain these raw point fields: their point-scalar status remains UNKNOWN.
No prior vector-norm gate or frozen recorder/auditor/decision is changed.
"""
from __future__ import annotations

import numpy as np

EVENT_SCHEMA = 'safeduo.native_contact_points.event.v1'
CHUNK_SCHEMA = 'safeduo.native_contact_points.chunk.v1'
RAW_FIELDS = ('normal_forces', 'points', 'normals', 'separations')
INDEX_FIELDS = ('point_indices', 'sensor_indices', 'partner_indices')
SUMMARY_FIELDS = ('partner_abs_normal_sum_N', 'partner_abs_normal_max_N',
                  'sensor_abs_normal_sum_N', 'filter_abs_normal_sum_N',
                  'partner_signed_normal_vector_N')
EVENT_KEYS = {'schema', 'capacity', 'point_count', 'counts', 'starts', *RAW_FIELDS,
              *INDEX_FIELDS, *SUMMARY_FIELDS}
CHUNK_KEYS = (EVENT_KEYS - {'point_count'}) | {'event_offsets', 'event_count'}


class InvalidContactData(ValueError):
    pass


class ContactCapacityReached(InvalidContactData):
    """The active stream touches capacity; completeness is not established."""


def _require(condition, message):
    if not condition:
        raise InvalidContactData(message)


def _array(value, name):
    _require(isinstance(value, np.ndarray) and not np.ma.isMaskedArray(value),
             name+' must be an unmasked NumPy array; no implicit conversion')
    return value


def _capacity(value):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
             'capacity must be an integer, not coerced')
    _require(0 < int(value) <= np.iinfo(np.int64).max, 'positive int64 capacity required')
    return int(value)


def _scalar_int(value, name):
    value = _array(value, name)
    _require(value.shape == () and value.dtype == np.dtype(np.int64), name+' must be scalar int64')
    return int(value)


def _float_buffer(value, name, length, width):
    value = _array(value, name)
    _require(value.dtype in (np.dtype(np.float32), np.dtype(np.float64)), name+' must be float32/float64')
    _require(value.shape == (length, width), name+' shape must be '+str((length, width)))
    return value


def _mapping(counts, starts, capacity):
    """Validate integer intervals before any native point is dereferenced."""
    for name, array in [('counts', counts), ('starts', starts)]:
        _array(array, name)
        _require(array.ndim == 2 and all(n > 0 for n in array.shape), name+' must be nonempty sensor/partner matrix')
        _require(array.dtype.kind in 'iu' and array.dtype.itemsize in (4, 8)
                 and array.dtype.isnative,
                 name+' must have native-endian 32/64-bit integer dtype')
    _require(counts.shape == starts.shape, 'counts/starts shapes differ')
    intervals, total = [], 0
    for pair, (raw_count, raw_start) in enumerate(zip(counts.flat, starts.flat)):
        # Convert only already-validated integers. Python ints prevent unsigned
        # or narrow-integer wraparound in range/sum checks.
        count, start = int(raw_count), int(raw_start)
        _require(count >= 0 and 0 <= start <= capacity, 'negative count/start or start outside capacity')
        _require(count <= capacity-start, 'contact interval out of bounds')
        if count:
            if start+count == capacity:
                raise ContactCapacityReached('active contact interval reaches capacity; truncation unknown')
            intervals.append((start, start+count, pair))
            total += count
    if total >= capacity:
        raise ContactCapacityReached('total contact count reaches capacity; truncation unknown')
    previous_end = 0
    for start, end, _ in sorted(intervals):
        _require(start >= previous_end, 'overlap/duplicate point ownership across sensor/partner pairs')
        previous_end = end
    point = np.empty(total, np.int64)
    sensor = np.empty(total, np.int64)
    partner = np.empty(total, np.int64)
    offset = 0
    for start, end, pair in intervals:
        size = end-start
        point[offset:offset+size] = np.arange(start, end, dtype=np.int64)
        sensor[offset:offset+size], partner[offset:offset+size] = divmod(pair, counts.shape[1])
        offset += size
    return dict(point_indices=point, sensor_indices=sensor, partner_indices=partner)


def _summaries(raw, mapping, pair_shape):
    for name, value in raw.items():
        _require(np.isfinite(value).all(), 'nonfinite valid point '+name)
    force = raw['normal_forces'][:, 0].astype(np.float64)
    magnitude = np.abs(force)
    owners = (mapping['sensor_indices'], mapping['partner_indices'])
    total = np.zeros(pair_shape, np.float64)
    maximum = np.zeros(pair_shape, np.float64)
    vectors = np.zeros((*pair_shape, 3), np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        np.add.at(total, owners, magnitude)
        np.maximum.at(maximum, owners, magnitude)
        np.add.at(vectors, owners, force[:, None]*raw['normals'].astype(np.float64))
        result = dict(partner_abs_normal_sum_N=total, partner_abs_normal_max_N=maximum,
            sensor_abs_normal_sum_N=total.sum(axis=1, dtype=np.float64),
            filter_abs_normal_sum_N=total.sum(axis=0, dtype=np.float64),
            partner_signed_normal_vector_N=vectors)
    _require(all(np.isfinite(value).all() for value in result.values()), 'nonfinite float64 point-force aggregate')
    return result


def pack_contact_event(normal_forces, points, normals, separations, counts, starts, *, capacity):
    """Copy ONLY owned valid point slots; API tuple can be passed with *arrays."""
    capacity = _capacity(capacity)
    arrays = {name: _float_buffer(value, name, capacity, width)
              for name, value, width in zip(RAW_FIELDS,
                  (normal_forces, points, normals, separations), (1, 3, 3, 1))}
    mapping = _mapping(counts, starts, capacity)
    raw = {name: value[mapping['point_indices']].copy() for name, value in arrays.items()}
    summaries = _summaries(raw, mapping, counts.shape)
    return dict(schema=np.array(EVENT_SCHEMA), capacity=np.array(capacity, np.int64),
                point_count=np.array(len(mapping['point_indices']), np.int64),
                counts=counts.copy(), starts=starts.copy(), **mapping, **raw, **summaries)


def _validate_event(event):
    _require(isinstance(event, dict) and set(event) == EVENT_KEYS, 'packed event schema keys missing/extra')
    schema = _array(event['schema'], 'schema')
    _require(schema.shape == () and schema.dtype.kind == 'U' and str(schema) == EVENT_SCHEMA, 'event schema')
    capacity = _capacity(_scalar_int(event['capacity'], 'capacity'))
    mapping = _mapping(event['counts'], event['starts'], capacity)
    count = len(mapping['point_indices'])
    _require(_scalar_int(event['point_count'], 'point_count') == count, 'point count vs original counts')
    for name, expected in mapping.items():
        actual = _array(event[name], name)
        _require(actual.dtype == np.dtype(np.int64) and actual.shape == expected.shape
                 and np.array_equal(actual, expected), 'packed index/ownership mismatch: '+name)
    raw = {name: _float_buffer(event[name], name, count, width)
           for name, width in zip(RAW_FIELDS, (1, 3, 3, 1))}
    summaries = _summaries(raw, mapping, event['counts'].shape)
    for name, expected in summaries.items():
        actual = _array(event[name], name)
        _require(actual.dtype == np.dtype(np.float64) and actual.shape == expected.shape
                 and np.array_equal(actual, expected), 'point summary does not reconstruct: '+name)
    return capacity, count


def reconstruct_contact_event(event):
    """Return original-index buffers plus observed_mask; unobserved slots = NaN.

    The mask is authoritative. Filled NaNs are an explicit missing-data marker,
    not a recovered copy of unused backend memory. Raw valid point bits and the
    original counts/starts are preserved; signed force is never replaced by abs.
    """
    capacity, _ = _validate_event(event)
    result = {}
    for name in RAW_FIELDS:
        compact = event[name]
        full = np.full((capacity, compact.shape[1]), np.nan, dtype=compact.dtype)
        full[event['point_indices']] = compact
        result[name] = full
    observed = np.zeros(capacity, bool)
    observed[event['point_indices']] = True
    result.update(counts=event['counts'].copy(), starts=event['starts'].copy(), observed_mask=observed)
    return result


def pack_contact_events(events):
    """One view, variable point counts, stable capacity/layout/dtypes; npz-safe."""
    events = list(events)
    _require(bool(events), 'empty event batch is missing evidence, not zero contact')
    dimensions = [_validate_event(event) for event in events]
    capacity = dimensions[0][0]
    first = events[0]
    for event, (cap, _) in zip(events, dimensions):
        _require(cap == capacity and event['counts'].shape == first['counts'].shape,
                 'view capacity or sensor/partner shape changed; start a distinct stream')
        for name in RAW_FIELDS + ('counts', 'starts'):
            _require(event[name].dtype == first[name].dtype, 'event dtype changed: '+name)
    offsets = [0]
    for _, count in dimensions:
        offsets.append(offsets[-1]+count)
    _require(offsets[-1] <= np.iinfo(np.int64).max, 'packed point offsets overflow')
    result = dict(schema=np.array(CHUNK_SCHEMA), capacity=np.array(capacity, np.int64),
                  event_count=np.array(len(events), np.int64), event_offsets=np.array(offsets, np.int64))
    for name in INDEX_FIELDS + RAW_FIELDS:
        result[name] = np.concatenate([event[name] for event in events], axis=0)
    for name in ('counts', 'starts') + SUMMARY_FIELDS:
        result[name] = np.stack([event[name] for event in events], axis=0)
    return result


def unpack_contact_events(chunk):
    """Validate a loaded npz chunk and reconstruct independent compact events.

    Use dict(np.load(path, allow_pickle=False)) at the caller boundary. Neither
    this function nor any other function in this module reads/writes files.
    """
    _require(isinstance(chunk, dict) and set(chunk) == CHUNK_KEYS, 'packed chunk schema keys missing/extra')
    schema = _array(chunk['schema'], 'schema')
    _require(schema.shape == () and schema.dtype.kind == 'U' and str(schema) == CHUNK_SCHEMA, 'chunk schema')
    capacity = _capacity(_scalar_int(chunk['capacity'], 'capacity'))
    size = _scalar_int(chunk['event_count'], 'event_count')
    _require(size > 0, 'positive event_count required')
    offsets = _array(chunk['event_offsets'], 'event_offsets')
    _require(offsets.dtype == np.dtype(np.int64) and offsets.shape == (size+1,)
             and offsets[0] == 0 and (offsets >= 0).all()
             and (offsets[1:] >= offsets[:-1]).all(), 'invalid ragged event offsets')
    total = int(offsets[-1])
    for name in INDEX_FIELDS:
        value = _array(chunk[name], name)
        _require(value.dtype == np.dtype(np.int64) and value.shape == (total,), 'chunk index length/dtype: '+name)
    for name, width in zip(RAW_FIELDS, (1, 3, 3, 1)):
        _float_buffer(chunk[name], name, total, width)
    for name in ('counts', 'starts') + SUMMARY_FIELDS:
        value = _array(chunk[name], name)
        _require(value.ndim > 0 and len(value) == size, 'chunk event dimension: '+name)
    result = []
    for i in range(size):
        start, stop = int(offsets[i]), int(offsets[i+1])
        event = dict(schema=np.array(EVENT_SCHEMA), capacity=np.array(capacity, np.int64),
                     point_count=np.array(stop-start, np.int64))
        for name in INDEX_FIELDS + RAW_FIELDS:
            event[name] = chunk[name][start:stop].copy()
        for name in ('counts', 'starts') + SUMMARY_FIELDS:
            event[name] = chunk[name][i].copy()
        _validate_event(event)
        result.append(event)
    return result
