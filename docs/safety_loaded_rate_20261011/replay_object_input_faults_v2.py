"""Software replay with registered object-size and native event-clock binding."""
from pathlib import Path
from dataclasses import replace
import datetime
import hashlib
import json
import numpy as np
from object_input_contract_v1 import ObjectFrame
from object_input_contract_v2 import RegisteredNativeObjectInput

H = Path(__file__).resolve().parent
D = H / 'object_input_fault_replay_v2'
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def injected_faults(packet):
    state, size = packet.root_state, packet.size_m
    nan_position = state.copy(); nan_position[0, 0] = np.nan
    inf_linear = state.copy(); inf_linear[0, 7] = np.inf
    nan_angular = state.copy(); nan_angular[0, 12] = np.nan
    zero_quaternion = state.copy(); zero_quaternion[0, 3:7] = 0
    bad_size = size.copy(); bad_size[0, 0] = 0
    return {
        'missing': None,
        'stale': replace(packet, state_time_s=packet.state_time_s - .051),
        'future': replace(packet, state_time_s=packet.state_time_s + .001),
        'wrong_event': replace(packet, physics_event=packet.physics_event + 1),
        'missing_inventory': replace(packet, names=()),
        'local_frame': replace(packet, coordinate_frame='local'),
        'half_extents': replace(packet, extent_convention='half'),
        'millimetres': replace(packet, length_unit='millimetre'),
        'nan_position': replace(packet, root_state=nan_position),
        'inf_linear_velocity': replace(packet, root_state=inf_linear),
        'nan_angular_velocity': replace(packet, root_state=nan_angular),
        'zero_quaternion': replace(packet, root_state=zero_quaternion),
        'zero_size': replace(packet, size_m=bad_size),
        'positive_wrong_size_with_metre_label': replace(packet, size_m=size * 2),
        'relabelled_current_event_with_wrong_time': replace(packet, state_time_s=packet.state_time_s - .0005),
    }


def main():
    result = D / 'RESULT_V1.json'
    assert not result.exists(), 'immutable replay already completed'
    plan = read(D / 'REPLAY_PLAN_V1.json')
    for source, digest in plan['source_sha256'].items():
        assert sha(source) == digest, 'frozen replay source changed: ' + source
    p = H / 'fourhand_native_probe_v59'
    native = read(p / 'ACTUAL_COMPLETION_EXIT_V59.json')
    assert native['actual_outer_exit'] == native['actual_native_exit'] == native['actual_finite_exit'] == 0
    capture = read(p / 'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    assert capture['record_integrity_verified']
    raw = Path(read(p / 'RESOURCE_EXECUTION_V1.json')['out'])
    gate = RegisteredNativeObjectInput(('common_beam',), plan['registered_full_sizes_m'], physics_dt_s=plan['registered_physics_dt_s'])
    counts, reasons = {}, {}
    visited = 0
    for row in read(raw / 'physics_chunks.json'):
        source = raw / row['path']
        assert sha(source) == row['sha256']
        with np.load(source, allow_pickle=False) as z:
            for index, event in enumerate(z['physics_event']):
                event = int(event)
                assert event == visited
                now = float(z['state_time_s'][index])
                packet = ObjectFrame(('common_beam',), event, now,
                                     z['object_state'][index:index + 1],
                                     z['object_size_m'][index:index + 1])
                for name, damaged in injected_faults(packet).items():
                    decision = gate.assess(damaged, expected_event=event, now_s=now)
                    assert decision.features is None and decision.reason is not None, (event, name)
                    assert decision.lock_request == (True,) * 4
                    assert decision.actor_input_modified is False and decision.physical_stop_verified is False
                    counts[name] = counts.get(name, 0) + 1
                    reasons.setdefault(name, set()).add(decision.reason)
                # Invalid packets must not poison the subsequent fresh packet.
                accepted = gate.assess(packet, expected_event=event, now_s=now)
                assert accepted.reason is None and accepted.lock_request == (False,) * 4
                state = np.asarray(packet.root_state, dtype=np.float64)
                q = state[:, 3:7]
                expected = np.concatenate((state[:, :3], q / np.sqrt((q*q).sum(1, keepdims=True)),
                                           packet.size_m, state[:, 7:]), axis=1)
                np.testing.assert_array_equal(accepted.features, expected)
                visited += 1
    assert visited == capture['physics_events'] == 9584
    assert len(counts) == 15 and set(counts.values()) == {visited}
    result.write_text(json.dumps(dict(
        status='PASS_SOFTWARE_NATIVE59_REGISTERED_OBJECT_SIZE_CLOCK_AND_15_INPUT_FAULT_TYPES',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reused_known_native_events=visited, software_fault_cases=sum(counts.values()),
        fault_counts=counts, fault_reasons={k: sorted(v) for k,v in reasons.items()},
        all16_features_independently_equal=True, invalid_packets_do_not_advance_history=True,
        registered_full_object_size_bound=True, registered_physics_event_clock_bound=True,
        fresh_packet_recovery_events=visited, requested_lock_mask=[True]*4,
        native_actuator_fault_runs=0, new_native_trajectories=0, new_random_trajectories=0,
        learned_actor_object_input_modified=False, vision_estimator_verified=False,
        physical_safe_stop_verified=False, fullSystem0_accepted=False,
        formal_holdout_acceptance_count=0, source_plan_sha256=sha(D/'REPLAY_PLAN_V1.json')),
        indent=2, allow_nan=False) + '\n')
    print('NATIVE_OBJECT_SOFTWARE_REPLAY_PASS', visited, sum(counts.values()), flush=True)


if __name__ == '__main__':
    main()
