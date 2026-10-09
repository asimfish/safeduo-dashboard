"""Prove mode-specific issue records; preserve and clarify copied family metadata."""
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
H = P.parent
REG = json.loads((H/'FIXEDHAND_TRIPLET_DEV_REG_V1.json').read_text())


def main():
    result = json.loads((H/'fixedhand_triplet64_v1/FIXEDHAND_TRIPLET64_RESULT_V1.json').read_text())
    assert result['all3_full960_independent_oracles_pass']
    records = []
    for method in REG['methods']:
        leaf = Path(REG['native_namespace'])/f'paired_batch0_{method}_v8'
        protocol = json.loads((leaf/'response_protocol.json').read_text())
        assert protocol['status']=='complete' and protocol['physics_events']==960
        assert protocol['method']==method
        with np.load(leaf/'response_stream.npz') as s:
            assert s['issued_target'].shape==(480,64,26)
            if method=='raw':
                assert np.array_equal(s['issued_target'], s['reference_target'])
                assert 'fallback_from_model_unsatisfied' not in s.files
                semantics = 'Every issued target equals the frozen raw random reference; no multirow/hold branch'
            elif method=='multirow':
                assert np.array_equal(s['issued_target'], s['nominal_issued_target'])
                assert not s['fallback_from_model_unsatisfied'].any()
                semantics = 'Every issued target equals the nominal multirow proposal; no hold override'
            else:
                with np.load(leaf/'resolved_native_parameters.npz') as p:
                    initial = np.concatenate([p[a+'_initial_q'][:,p[a+'_controlled_joint_indices']] for a in ['F_L','F_R','U_L','U_R']],-1)
                blocked = s['fallback_from_model_unsatisfied']
                assert np.array_equal(blocked, ~s['multi_model_constraints_satisfied'])
                assert np.array_equal(s['issued_target'], np.where(blocked[...,None],initial[None],s['nominal_issued_target']))
                semantics = 'Model-invalid lane holds all26 initial arm targets; valid lane issues nominal multirow proposal'
            record=dict(method=method, issued_lane_controls=30720, exact_issue_branch_verified=True, actual_semantics=semantics, copied_family_description=protocol['paired_controller'])
            if method!='raw':
                record.update(model_satisfied_lane_controls=int(s['multi_model_constraints_satisfied'].sum()), model_unsatisfied_lane_controls=int((~s['multi_model_constraints_satisfied']).sum()), fallback_lane_controls=int(s['fallback_from_model_unsatisfied'].sum()))
                expected=(s['multi_geometry_velocity_residual']<=1e-4)&(s['multi_controlled_velocity_limit_residual']<=1e-3)&(s['multi_full_drive_effort_residual']<=1e-3)&~s['multi_missing_direction']&~s['multi_invalid_box']&~s['multi_uncontrolled_nearby_raw_negative']
                assert np.array_equal(expected,s['multi_model_constraints_satisfied'])
                record.update(native_model_flag_recalculated_exact=True, bounded_impossible_lane_controls=int((s['multi_bounded_impossible_constraints']>0).sum()), max_nominal_geometry_velocity_residual=float(s['multi_geometry_velocity_residual'].max()),max_nominal_controlled_velocity_residual=float(s['multi_controlled_velocity_limit_residual'].max()),max_nominal_full_drive_effort_residual=float(s['multi_full_drive_effort_residual'].max()),nominal_model_residual_is_not_native_force_measurement=True)
            records.append(record)
    clarification = dict(status='PASS_ALL480_NATIVE_ISSUE_BRANCHES_EXACT', records=records,
        metadata_defect='paired_controller is a copied family-level description for all3 methods; use the method and verified native issue branch, not this description, to identify behavior',
        original_native_records_modified=False, frozen_sources_modified=False, no_result_reclassification=True)
    (P/'ISSUE_BRANCH_METADATA_CLARIFICATION_V1.json').write_text(json.dumps(clarification,indent=2)+'\n')
    print(clarification['status'],flush=True)


if __name__=='__main__':
    main()
