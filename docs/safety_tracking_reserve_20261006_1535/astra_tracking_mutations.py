"""Reject nine in-memory defects; never modify parent or production source."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import io
import json
import unittest
import astra_tracking_tests as tests

H=Path(__file__).resolve().parent


def main():
    base=tests.load_candidate('current')
    cases=[
        ('ignore_velocity','tracking_reserve.py','qd[a] * (18 * dt)','qd[a] * 0',tests.ReserveContractTests,'test_velocity_at_zero_debt_and_direction'),
        ('last_pending_only','tracking_reserve.py','enumerate(pending)','enumerate([pending[-1]])',tests.ReserveContractTests,'test_every_pending_slot_can_determine_minimum'),
        ('drop_right_arm','tracking_reserve.py',"displacement[r+'_R']","displacement[r+'_R'] * 0",tests.ReserveContractTests,'test_joint_reinforcement_and_cancellation'),
        ('unsigned_projected_motion','tracking_reserve.py','endpoint = distance + delta','endpoint = distance + delta.abs()',tests.ReserveContractTests,'test_velocity_at_zero_debt_and_direction'),
        ('omit_dmin','tracking_reserve.py','prediction - dmin','prediction',tests.ReserveContractTests,'test_exact_original_exemption_and_per_row_dmin'),
        ('ignore_original_exemptions','tracking_reserve.py',"masked_fill(exempt, float('inf'))","masked_fill(torch.zeros_like(exempt), float('inf'))",tests.ReserveContractTests,'test_exact_original_exemption_and_per_row_dmin'),
        ('suppress_finite_abort','tracking_reserve.py','from target_forecast import require_finite','require_finite = lambda *args: None',tests.ReserveContractTests,'test_nonfinite_inputs_including_exempt_entries_abort'),
        ('snap_unreachable_gap','reference_envelope.py',
         'return ((q - gap - target).maximum(lo).minimum(hi),\n            (q + gap - target).maximum(lo).minimum(hi))',
         'return q - gap - target, q + gap - target',tests.ReferenceContractTests,'test_perenv_gap_and_unreachable_does_not_snap'),
        ('new_command_contaminates_risk','guard_runner.py','rows.J, state.q, state.qd, env._evaluation_actuator_delay.queue.pending',
         'rows.J, state.q, env._pending_cmd.delta_q, env._evaluation_actuator_delay.queue.pending',tests.GuardWiringTests,'test_current_proposal_changes_admission_but_not_prequeue_reserve'),
    ]
    outcomes=[]
    for name,file,old,new,cls,method in cases:
        if base.text[file].count(old)!=1:raise ValueError('mutation anchor changed: '+name)
        mutated=base.text[file].replace(old,new,1)
        candidate=SimpleNamespace(**vars(base));candidate.text=dict(base.text);candidate.text[file]=mutated
        if file!='guard_runner.py':
            namespace={'__file__':str(H/file),'__name__':'astra_mutant_'+name}
            exec(compile(mutated,'<'+name+'>','exec'),namespace)
            setattr(candidate,'tracking' if file=='tracking_reserve.py' else 'reference',SimpleNamespace(**namespace))
        tests.CANDIDATE=candidate
        output=io.StringIO()
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.TestSuite([cls(method)]))
        if result.wasSuccessful():raise AssertionError('surviving mutation: '+name)
        outcomes.append(dict(mutation=name,test=method,rejected=True,failures=len(result.failures),errors=len(result.errors),output=output.getvalue()))
    tests.CANDIDATE=base
    for name,digest in base.source_sha256.items():
        if hashlib.sha256((H/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('parent source changed')
    result=dict(status='PASS_ALL9_IN_MEMORY_MUTATIONS_REJECTED',source_sha256=base.source_sha256,
                mutations=outcomes,parent_files_modified=False,physical_safety_claim=False)
    with (H/'astra_tracking_mutation_results.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'])


if __name__=='__main__':main()
