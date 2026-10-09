"""Final scoped review, original479/493 protection, input/output SHA handoff."""
import json
from pathlib import Path
import sys

import numpy as np

from run_native_calibration_v1 import D,H,sha,write_json
from native_teacherforced_cpu_v1 import ARMS,WIDTHS,require

F=D/'first_candidate480x32_v2'


def main():
    full=json.loads((F/'FULL480_NATIVE_CALIBRATION_REPORT_V2.json').read_text())
    tests=json.loads((D/'full480_positive_negative_tests_v2.json').read_text())
    require(tests['passed'] and tests['tests_run']==40 and tests['errors']==tests['failures']==0,'40tests required')
    frozen=json.loads((F/'frozen493_before_after.json').read_text())
    require(frozen['union_count']==493 and frozen['all493_unchanged'],'493 frozen gate')
    for p,e in frozen['sources'].items():require(sha(p)==e['expected_sha256'],'frozen source changed: '+p)
    inputs={}
    for p in [D/'two_witness_v1/input_source_sha256.json',F/'input_source_sha256.json']:
        for name,e in json.loads(p.read_text())['entries'].items():
            require(name not in inputs or inputs[name]['sha256']==e['sha256'],'source drift across calibration versions')
            inputs[name]=e
    legacy={}
    for name in ('QUEUE74_PREFLIGHT_PLAN_V1.json','QUEUE74_FULL_PLAN_V1.json'):
        p=H/name;source=json.loads(p.read_text())['sources'];require(len(source)==479,'legacy479 expected')
        require(all(path in frozen['sources'] and frozen['sources'][path]['expected_sha256']==s for path,s in source.items()),'legacy479 not exact subset493')
        legacy[name]=dict(path=str(p),sha256=sha(p),count=479,exact_subset_of493=True,all_unchanged=True)
        inputs[str(p)]=dict(sha256=sha(p),role='legacy479 protected source manifest')
    for p in [H/'astra/next_mechanism/demo_original_snapshot_v1.py',H/'astra/next_mechanism/frozen_cpu_providers_v2.py',
              H/'astra/adaptation32/analyze_short_frontier_v4.py',*sorted(D.glob('*.py'))]:
        inputs[str(p)]=dict(sha256=sha(p),role='read-only interpretation or delivered calibration source')
    for p,e in inputs.items():require(sha(p)==e['sha256'],'material input changed at close: '+p)
    wait_rows=[]
    for label in ('two_witness_attempt01','positive_negative_attempt01','first_candidate480_attempt01','full480_tests_attempt01'):
        p=D/(label+'_actual_wait.json');r=json.loads(p.read_text())
        require(r['actual_exit']==0 and r['wait_returned'] and r['owned_pid']==r['waited_pid'],'actual wait0 absent')
        require(sha(r['log'])==r['log_sha256'],'wait log SHA mismatch')
        require(r['script_sha256_before']==r['script_sha256_after'],'executed source changed')
        peak_sum=r['child_max_rss_KiB']+r['supervisor_max_rss_KiB']
        require(peak_sum*1024<1024**3,'conservative parent+child memory upper bound exceeds1GiB')
        wait_rows.append(dict(path=str(p),sha256=sha(p),actual_exit=0,wait_returned=True,
            elapsed_s=r['elapsed_s'],child_max_rss_KiB=r['child_max_rss_KiB'],
            parent_child_peak_sum_upper_KiB=peak_sum))
    for base in (D/'two_witness_v1',F):
        for p,expected in json.loads((base/'output_sha256.json').read_text()).items():
            require(sha(p)==expected,'recorded artifact SHA mismatch')
    params_path=Path(full['original_native_root'])/'resolved_native_parameters.npz'
    parameter_rows=[];offset=0
    with np.load(params_path,allow_pickle=False) as z:
        for arm,width in zip(ARMS,WIDTHS):
            for j,joint in enumerate(z[arm+'_native_joint_names']):
                parameter_rows.append(dict(column=offset+j,joint=arm+'/'+str(joint),
                    native_armature_per32lanes=z[arm+'_armature'][:,j],
                    stiffness_per32lanes=z[arm+'_stiffness'][:,j],damping_per32lanes=z[arm+'_damping'][:,j],
                    max_force_per32lanes=z[arm+'_max_force'][:,j],
                    friction_properties_per32lanes=z[arm+'_friction'][:,j],
                    drive_model_properties_per32lanes=z[arm+'_drive_model'][:,j]))
            offset+=width
    provenance=dict(schema='native_getter_provenance.v2',native_parameter_path=str(params_path),native_parameter_sha256=sha(params_path),
        capture_source=str(H/'native_verified_v4.py'),capture_source_sha256=sha(H/'native_verified_v4.py'),
        getters=dict(armature='get_dof_armatures at native_verified_v4.py:160',
            mass='get_generalized_mass_matrices at native_verified_v4.py:271',
            actuation_command='get_dof_actuation_forces at native_verified_v4.py:273',
            incoming_projected_force='get_dof_projected_joint_forces at native_verified_v4.py:274',
            gravity='get_gravity_compensation_forces at native_verified_v4.py:275',
            coriolis='get_coriolis_and_centrifugal_compensation_forces at native_verified_v4.py:276'),
        parameter_rows74=parameter_rows,
        actual_drive_torque='UNKNOWN: no isolated implicit drive torque observation in retained files.',
        source_reasoning=full['torque_provenance'],
        armature_inclusion_in_generalized_M='UNKNOWN; explicit recorded armature values preserved; as_recorded/add_diagonal remain separate unvalidated model assumptions.',
        no_backend_or_simulator_imported=True)
    write_json(D/'NATIVE_GETTER_PROVENANCE_V2.json',provenance)
    review=dict(verdict='Pass',scope='READ_ONLY_CPU_CALIBRATION_ARTIFACTS_AND_HELPER_ONLY',
        reviewer='Current implementing agent performed scoped code/evidence self-review; not independent Astra review.',
        comparison_base='Explicit new files only; no Git base or runtime controller diff is claimed.',
        dimensions=[
            dict(name='Functional achievement',status='Pass',evidence='2known integration witnesses plus first completed480x32; sameactual74 macro2micro; all74 stratified errors.'),
            dict(name='Correctness and reliability',status='Pass',evidence='40positive/negative tests, raw field/chunk SHA, bitexact state/target chains, analytical saturated and coupled residual oracles, wait0.'),
            dict(name='Architecture',status='Pass',evidence='Pure numerical helper, scalar strata helper, separate archive readers and supervisor. Existing frozen solver source reused by SHA; no geometry/camera/runtime code imported.'),
            dict(name='Function and API design',status='Pass',evidence='Macro74 explicit actual target/state/native clock; invalid shape/type/mismatch/nonconvergence rejects. Returns signed errors and model torque only.'),
            dict(name='Style and maintainability',status='Pass',evidence='NEW named Python files; original v1 evidence preserved; every result exclusive-created; no silent fitted fallback or data filtering.'),
            dict(name='Performance',status='Pass',evidence='Measured full8-variant480x32 wait96.8438s; childRSS419228KiB; conservative parent+child peak sum899720KiB <1GiB; one numerical CPU thread.')],
        blocking_findings=[],non_blocking_improvements=[],minimum_required_repair='None for stated diagnostic scope.',
        evidence_lines=dict(target_shape_and_clock=str(D/'native_teacherforced_cpu_v1.py')+':82',
            macro_two_micro_propagation=str(D/'native_teacherforced_cpu_v1.py')+':135',
            divergent_future_invalidated=str(D/'native_teacherforced_cpu_v1.py')+':188',
            raw_chain_crosschecks=str(D/'run_full480_native_calibration_v2.py')+':119',
            strata_are_proxies=str(D/'native_calibration_strata_v2.py')+':20'),
        limitations=['No physical safety/model-error bound/actual torque/contact predictor/terminal-set certificate.',
            'Frozen M,C,g within2micro cannot identify native second-micro dynamics, contact/limit impulses or solver state.',
            'Historical first candidate has no original request pause240:272; new active/pause/resume task contract is not retroactively passed.',
            'Extra GPT6 Astra xhigh session was not available through collaboration APIs; no independent reviewer or model-setting verification is claimed.'])
    write_json(D/'SCOPED_CODE_REVIEW_V2.json',review)
    witness=json.loads((F/'known_witness_stratification_v2.json').read_text())
    primary=full['summaries'][0]
    selected={x['name']:x for x in primary['strata'] if x['name'] in ('all','recorded_contact_free_proxy','contact_linked_macro_or_preboundary')}
    doc='''Closed native calibration handoff v2

Completed scope: two known integration witnesses first, then first CLOSED original
candidate480x32 (15360 macros,30720 micro-lane samples), eight variants, DEVELOPMENT.
All computations are teacher-forced at each native control start, followed by two
model micros using that control's actual applied full74 target. Nothing is fitted.
Source state/target/clock arrays and all74 response fields are preserved by SHA.

Callable helper: native_teacherforced_cpu_v1.teacher_forced_two_micro(Macro74, Variant).
Reference frozen saturated implicit numerical solver is imported by checked SHA.
Tests:40pass. Real wait4: minimal calibration0, initial tests0, full480 calibration0,
full tests0. Full run96.8438s, child maximumRSS409.40MiB; conservative sum of parent
and child recorded peakRSS878.63MiB, below1GiB. OMP/MKL/OpenBLAS1; Python -B and
PYTHONDONTWRITEBYTECODE=1; child address space capped1GiB. No GPU/Isaac/AppLauncher
execution, no runtime policy edit, no other process stopped. Large temporary NPY
files were under R/native_calibration_cpu_tmp and cleaned after compression.

Observed baseline empirical maxima, NOT certified bounds:
  recorded contact-free proxy:18224 micro-lane samples; q0.08129041rad, qd5.24526333rad/s.
  contact-linked macro/preboundary:12496 samples; q0.19354775rad, qd24.89268723rad/s.
Proxy requires zero scalar in both macro micros and prior boundary. Initial prior
boundary is unmeasured. Zero scalar is not proof of no contacts/limit constraints;
the contact strata are associations, not unique causal explanations.

Lane18 micro12 (control6 sub0): recorded scalar0, actual index_1 q=-0.001607122rad,
baseline model q=+0.007796023rad, signed q error+0.009403145rad. ALL7 implicit
variants still miss this lower-limit failure. Explicit PD predicts+1.851258rad
above the upper limit and has huge errors; it does not repair the model.
Lane19 micro13: scalar9.281611443N, known omitted self-contact; CONTACT staysUNKNOWN.
Original36micro repeated-plan targets diverge from actual targets atmicro14 in
both witnesses; full36 direct-error comparison is disabled. Full480 results use
actual-target macro2micro only, never those repeated-plan36micro states.

M/C/gravity/armature/drive assumptions remain distinct. Recorded armature comes
from get_dof_armatures, with per74/per32 values exported in getter provenance.
M's inclusion of armature remains unresolved. Native get_dof_actuation_forces
is a retained actuation command getter; projected joint forces mix incoming link
forces. IsaacLab ImplicitActuator computed/applied effort is expressly approximate.
No retained getter isolates actual implicit motor torque. Full baseline model
reports5637 saturated coordinate-micro cases; these are MODEL saturation flags,
not observations establishing that native saturation occurred.

TASK_ACCEPTANCE_PROTOCOL_V1.json and prospective_task_gate_v2.py were hash checked
and left unchanged. Task reference remains original request26 delayed6controls,
distinct from actual applied74 calibration targets. This historical candidate tape
has NO requested pause240:272. Time bins246:278/278:310 are reported only as windows;
no new active/pause/resume acceptance is claimed. Future task evaluation must use
the frozen protocol and original requests, not fallback/issued targets.

CONTACT / ACTUAL_DRIVE_TORQUE / TERMINAL_SET / GLOBAL_MODEL_ERROR_BOUND = UNKNOWN.
All493 frozen sources (including original479) remain unchanged. See delivery JSON
for all material inputs, sources, outputSHA, actualwait paths and scoped code review.
The review is an implementing-agent code/evidence self-review; extra independent
GPT6 Astra xhigh was unavailable because collaboration APIs lacked session binding.

Primary artifacts:
  NATIVE_CALIBRATION_DELIVERY_V2.json
  SCOPED_CODE_REVIEW_V2.json
  NATIVE_GETTER_PROVENANCE_V2.json
  first_candidate480x32_v2/FULL480_NATIVE_CALIBRATION_REPORT_V2.json
  first_candidate480x32_v2/known_witness_stratification_v2.json
  first_candidate480x32_v2/per74_stratified_error_summary.csv
  first_candidate480x32_v2/*_errors74.npz and actual_native_shared_arrays.npz
  first_candidate480x32_v2/input_source_sha256.json
  first_candidate480_attempt01_actual_wait.json
  full480_tests_attempt01_actual_wait.json
  reviewed_delivery_v2_attempt01_actual_wait.json (written by supervisor after close)

Version1 files remain historical minimal results; v2 expands scope without editing
their bytes. No run was repeated for this review.
'''
    with (D/'HANDOFF_V2.txt').open('x') as f:f.write(doc)
    artifacts={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(D.rglob('*'))
               if p.is_file() and not p.name.startswith('reviewed_delivery_v2_attempt')}
    require(not list(D.rglob('__pycache__')),'unexpected bytecode writes')
    require(not any(k=='torch' or k.startswith(('isaac','omni.')) for k in sys.modules),'unexpected simulator import')
    delivery=dict(schema='safeduo.astra.native_calibration.closed_delivery.v2',status='CLOSED_CPU_RESULTS_REVIEWED',
        helper=str(D/'native_teacherforced_cpu_v1.py'),callable='teacher_forced_two_micro(Macro74, Variant)',
        primary_report=str(F/'FULL480_NATIVE_CALIBRATION_REPORT_V2.json'),
        known_witness_report=str(F/'known_witness_stratification_v2.json'),
        per74_table=str(F/'per74_stratified_error_summary.csv'),review=str(D/'SCOPED_CODE_REVIEW_V2.json'),
        native_getter_provenance=str(D/'NATIVE_GETTER_PROVENANCE_V2.json'),
        baseline_stratified_empirical_results=selected,known_witness_findings=witness['findings'],
        full_unique_macro_lane_samples=15360,full_unique_micro_lane_samples=30720,
        variants=8,tests=tests,actual_wait_receipts=wait_rows,all493_frozen_unchanged=True,
        original479_manifests=legacy,all_material_inputs_and_sources=inputs,artifacts=artifacts,
        task_protocol=full['task_protocol'],CONTACT='UNKNOWN',ACTUAL_DRIVE_TORQUE='UNKNOWN',TERMINAL_SET='UNKNOWN',
        GLOBAL_MODEL_ERROR_BOUND='UNKNOWN',certified_bound=None,physical_safety_certified=False,
        independent_astra_xhigh_review='UNAVAILABLE_NOT_CLAIMED',
        closure_actual_wait=str(D/'reviewed_delivery_v2_attempt01_actual_wait.json'),
        closure_hash_rule='Delivery file excludes itself and currently running closure log/receipt. Supervisor hashes closure log after real wait0.')
    write_json(D/'NATIVE_CALIBRATION_DELIVERY_V2.json',delivery)
    print(json.dumps(dict(status='CLOSED_CPU_RESULTS_REVIEWED',tests=40,frozen493_unchanged=True,original479_unchanged=True,
        delivery=str(D/'NATIVE_CALIBRATION_DELIVERY_V2.json'),delivery_sha256=sha(D/'NATIVE_CALIBRATION_DELIVERY_V2.json'))),flush=True)


if __name__=='__main__':main()
