"""Additive CUDA1 camera invocation; original audit math and receipts stay intact."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
REGISTRATION = 'ASTRA_ZERO_CAMERA_AMENDMENT_REGISTRATION_01.json'
PINS = {
    'visual_plan.json': '0ed20b89f9f3f1978e3fded6c9c4e6440c4e63fc776a7a1fa82694ae57633589',
    'PLANS_RECOVERY_AMENDMENT.json': 'a6625e21a881f6085a25bd7ff83f8442d9e4df9e8b362224ec731197b79bdf76',
    'visual_recovery_plan.json': '3e72cc373a52e35e3c99f3613a155e0cc144c2fa2ac0e233cfd6e4123de45fb2',
    'astra_zero_camera_plan.json': '3d82ea76b9ee6228e35a5ff8ddcfe2dbf1dff7a757aac48eb10a25ff000511e1',
    'ASTRA_ZERO_CAMERA_SOURCE_REGISTRATION.json': '3d82ea76b9ee6228e35a5ff8ddcfe2dbf1dff7a757aac48eb10a25ff000511e1',
    'astra_zero_camera_selection.json': '35356369abca6d24ce7ec3eba23e726ad6b5f587f33f38f88e3f92208398d6c3',
}
MODES = ['joint_reference', 'zero_inclusive']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat()


def owned_new(name, value):
    require(Path(name).name == name and name.startswith(('astra_zero_', 'ASTRA_ZERO_')), 'unowned output')
    with (H / name).open('x') as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')


def validate_delta(original, amended):
    """Whole-document equality after precisely two declared scalar replacements."""
    require([j['mode'] for j in original['jobs']] == MODES, 'original mode inventory')
    expected = copy.deepcopy(original)
    changes = []
    for job in expected['jobs']:
        argv = job['argv']
        require(argv.count('--device') == 1, 'ambiguous device option')
        index = argv.index('--device') + 1
        require(argv[index] == 'cuda:0', 'original camera device must be cuda:0')
        argv[index] = 'cuda:1'
        changes.append(dict(mode=job['mode'], argv_index=index, before='cuda:0', after='cuda:1'))
    require(amended == expected, 'only the two registered cuda:0->cuda:1 argv tokens may differ')
    return changes


def bound_documents():
    values = {}
    for name, expected in PINS.items():
        data = (H / name).read_bytes()
        require(digest(data) == expected, 'registered input changed: ' + name)
        values[name] = json.loads(data)
    original = values['visual_plan.json']
    amended = values['visual_recovery_plan.json']
    amendment = values['PLANS_RECOVERY_AMENDMENT.json']
    own = values['astra_zero_camera_plan.json']
    require(amendment['visual_recovery_plan_sha256'] == PINS['visual_recovery_plan.json'], 'amendment recovery hash')
    require(own['visual_plan_sha256'] == PINS['visual_plan.json'], 'original plan anchor')
    require(own['selection_sha256'] == PINS['astra_zero_camera_selection.json'], 'original selection anchor')
    require(values['astra_zero_camera_selection.json']['visual_plan_sha256'] == PINS['visual_plan.json'], 'selection plan anchor')
    require(sha(H / 'recover_unstarted.py') == amendment['recovery_driver_sha256'], 'registered recovery producer changed')
    changes = validate_delta(original, amended)
    for name, expected in own['helper_source_sha256'].items():
        require(sha(H / name) == expected, 'frozen helper changed: ' + name)
    return values, changes


def validate_execution(execution, amended):
    require(execution.get('status') == 'complete', 'camera execution not terminal complete')
    rows = execution.get('jobs', [])
    require(len(rows) == 2 and len({r['mode'] for r in rows}) == 2,
            'camera missing/duplicate/extra jobs')
    require({r['mode'] for r in rows} == set(MODES), 'camera mode inventory differs')
    for job in amended['jobs']:
        row = next(r for r in rows if r['mode'] == job['mode'])
        require(row.get('status') == 'complete' and type(row.get('exit_code')) is int
                and row['exit_code'] == 0, 'camera actual wait unrecorded, nonzero or incomplete')
        require(row['argv'] == job['argv'], 'actual camera argv differs from amended plan')
        require(row['out'] == str(Path(amended['actual_visual_root']) / job['mode']), 'camera root mismatch')
    # Missing legacy fields are disclosed, never synthesized. If supplied, an
    # execution-plan field must identify the actual amended plan, not CUDA0.
    for field in ['plan_sha256', 'recovery_plan_sha256', 'executed_plan_sha256']:
        if field in execution:
            require(execution[field] == PINS['visual_recovery_plan.json'], 'false executed-plan hash: ' + field)
    if 'original_plan_sha256' in execution:
        require(execution['original_plan_sha256'] == PINS['visual_plan.json'], 'wrong original plan hash')


def validate_binding(binding, execution_sha256, changes):
    require(binding.get('status') == 'PASS_ACTUAL_CAMERA_DEVICE_AMENDMENT_BINDING', 'missing successful amendment binding')
    expected = {
        'original_plan_sha256': PINS['visual_plan.json'],
        'executed_recovery_plan_sha256': PINS['visual_recovery_plan.json'],
        'registered_recovery_plan_sha256': PINS['visual_recovery_plan.json'],
        'amendment_sha256': PINS['PLANS_RECOVERY_AMENDMENT.json'],
        'actual_camera_execution_sha256': execution_sha256,
    }
    for field, value in expected.items():
        require(binding.get(field) == value, 'amendment binding mismatch: ' + field)
    expected_changes = [dict(mode=r['mode'], argument_index=r['argv_index'], before=r['before'], after=r['after'])
                        for r in changes]
    require(binding.get('device_only_deltas') == expected_changes, 'binding device delta mismatch')
    for field in ['original_receipt_not_modified', 'original_plan_not_relabelled_as_executed',
                  'original_selection_and_math_unchanged']:
        require(binding.get(field) is True, 'binding claim missing: ' + field)


def invocation_body(source):
    """Reuse every original math/selection statement after its three IO guards."""
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    require(len(node.body) > 3, 'frozen run shape')
    header = ast.dump(ast.Module(body=node.body[:3], type_ignores=[]), include_attributes=False)
    body = ast.dump(ast.Module(body=node.body[3:], type_ignores=[]), include_attributes=False)
    return node, digest(header.encode()), digest(body.encode())


def compile_invocation(frozen, registration, writer):
    code = (H / 'astra_zero_camera_prepare.py').read_text()
    node, header_sha, body_sha = invocation_body(code)
    require(header_sha == registration['original_run_header_ast_sha256'], 'frozen IO header changed')
    require(body_sha == registration['unchanged_run_body_ast_sha256'], 'frozen math/inventory body changed')
    node = copy.deepcopy(node)
    node.name = 'invoke_registered_camera_amendment'
    node.args.args.append(ast.arg(arg='execution'))
    node.body = node.body[3:]
    namespace = dict(vars(frozen))
    namespace['write'] = writer
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, '<unchanged frozen camera body; separately validated amendment handshake>', 'exec'), namespace)
    return namespace[node.name]


def check_registration():
    registration = json.loads((H / REGISTRATION).read_text())
    require(registration['status'] == 'REGISTERED_ADDITIVE_CAMERA_DEVICE_AMENDMENT_ONLY', 'invocation not registered')
    require(registration['input_sha256'] == PINS, 'invocation anchors changed')
    for name, expected in registration['new_source_sha256'].items():
        require(sha(H / name) == expected, 'new invocation source changed: ' + name)
    values, changes = bound_documents()
    _, header_sha, body_sha = invocation_body((H / 'astra_zero_camera_prepare.py').read_text())
    require(header_sha == registration['original_run_header_ast_sha256'] and
            body_sha == registration['unchanged_run_body_ast_sha256'], 'registered frozen invocation changed')
    return registration, values, changes


def audit_closed():
    registration, values, changes = check_registration()
    # Verification precedes imports; import the unchanged local CPU auditor only.
    frozen = importlib.import_module('astra_zero_camera_prepare')
    ledger = frozen.Ledger()
    ledger.json(H / REGISTRATION)
    for name, expected in PINS.items():
        ledger.bind(H / name, expected)
    for name, expected in registration['new_source_sha256'].items():
        ledger.bind(H / name, expected)
    original, selection = frozen.registered(ledger)
    amended = ledger.json(H / 'visual_recovery_plan.json', PINS['visual_recovery_plan.json'])
    validate_delta(original, amended)
    amendment = values['PLANS_RECOVERY_AMENDMENT.json']
    ledger.bind(H / 'recover_unstarted.py', amendment['recovery_driver_sha256'])
    recovery = ledger.json(H / 'RECOVERY_EXECUTION.json')
    require(recovery.get('status') == 'PASS_REGISTERED_UNSTARTED_RECOVERY_CLOSED', 'recovery not closed')
    outer = ledger.json(H / 'recovery_closed_execution.json')
    require(outer.get('status') == 'PASS_ACTUAL_CHILD_CLOSED' and outer.get('child_reaped') is True
            and type(outer.get('actual_exit_code')) is int and outer['actual_exit_code'] == 0,
            'no successful actual outer wait receipt')
    require(outer['source_sha256_before'] == outer['source_sha256_after'] == amendment['recovery_driver_sha256'],
            'outer receipt producer source mismatch')
    execution = ledger.json(H / 'visual_execution.json')
    require(recovery['camera_execution_sha256'] == ledger.hashes[str(H / 'visual_execution.json')], 'closed camera receipt changed')
    validate_execution(execution, amended)
    external_binding = ledger.json(H / 'CAMERA_AMENDMENT_EXECUTION_BINDING.json')
    validate_binding(external_binding, ledger.hashes[str(H / 'visual_execution.json')], changes)
    terminal = ledger.json(H / 'RECOVERY_TERMINAL_CLOSURE.json')
    require(terminal.get('status') == 'PASS_ACTUAL_RECOVERY_OUTER_AND_ALL_PERSISTED_CHILD_CLOSURES'
            and type(terminal.get('actual_outer_exit_code')) is int and terminal['actual_outer_exit_code'] == 0
            and terminal.get('complete_numerical_conditions') == 9 and terminal.get('complete_camera_conditions') == 2
            and terminal.get('no_unrecorded_success_child_waits') is True, 'recovery terminal closure incomplete')
    require(terminal['camera_binding_sha256'] == ledger.hashes[str(H / 'CAMERA_AMENDMENT_EXECUTION_BINDING.json')],
            'terminal camera binding changed')
    require(not any((H / n).exists() for n in ['astra_zero_camera_metadata.json', 'astra_zero_camera_image_selection.json',
            'astra_zero_camera_amended_invocation_01.json']), 'retain prior audit attempt; no overwrite or retry')
    binding = dict(status='PASS_AMENDED_CAMERA_INVOCATION_BINDING_ONLY_NOT_RAW_OR_PIXEL_PASS', utc=now(),
        original_plan_sha256=PINS['visual_plan.json'], amendment_sha256=PINS['PLANS_RECOVERY_AMENDMENT.json'],
        executed_recovery_plan_sha256=PINS['visual_recovery_plan.json'], declared_device_delta=changes,
        actual_execution_sha256=ledger.hashes[str(H / 'visual_execution.json')],
        independent_parent_binding_sha256=ledger.hashes[str(H / 'CAMERA_AMENDMENT_EXECUTION_BINDING.json')],
        actual_execution_contains_plan_sha256='plan_sha256' in execution,
        actual_argv=[r['argv'] for r in execution['jobs']], actual_receipt_modified=False,
        original_plan_execution_claimed=False, frozen_math_and_selection_unchanged=True,
        original_run_header_replaced_by_explicit_amendment_verification=True,
        unchanged_run_body_ast_sha256=registration['unchanged_run_body_ast_sha256'], actual_images_viewed=0)
    owned_new('astra_zero_camera_amended_invocation_01.json', binding)
    progress = 0

    def writer(name, value):
        nonlocal progress
        if name == 'astra_zero_camera_progress.json':
            progress += 1
            name = 'astra_zero_camera_amended_progress_%02d.json' % progress
        require(name in ['astra_zero_camera_metadata.json', 'astra_zero_camera_image_selection.json']
                or name.startswith('astra_zero_camera_amended_progress_'), 'unexpected frozen output')
        enriched = dict(value, camera_amendment_binding='astra_zero_camera_amended_invocation_01.json',
                        camera_execution_plan='visual_recovery_plan.json', original_plan_execution_claimed=False)
        owned_new(name, enriched)

    invoke = compile_invocation(frozen, registration, writer)
    try:
        invoke(ledger, amended, selection, execution)
    except BaseException as error:
        owned_new('astra_zero_camera_amended_error_01.json', dict(status='FAIL_AMENDED_CAMERA_AUDIT',
            utc=now(), error=type(error).__name__ + ': ' + str(error), input_sha256=ledger.hashes,
            actual_images_viewed=0, original_receipts_unmodified=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-closed', action='store_true')
    args = parser.parse_args()
    if args.audit_closed:
        audit_closed()
    else:
        check_registration()
        print('PASS_ADDITIVE_AMENDMENT_REGISTRATION_ONLY_NO_OUTCOME_READS', flush=True)
