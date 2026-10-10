"""New V4 source graph; explicitly reuse unchanged camera V3 numerical proofs."""
from repair_common_v2 import *
from request_review_builder_v2 import prepare

def main():
    prior_path = H / 'native_camera_repair_v3/REAPED_CLOSURE_V3.json'
    prior = read(prior_path)
    require(prior['actual_exit'] == prior['close_outer_tool_exit'] == 0 and prior['all_owned_reaped'], 'prior closed CPU release')
    core.verify(read(HERE / 'PRESERVED_V3_BYTES_V4.json')['files'])
    for ref in [prior['release'], prior['artifact_graph'], prior['source_graph']]:
        bound_bytes(ref['path'], ref['sha256'])
    core.verify(read(prior['artifact_graph']['path'])['files'])
    bootstrap_wait = CPU / 'environment1/ACTUAL_WAIT.json'
    successful_wait(read(bootstrap_wait))
    bootstrap_report = CPU / 'CPU_BOOTSTRAP_ENVIRONMENT_REPORT_V4.json'
    report = read(bootstrap_report)
    require(report['tests'] == 3 and report['failures'] == report['errors'] == 0, 'bootstrap regressions')
    core.verify(report['sources'])
    require(sha(report['official_installed_bootstrap_source']['path']) == report['official_installed_bootstrap_source']['sha256'], 'SDK code source')
    failed_root = BASE / 'env002_FR_native_entry_v3_attempt1'
    native_wait = read(failed_root / 'native_process/ACTUAL_WAIT.json')
    outer_wait = read(H / 'camera_v3_execute_execution.json')
    require(native_wait['actual_wait_exit'] == 1 and native_wait['raw_wait_status'] == 256 and
            outer_wait['actual_exit'] == 1 and outer_wait['all_owned_reaped'], 'failed actual predecessor preserved')
    require(all(not i['live'] for i in native_wait['owned_identities']), 'old native reaped')
    require('EOF when reading a line' in Path(native_wait['log']).read_text(), 'observed bootstrap failure')
    plan = prepare()
    release = read(prior['release']['path'])
    release.update(status='CLOSED_CPU_ONLY_V4_BOOTSTRAP_ENVIRONMENT_REVIEW_PENDING',
        source_revision='V4_BOOTSTRAP_ENVIRONMENT_FIX_REUSE_EXACT_CAMERA_V3',
        source_graph=dict(path=str(HERE/'SOURCE_GRAPH_V2.json'),sha256=sha(HERE/'SOURCE_GRAPH_V2.json')),
        parent_plan=dict(path=str(HERE/'PARENT_PLAN_DRAFT_V2.json'),sha256=sha(HERE/'PARENT_PLAN_DRAFT_V2.json')),
        request_draft=dict(path=str(HERE/'REQUEST_DRAFT_V2.json'),sha256=sha(HERE/'REQUEST_DRAFT_V2.json')),
        parent_commands_replaced=True, commands=None, closed_CPU_handoff=None,
        prior_camera_V3_reaped_closure=dict(path=str(prior_path),sha256=sha(prior_path)),
        V3_integration31_angle14_helper4_reused_unchanged=True,
        V4_bootstrap_tests=3, V4_bootstrap_wait=dict(path=str(bootstrap_wait),sha256=sha(bootstrap_wait)),
        V4_bootstrap_report=dict(path=str(bootstrap_report),sha256=sha(bootstrap_report)),
        failed_V3_native_wait=dict(path=str(failed_root/'native_process/ACTUAL_WAIT.json'),sha256=sha(failed_root/'native_process/ACTUAL_WAIT.json')),
        failed_V3_outer_wait=dict(path=str(H/'camera_v3_execute_execution.json'),sha256=sha(H/'camera_v3_execute_execution.json')),
        native_started=False,native_request_frozen=False)
    write(HERE/'CLOSED_RELEASE_V4.json', release)
    write(HERE/'RELEASE_ARTIFACT_SHA256_V4.json', dict(files={str(p):sha(p) for p in HERE.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=release['status'],plan_sha256=sha(HERE/'PARENT_PLAN_DRAFT_V2.json'),
        source_graph_sha256=sha(HERE/'SOURCE_GRAPH_V2.json'),release_sha256=sha(HERE/'CLOSED_RELEASE_V4.json'))),flush=True)

if __name__ == '__main__':
    main()
