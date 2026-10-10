"""V9 attempt/entry/diagnostic binding. No simulator or current-pointer trust."""
from pathlib import Path
import datetime
from evidence_io import H,HERE,NATIVE,SCREEN,Evidence,contained,require

EXECUTION_ENTRY=H/'native_lifecycle_repair_v1/native_short_entry_v6.py'
THREAD_CONFIG_ENTRY=H/'native_lifecycle_repair_v1/native_short_entry_v4.py'
ACTUAL_THREAD_KEYS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')


def execution_sources(schema,anchors):
    sources=dict(schema['caller_sources'])
    require(len(sources)==4 and str(EXECUTION_ENTRY) not in sources,'original four execution sources')
    extras=anchors['execution_extras']
    expected={str(EXECUTION_ENTRY)}|{str(H/'native_root_restore_equivalence_v1'/n) for n in
        ('native_full74_screen_v3.py','root_equivalence_v1.py','test_root_equivalence_v1.py')}
    expected|={str(H/'native_initialization_diagnostics_v1'/n) for n in ('phase_trace_v1.py','test_diagnostics_v1.py')}
    require(set(extras)==expected and all(anchors['files'][p]==s for p,s in extras.items()),'exact V9 execution extras')
    sources.update(extras)
    return sources


def thread_settings(entry):
    values=entry.get('CPU_thread_settings_after_app_launch')
    require(isinstance(values,dict) and all(type(values.get(k)) is str and values[k]=='1'
            for k in ACTUAL_THREAD_KEYS),'actual post-launch OMP/MKL/OPENBLAS must each be string 1')
    # PXR is recorded by V4 but not part of its strict three-variable assertion.
    return dict(values)


def thread_cpu_binding(e,review,anchors):
    path=anchors['thread_CPU_report'];digest=anchors['files'][path]
    require(review['thread_CPU_report']==path and review['thread_CPU_report_sha256']==digest,
            'parent V7 installed-thread CPU proof binding')
    proof=e.js(path,digest)
    require(proof['status']=='PASS_CPU_INSTALLED_STARTUP_THREAD_CONFIG_ONLY' and
            type(proof['tests']) is int and proof['tests']==3 and
            type(proof['failures']) is int and proof['failures']==0 and
            type(proof['errors']) is int and proof['errors']==0 and
            proof['native_started'] is False and proof['GPU_initialized'] is False and
            proof['original_solver_gravity_timestep_parameters_unchanged'] is True,
            'closed installed-thread CPU counterfactual')
    require(proof['sources'].get(str(THREAD_CONFIG_ENTRY))==anchors['files'][str(THREAD_CONFIG_ENTRY)],
            'thread proof entry V4 SHA')
    require(all(anchors['files'].get(p)==s for p,s in proof['sources'].items()),'thread proof unpinned source')
    e.verify(proof['sources'])
    path=anchors['thread_CPU_execution'];execution=e.js(path,anchors['files'][path])
    require(type(execution['actual_exit']) is int and execution['actual_exit']==0 and
            type(execution['pid']) is int and execution['pid']>0,'installed-thread CPU execution exit')
    require(execution['argv'][1:]==['-B',str(EXECUTION_ENTRY.parent/'test_thread_config_v1.py')],
            'installed-thread CPU execution argv')
    log=anchors['thread_CPU_log']
    require(execution['log_sha256']==anchors['files'][log],'installed-thread CPU log anchor')
    e.hash(log,execution['log_sha256'])
    return proof


def root_cpu_binding(e,review,anchors):
    path=anchors['root_CPU_report'];digest=anchors['files'][path]
    require(review['root_restore_CPU_report']==path and review['root_restore_CPU_report_sha256']==digest,'V8 root CPU proof binding')
    proof=e.js(path,digest)
    require(proof['status']=='PASS_CPU_RECORDED_RESTORE_REPLAY_ONLY' and proof['tests']==6 and
            proof['errors']==proof['failures']==0 and proof['native_started'] is False and proof['GPU_initialized'] is False and
            proof['camera_held_native27_bitwise_unchanged'] is True and proof['prior_failed_V7_not_reclassified'] is True and
            proof['formal_holdout_states']==0,'V8 root CPU proof incomplete')
    require(all(anchors['files'].get(p)==s for p,s in proof['sources'].items()),'root CPU unbound sources')
    e.verify(proof['sources'])
    p=anchors['root_CPU_execution'];execution=e.js(p,anchors['files'][p])
    require(type(execution['actual_exit']) is int and execution['actual_exit']==0 and
            execution['argv'][1:]==['-B',str(H/'native_root_restore_equivalence_v1/test_root_equivalence_v1.py')],
            'root CPU execution receipt')
    p=anchors['root_CPU_log'];require(execution['log_sha256']==anchors['files'][p],'root CPU log binding')
    e.hash(p,execution['log_sha256'])


def diagnostics_cpu_binding(e,review,anchors):
    path=anchors['diagnostics_CPU_report'];digest=anchors['files'][path]
    require(review['initialization_diagnostics_CPU_report']==path and
            review['initialization_diagnostics_CPU_report_sha256']==digest,'V9 diagnostic CPU proof binding')
    proof=e.js(path,digest)
    require(proof['status']=='PASS_CPU_DIAGNOSTICS_ONLY' and type(proof['tests']) is int and proof['tests']==4 and
            proof['original_500_unchanged'] is True and proof['native_scientific_functions_AST_unchanged'] is True and
            proof['native_launched'] is False and proof['GPU_initialized'] is False and
            proof['physical_safety_certified'] is False and proof['all_requirements_completed'] is False,
            'closed diagnostic CPU-only contract')
    sources=({str(H/'native_lifecycle_repair_v1'/n) for n in ('native_short_entry_v5.py','native_short_entry_v6.py')}|
        {str(H/'native_initialization_diagnostics_v1'/n) for n in ('phase_trace_v1.py','test_diagnostics_v1.py')})
    require(set(proof['sources'])==sources and all(anchors['files'].get(p)==s for p,s in proof['sources'].items()),
            'diagnostic CPU exact source inventory')
    e.verify(proof['sources']);e.verify(proof['artifacts'])
    p=anchors['diagnostics_CPU_execution'];execution=e.js(p,anchors['files'][p])
    require(type(execution['actual_exit']) is int and execution['actual_exit']==0 and type(execution['pid']) is int and
            execution['pid']>0 and execution['argv'][1:]==
            ['-B',str(H/'native_initialization_diagnostics_v1/test_diagnostics_v1.py')],'diagnostic CPU execution receipt')
    p=anchors['diagnostics_CPU_log'];require(execution['log_sha256']==anchors['files'][p],'diagnostic CPU log binding')
    e.hash(p,execution['log_sha256'])


def attempt_binding(w,anchors):
    known=anchors['actual_native_identity']
    require(isinstance(known,dict) and set(known)=={'pid','startticks'},'exact enrolled V9 identity required')
    own=next(x for x in w['owned_identities'] if x['pid']==w['child_pid'])
    require({k:own[k] for k in known}==known and own['ppid']==anchors['actual_native_parent_pid'],
            'V9 enrolled child/parent binding')
    request=anchors['frozen_request_binding'];review=anchors['parent_review_binding']
    require(w['request']==request['path'] and w['request_sha256']==request['sha256'],'exact V9 frozen request binding')
    require(w['parent_review_path']==review['path'] and w['parent_review_sha256']==review['sha256'],'exact V9 parent review binding')


def producer_complete(e,w,root,anchors):
    """Check source-bound producer closure before a reader child may be spawned.

    This is an admission gate, not independent scientific acceptance. Full array,
    camera-pixel and ledger checks remain in reader.py after this gate succeeds.
    """
    from reader import entry_lifecycle
    closure=e.js(root/'PARENT_PRODUCT_CLOSURE.json')
    require(closure['status']=='PRODUCER_SHORT_CLOSURE_ONLY_INDEPENDENT_READER_REQUIRED' and
            closure['native_wait']==str(root/'PARENT_NATIVE_WAIT.json') and
            closure['native_wait_sha256']==e.hash(root/'PARENT_NATIVE_WAIT.json'),'producer complete wait binding')
    req=e.js(w['request'],w['request_sha256']);out=root/'native'
    from root_restore_reader import protocol_binding
    protocol_binding(e,req,anchors,w['started_utc'])
    require(req['out']==str(out),'producer output root')
    schema=e.js(SCREEN/'NATIVE_SHORT_READER_SCHEMA_V3.json',anchors['files'][str(SCREEN/'NATIVE_SHORT_READER_SCHEMA_V3.json')])
    require(req['runner_sources']==execution_sources(schema,anchors),'producer entry V6 execution sources')
    e.verify(req['runner_sources'])
    require(w['argv'][1:3]==['-B',str(EXECUTION_ENTRY)],'producer V6 argv')
    entry=e.js(w['entry_receipt'],w['entry_receipt_sha256'])
    receipt=e.js(w['native_receipt'],w['native_receipt_sha256'])
    entry_lifecycle(e,out,entry);thread_settings(entry)
    require(entry['schema']=='astra.full74.native_short_entry.v1' and entry['native_started'] is True and
            entry['native_launch_attempted'] is True and entry['entry_sha256']==anchors['files'][str(EXECUTION_ENTRY)],
            'producer complete native entry V6')
    require(entry['request']==w['request'] and entry['request_sha256']==w['request_sha256'] and
            entry['native_receipt']==w['native_receipt'] and entry['native_receipt_sha256']==w['native_receipt_sha256'],
            'producer entry request/native chain')
    require(receipt['schema']=='astra.full74.native_screen_receipt.v1' and receipt['status']=='complete' and
            receipt['backend']=='isaac_physx_native' and receipt['runner_sources']==req['runner_sources'] and
            receipt['mode']=='camera-short' and receipt['completed_batches']==1 and
            receipt['actual_prefix_microsteps']==12 and receipt['camera_all32_initial_final_qualified'] is True and
            receipt['native27_checks_unchanged'] is True,'producer native products incomplete')
    for key in ('camera_closure','camera_cleanup'):
        require(receipt[key]['status']=='closed' and receipt[key]['errors']==[],'producer camera not closed')
    for rel in schema['required_top_files']+schema['required_batch_files']:
        require(contained(out/rel,out).is_file(),'producer required product missing '+rel)
    fields=out/'batch_00000/native_fields';closed=e.js(fields/'CLOSED.json')
    require(closed['filled']==13 and set(closed['files'])=={str(fields/(k+'.npy')) for k in schema['native_fields']},
            'producer full13 field inventory')
    for p in closed['files']:
        require(contained(p,fields).is_file(),'producer native field missing')
    for i in range(13):
        require(e.js(fields/f'row_{i:03d}_complete.json')=={'row':i},'producer missing completed micro row')
    restore=out/'batch_00000/ROOT_RESTORE_EQUIVALENCE_V1.json'
    require(str(restore) in receipt['artifacts'],'producer root restoration report binding')
    e.hash(restore,receipt['artifacts'][str(restore)])
    review=e.js(w['parent_review_path'],w['parent_review_sha256'])
    require(review['attempt_root']==str(root) and review['execute_authorized'] is True,'producer review root')
    thread_cpu_binding(e,review,anchors)
    root_cpu_binding(e,review,anchors)
    diagnostics_cpu_binding(e,review,anchors)


def outer_completion(e,anchors,w,outer_sha):
    require(isinstance(outer_sha,str) and len(outer_sha)==64 and all(c in '0123456789abcdef' for c in outer_sha),
            'parent-authoritative outer execution SHA required')
    outer=e.js(anchors['outer_execution_path'],outer_sha)
    require(outer['tag']=='full74_screen_parent_failure_v9' and type(outer['actual_exit']) is int and
            outer['actual_exit']==0,'outer parent actual exit must be zero')
    own=next(x for x in w['owned_identities'] if x['pid']==w['child_pid'])
    require(type(outer['pid']) is int and outer['pid']==own['ppid'] and outer['argv'][1:]==
            ['-B',str(H/'launch_full74_screen_after_failure_v9.py')],'outer parent identity/argv binding')
    dates=[datetime.datetime.fromisoformat(v) for v in
           (outer['started_utc'],w['started_utc'],w['closed_utc'],outer['closed_utc'])]
    require(all(d.tzinfo is not None for d in dates) and dates==sorted(dates),'outer/native closure time order')
    e.hash(anchors['outer_log_path'],outer['log_sha256'])
    return outer


def closure_preflight(attempt_root,wait_sha,request_sha,outer_sha=None):
    # Import at call time avoids a cycle. Neither module imports native producers.
    from reader import parent_wait,identity_anchor
    e=Evidence();anchors=e.js(HERE/'anchors.json')
    require(str(attempt_root)==anchors['actual_attempt_root'],'parent-confirmed exact V9 attempt root')
    root=contained(attempt_root,NATIVE)
    w=parent_wait(e,root,wait_sha,request_sha)
    identity_anchor(w,anchors)
    attempt_binding(w,anchors)
    outer_completion(e,anchors,w,outer_sha)
    producer_complete(e,w,root,anchors)
    e.recheck()
    return w
