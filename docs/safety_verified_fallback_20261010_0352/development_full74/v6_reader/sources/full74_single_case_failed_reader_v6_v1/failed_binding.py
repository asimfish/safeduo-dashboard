"""Bind a reaped failed outer process; missing native wait remains UNKNOWN."""
import os,datetime
from pathlib import Path
from evidence_io import H,R,HERE,Evidence,require,contained,parse_json
from receipt_schema import stream_receipt,CASE,strict_equal
ATTEMPT=R/'astra_full74_camera_repair_development_v1/env002_FR_native_entry_v6_attempt1'
WAIT_SHA='ed5f2a204cb73d3edbd2d8931d54cc6b54e0a7dda9020e3d6197d1fa275e362a'
REQUEST_SHA='18a972196e9d26574ff744f00915829f13e0cf75963e0081ef8888d2966ea863'

def failed_wait(w):
    require(w['schema']=='astra.full74.owned_cpu_wait.v1' and w['backend']=='PARENT_SUPERVISING_NATIVE_SINGLE_CASE','actual failed outer schema/backend')
    for key in ('child_pid','waited_pid','actual_wait_exit','raw_wait_status'):require(type(w[key]) is int,'typed failed outer '+key)
    require(w['child_pid']==w['waited_pid']>0 and w['waitpid_observed'] is True and w['wait_mechanism']=='os.wait4(owned_pid, WNOHANG)','actual outer owned wait4')
    require(w['actual_wait_exit']==os.waitstatus_to_exitcode(w['raw_wait_status'])==-15 and w['raw_wait_status']==15,'preserve actual SIGTERM failure')
    require(w['resource_abort']==w['error']=='ValueError: descendant ancestry changed','exact failure reason retained')
    require(w['owned_identities'] and all(r['live'] is False for r in w['owned_identities']),'all recorded owned identities absent')
    roots=[r for r in w['owned_identities'] if r['pid']==w['child_pid']]
    require(len(roots)==1 and roots[0]['startticks']>0 and roots[0]['enrollment']==dict(kind='direct_fork_handshake',parent_pid=roots[0]['ppid']),'owned outer root provenance')
    return roots[0]

def absent_owned(w):
    count=0
    for row in w['owned_identities']:
        path=Path(f"/proc/{row['pid']}/stat")
        try:raw=path.read_text()
        except FileNotFoundError:count+=1;continue
        cols=raw[raw.rfind(')')+1:].split()
        require(int(cols[19])!=row['startticks'],'recorded owned PID/startticks still present; no raw audit yet')
        count+=1
    return count

def load(e,path,digest):
    bind=e.js(path,digest)
    require(bind['schema']=='astra.single_case.failed_native_binding.v6.v1' and bind['cpu_test_only'] is False and bind['attempt_root']==str(ATTEMPT),'exact actual failed binding')
    require(bind['outer_wait']['sha256']==WAIT_SHA and bind['request']['sha256']==REQUEST_SHA,'user pinned outer/request')
    def read(ref):return e.js(ref['path'],ref['sha256'])
    wait=read(bind['outer_wait']);own=failed_wait(wait);absence=absent_owned(wait)
    outer=read(bind['outer_execution'])
    require(outer['wait']==bind['outer_wait'] and outer['actual_exit']==-15 and outer['pid']==own['pid'] and outer['all_owned_reaped'] is True,'failed execution/real wait linkage')
    require(outer['argv']==wait['argv'],'outer executed argv agreement')
    for key in ('log','outer_monitor'):e.hash(wait[key],wait[key+'_sha256'])
    require(outer['log_sha256']==wait['log_sha256'],'outer log hash link')
    missing=[]
    for relative in bind['missing']:
        p=contained(ATTEMPT/relative,ATTEMPT);require(not p.exists(),'expected missing failure artifact changed');missing.append(str(p))
    req=read(bind['request']);parent=read(bind['parent_review']);approval=read(bind['parent_approval']);review=read(bind['minimal_review'])
    require(review['review_verdict']=='CONDITIONAL_PASS' and review['resolved_finding']=='V5_FABRIC_LOGICAL_CUDA1_UNSUPPORTED','closed V6 source review')
    require(req['source_files']==review['prospective']['request_source_files'] and req['source_graph_sha256']==review['source_graph']['sha256'],'exact reviewed source graph')
    require(strict_equal(req['case'],CASE) and req['parent_frozen'] is True and req['draft_only'] is False and req['cpu_test_only'] is False,'frozen actual single env case')
    require(req['attempt_root']==str(ATTEMPT) and req['out']==str(ATTEMPT/'native'),'exact actual output')
    require(parent['request_sha256']==REQUEST_SHA and parent['parent_approval_sha256']==bind['parent_approval']['sha256'] and parent['parent_execute_authorized'] is True,'actual request approval binding')
    require(approval['prospective_frozen_request_sha256']==REQUEST_SHA and approval['attempt_root']==str(ATTEMPT),'parent approved exact failed attempt')
    e.verify(req['source_files']);e.verify(req['input_files'])
    graph=read(review['source_graph']);anchors=e.js(HERE/'anchors.json');original_path=H/'astra/full74_screen/INPUT_BINDING_V1.json'
    original=e.js(original_path,anchors['files'][str(original_path)])
    require(len(original['original500_sources'])==500 and graph['original500']['files']==original['original500_sources'],'unchanged original500')
    e.verify(original['files'])
    receipt,stream=stream_receipt(e,bind['native_receipt']['path'],bind['native_receipt']['sha256'])
    require(receipt['status']=='FAILED' and receipt['request_sha256']==REQUEST_SHA and len(receipt['attempts'])==96,'exact failed 96-attempt receipt')
    # Bind the recorded native enrollment without inventing its missing wait.
    monitor=ATTEMPT/'native_process/resources.jsonl';e.hash(monitor);enrollment=None
    with monitor.open('rb') as f:
        for line in f:
            require(len(line)<=4*1024**2,'bounded native monitor row');row=parse_json(line)
            if row.get('event')=='OWNED_ENROLLED':require(enrollment is None,'one native enrollment');enrollment=row
    require(enrollment is not None,'recorded native enrollment')
    direct=[v for v in enrollment['identities'] if v['enrollment'].get('kind')=='direct_fork_handshake']
    require(len(direct)==1 and direct[0]['ppid']==own['pid'],'native enrollment linked to failed parent')
    native=direct[0]
    require(any(r['pid']==native['pid'] and r['startticks']==native['startticks'] and r['live'] is False for r in wait['owned_identities']),'native enrolled identity included in closed outer absence')
    return dict(bind=bind,wait=wait,root=ATTEMPT,request=req,original=original,anchors=anchors,receipt=receipt,stream=stream,
        closure=dict(outer_actual_exit=-15,outer_raw_status=15,resource_abort=wait['resource_abort'],signals=wait['signals'],
            outer_pid=own['pid'],outer_startticks=own['startticks'],native_enrolled_pid=native['pid'],native_enrolled_startticks=native['startticks'],
            native_actual_exit='UNKNOWN_MISSING_NATIVE_WAIT',ENTRY='MISSING',actual_CUDA_identity='UNKNOWN',actual_Kit_identity='UNKNOWN',actual_thread_environment='UNKNOWN',
            missing_artifacts=missing,recorded_owned_absent=absence,native_pass=False,qualification=False,safety_acceptance=False))

def recheck_closed(context):
    absent_owned(context['wait'])
    for p in context['closure']['missing_artifacts']:require(not Path(p).exists(),'missing evidence appeared during audit')
