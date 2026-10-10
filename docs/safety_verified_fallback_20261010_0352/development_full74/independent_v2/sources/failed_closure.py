"""Closed failed attempt provenance, independent of the complete-screen gate."""
import datetime
import os
from pathlib import Path
from evidence_io import H,HERE,NATIVE,Evidence,contained,parse_json,require
from binding_kernels import attempt_binding


class PendingClosure(RuntimeError):pass


def terminal_numbers(w):
    require(w['schema']=='astra.full74.parent_native_wait.v1' and w['backend']=='isaac_physx_native','native wait schema/backend')
    for key in ('child_pid','waited_pid','raw_wait_status','actual_wait_exit'):
        require(type(w[key]) is int,'terminal native integer '+key)
    raw=w['raw_wait_status']
    require(w['waitpid_observed'] is True and w['child_pid']==w['waited_pid']>0,'actual waited native PID')
    require(0<=raw<65536 and (os.WIFEXITED(raw) or os.WIFSIGNALED(raw)) and
            os.waitstatus_to_exitcode(raw)==w['actual_wait_exit'],'raw wait/exit mismatch or nonterminal wait')
    require(w['wait_mechanism']=='os.wait4(owned_pid, WNOHANG)','real parent wait4 required')
    own=[x for x in w['owned_identities'] if x['pid']==w['child_pid']]
    require(len(own)==1 and all(x['live'] is False for x in w['owned_identities']),'owned native processes not terminal')
    own=own[0];require(type(own['startticks']) is int and own['startticks']>0 and
        own['enrollment']['kind']=='direct_fork_handshake' and own['enrollment']['parent_pid']==own['ppid']>0,'native enrollment')
    return own


def outer_failure(e,w,anchors,digest):
    outer=e.js(anchors['outer_execution_path'],digest)
    require(outer['tag']=='full74_screen_parent_failure_v9' and type(outer['actual_exit']) is int and
            outer['actual_exit']==1,'actual failed outer1 required; no success-gate rewrite')
    require(type(outer['pid']) is int and outer['pid']==anchors['actual_native_parent_pid'] and
            outer['argv'][1:]==['-B',str(H/'launch_full74_screen_after_failure_v9.py')],'outer identity/argv')
    dates=[datetime.datetime.fromisoformat(x) for x in
           (outer['started_utc'],w['started_utc'],w['closed_utc'],outer['closed_utc'])]
    require(all(x.tzinfo is not None for x in dates) and dates==sorted(dates),'ordered actual closure dates')
    e.hash(anchors['outer_log_path'],outer['log_sha256'])
    return outer


def terminal_gate(e,anchors,root,wait_sha,request_sha,outer_sha):
    require(str(root)==anchors['actual_attempt_root'],'exact failed V9 root; CPU fixtures cannot become native')
    root=contained(root,NATIVE)
    for value in (wait_sha,request_sha,outer_sha):
        require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'explicit authoritative SHA required')
    path=root/'PARENT_NATIVE_WAIT.json'
    if not path.is_file():raise PendingClosure('actual parent native wait absent; no native products opened')
    w=e.js(path,wait_sha);own=terminal_numbers(w);attempt_binding(w,anchors)
    require(w['request_sha256']==request_sha,'authoritative request SHA')
    closed=anchors['actual_failed_closure_binding']
    require(wait_sha==closed['native_wait_sha256'] and outer_sha==closed['outer_execution_sha256'] and
            w['actual_wait_exit']==closed['native_actual_exit'] and w['raw_wait_status']==closed['native_raw_wait_status'],
            'exact failed V9 closure binding; no receipt substitution')
    outer=outer_failure(e,w,anchors,outer_sha)
    for key in ('log','outer_monitor','request','parent_prelaunch_gate','fresh_recheck'):
        contained(w[key],root);e.hash(w[key],w[key+'_sha256'])
    e.hash(w['parent_review_path'],w['parent_review_sha256'])
    events=[];resources=0;resource_failures=[]
    with Path(w['outer_monitor']).open('rb') as f:
        for line in iter(lambda:f.readline(4*1024**2+1),b''):
            require(len(line)<=4*1024**2,'bounded parent monitor line')
            item=parse_json(line)
            if item.get('event') in ('OWNED_ENROLLED','ACTUAL_WAIT'):events.append(item)
            if item.get('event')=='RESOURCE':
                resources+=1
                if item['failed']:resource_failures.append(dict(sample=resources,failed=item['failed']))
            require(len(events)<=2,'duplicate enrollment/wait event')
    enrolled=[x for x in events if x['event']=='OWNED_ENROLLED'];waited=[x for x in events if x['event']=='ACTUAL_WAIT']
    require(len(enrolled)==len(waited)==1 and resources>0,'actual enrollment/wait/resource events')
    first=[x for x in enrolled[0]['identities'] if x['pid']==own['pid']]
    require(len(first)==1 and all(first[0][k]==own[k] for k in ('pid','ppid','startticks','enrollment')) and
            enrolled[0]['argv']==w['argv'],'enrollment identity/argv disagreement')
    require(all(waited[0][k]==w[k] for k in ('child_pid','waited_pid','raw_wait_status','actual_wait_exit','resource_abort','signals')) and
            waited[0]['identities']==w['owned_identities'],'terminal event/receipt disagreement')
    receipts={}
    for key,name in [('native_receipt','NATIVE_RECEIPT.json'),('entry_receipt','NATIVE_SHORT_ENTRY_RECEIPT_V1.json')]:
        p=root/'native'/name;require(w[key]==str(p),'cross-attempt receipt')
        digest=w[key+'_sha256']
        if digest is None:
            require(not p.exists(),'unbound receipt appeared after terminal attachment');receipts[key]=None
        else:receipts[key]=e.js(p,digest)
    require(w['entry_receipt_sha256']==closed['entry_receipt_sha256'],'exact durable failed ENTRY SHA')
    require(receipts['native_receipt'] is not None and receipts['entry_receipt'] is not None and
            receipts['native_receipt']['status']==receipts['entry_receipt']['status']=='failed' and
            receipts['native_receipt']['actual_prefix_microsteps']==0,'durable failed initial-only receipts required')
    # Failure/error/signals are retained, never relabeled as native success.
    summary=dict(native_actual_exit=w['actual_wait_exit'],native_raw_wait_status=w['raw_wait_status'],
        outer_actual_exit=outer['actual_exit'],native_pid=own['pid'],native_startticks=own['startticks'],
        parent_pid=outer['pid'],resource_abort=w['resource_abort'],error=w['error'],signals=w['signals'],
        failed_resource_samples=resource_failures,resource_samples=resources,
        native_wait_sha256=wait_sha,outer_execution_sha256=outer_sha,request_sha256=request_sha,
        process_closure_only=True,native_pass=False,stage_pass=False,safety_acceptance=False)
    return w,outer,receipts,summary


def preflight(args):
    e=Evidence();anchors=e.js(HERE/'anchors.json')
    terminal_gate(e,anchors,Path(args.attempt_root),args.parent_wait_sha256,args.request_sha256,args.outer_execution_sha256)
    e.recheck()
