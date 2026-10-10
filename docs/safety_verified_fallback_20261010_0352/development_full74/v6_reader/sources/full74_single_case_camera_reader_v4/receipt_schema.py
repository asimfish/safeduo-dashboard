"""Bounded single-case packet schema; fixtures never confer native authority."""
import re
from pathlib import Path
import numpy as np
from evidence_io import require,parse_json
from json_stream import Stream,MAX_PACKET,MAX_RECORD

CASE=dict(env_id=2,arm='F_R',native_point=0,proposal_ids=list(range(32)),physical_steps=0,
          policy_actions=0,prefix_qualification=False)
KNOWN_SCHEMAS=('safeduo.camera_repair_diagnostic_receipt.v2',)


def strict_equal(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return set(actual)==set(expected) and all(strict_equal(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return len(actual)==len(expected) and all(strict_equal(x,y) for x,y in zip(actual,expected))
    return actual==expected


def reference(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'exact path/SHA reference schema')
    path=ref['path'];require(isinstance(path,str),'reference path string')
    p=Path(path);require(path and not p.is_absolute() and '..' not in p.parts and str(p)==path,'relative normalized reference')
    require(isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,'reference SHA256')
    return ref


def case_contract(r):
    require(r.get('schema') in KNOWN_SCHEMAS and strict_equal(r.get('case'),CASE),'exact typed env002/F_R case')
    require(type(r.get('physical_steps')) is int and type(r.get('policy_actions')) is int and
        r['physical_steps']==r['policy_actions']==0,'zero additional physical/policy steps')
    require(r.get('prefix_qualification') is False and r.get('aggregate_acceptance') is False and
        r.get('hidden_solver_state_restored') is False,'single-case observation scope only')
    for k in ('CONTACT','ACTUAL_DRIVE_TORQUE','TERMINAL_SET'):require(r.get(k)=='UNKNOWN','unsupported physical assertion '+k)
    refs=r.get('attempts');require(isinstance(refs,list) and 1<=len(refs)<=96,'attempt count1..96')
    for number,ref in enumerate(refs,1):
        reference(ref);require(ref['path']==f'camera_repair_env002_F_R/F_R_attempt_{number:02d}.attempt.json','exact ordered96 attempt references')
    for k in ('native_before','native_after','native_path_mapping','semantic_setup','native_body_bounds'):reference(r[k])
    return refs


def stream_receipt(e,path,expected,allowed_keys=None):
    """No embedded mask arrays/huge aggregate load; retain at most96 SHA refs."""
    path=Path(path);require(path.name=='CAMERA_REPAIR_DIAGNOSTIC_RECEIPT_V2.json','single-case receipt filename')
    digest=e.hash(path,expected,limit=MAX_PACKET);result={};keys=set()
    with path.open('rb') as f:
        stream=Stream(f);stream.take(b'{')
        while stream.peek()!=b'}':
            key=stream.value(4096);require(isinstance(key,str) and key not in keys,'duplicate/nonstring packet field');keys.add(key)
            require(len(keys)<=64 and (allowed_keys is None or key in allowed_keys),'receipt field inventory bound')
            stream.take(b':')
            if key=='attempts':
                refs=[];stream.take(b'[')
                while stream.peek()!=b']':
                    require(len(refs)<96,'attempt index over96');refs.append(reference(stream.value(4096)))
                    if stream.peek()==b']':break
                    stream.take(b',');require(stream.peek()!=b']','trailing attempt comma')
                stream.take(b']');result[key]=refs
            else:result[key]=stream.value()
            if stream.peek()==b'}':break
            stream.take(b',');require(stream.peek()!=b'}','trailing receipt comma')
        stream.take(b'}');require(stream.peek()==b'' and stream.eof,'trailing receipt bytes')
        require(stream.digest.hexdigest()==digest,'receipt changed during streaming')
    if allowed_keys is not None:require(set(result)==set(allowed_keys),'exact reviewed receipt keys')
    case_contract(result);e.hash(path,digest,limit=MAX_PACKET)
    return result,dict(file_bytes=stream.bytes,peak_buffer_bytes=stream.peak_buffer_bytes,packet_limit=MAX_PACKET,
        value_limit=MAX_RECORD,attempt_ref_limit=96,decoded_attempts=len(result['attempts']))


def group_metadata(declared,computed):
    require(isinstance(declared,dict),'group metadata object')
    for k in ('palm_visual_peak_pixels','finger_family_peak_pixels','per_native_body_peak_pixels','native_body_visibility_gaps','all_native_hand_bodies_observed'):
        require(strict_equal(declared.get(k),computed[k]),'typed group metric mismatch '+k)
    require(declared.get('status')==('qualified' if computed['qualified'] else 'failed'),'group status differs from raw predicate')


def thread_record(entry,source_bound_assertion=None):
    value=entry.get('actual_thread_settings_after_app')
    if value is None and source_bound_assertion is not None:
        require(source_bound_assertion.get('validated') is True and
            source_bound_assertion.get('complete_native_and_outer_actual0') is True and
            entry.get('status')=='diagnostic_complete_only' and entry.get('measurement_bytes_verified') is True,
            'reached post-app assertion requires source binding and actual native/outer closure')
        return dict(evidence_type='SOURCE_BOUND_POST_APP_ASSERT_REACHED_WITH_ACTUAL_WAIT0',values_persisted=False,
            checked_values={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            assertion=source_bound_assertion)
    require(isinstance(value,dict),'actual post-app thread settings required; source assert insufficient')
    require(all(value.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'actual threads1')
    return value


def actual_binding(bind):
    keys={'schema','cpu_test_only','attempt_root','request_sha256','review_sha256','native_pid','native_startticks',
        'parent_pid','native_wait_sha256','outer_execution_path','outer_execution_sha256','outer_log_path',
        'parent_closed_sha256','entry_sha256','native_receipt_sha256'}
    require(isinstance(bind,dict) and set(bind)==keys,'actual binding exact field inventory')
    require(bind['schema']=='astra.single_case.reader_actual_binding.v1' and bind['cpu_test_only'] is False,'actual binding authority')
    for k in keys:
        if k.endswith('_sha256'):require(isinstance(bind[k],str) and re.fullmatch('[0-9a-f]{64}',bind[k]) is not None,'actual binding SHA '+k)
    for k in ('native_pid','native_startticks','parent_pid'):require(type(bind[k]) is int and bind[k]>0,'actual identity typed '+k)
    for k in ('attempt_root','outer_execution_path','outer_log_path'):
        require(isinstance(bind[k],str) and Path(bind[k]).is_absolute() and str(Path(bind[k]))==bind[k] and '..' not in Path(bind[k]).parts,'actual absolute path '+k)
    return bind
