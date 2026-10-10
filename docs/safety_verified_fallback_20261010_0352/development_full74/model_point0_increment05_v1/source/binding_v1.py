"""Bound metadata only. Unknown or future products never become report results."""
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import types

H = Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R = Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
HERE = Path(__file__).resolve().parent
OUT = R/'astra_report_increment05_v4'
BASE = '314ff23fd696ee7449fe5e2fdeca5bc7aa48dfdc'
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
RULES = H/'astra/full8192_model_geometry_runner_v2/prospective_rules_v1.py'
RULES_SHA = '80ed388fca21c9ec6c67f68e0104ac3df466d3d6b3d0ee2999f09c0a740892e9'
STATES = ('MODEL_UNKNOWN','MODEL_NEGATIVE','MODEL_PREDICTED_INITIAL_CLEAR')

def need(v, message):
    if not v:
        raise ValueError(message)

def owned_path(path):
    p = Path(path)
    need(p.is_absolute() and not p.is_symlink() and (p.resolve().is_relative_to(H) or p.resolve().is_relative_to(R)), 'owned immutable artifact path')
    return p

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def sha(path):
    p = owned_path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        a = os.fstat(f.fileno())
        for raw in iter(lambda:f.read(1024**2), b''):
            h.update(raw)
        b = os.fstat(f.fileno())
    need((a.st_dev,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_size,b.st_mtime_ns,b.st_ctime_ns), 'stable FD bytes/dev/size/mtime/ctime; CIFS inode not authority')
    return h.hexdigest()

def ref(path):
    return dict(path=str(path),sha256=sha(path))

def buffer(q, cap=16*1024**2):
    need(isinstance(q,dict) and set(q)=={'path','sha256'} and isinstance(q['sha256'],str) and len(q['sha256'])==64 and all(c in '0123456789abcdef' for c in q['sha256']), 'complete exact ref; no future placeholder')
    p = owned_path(q['path'])
    with p.open('rb') as f:
        a = os.fstat(f.fileno()); raw = f.read(cap+1); b = os.fstat(f.fileno())
    need(len(raw)<=cap and digest(raw)==q['sha256'], 'bounded authenticated artifact bytes')
    need((a.st_dev,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_size,b.st_mtime_ns,b.st_ctime_ns), 'stable authenticated buffer')
    return raw

def read(q):
    def unique(pairs):
        result = {}
        for k,v in pairs:
            need(k not in result, 'duplicate JSON key'); result[k]=v
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON '+value)
    return json.loads(buffer(q), object_pairs_hook=unique, parse_constant=nonfinite)

def write(path, value):
    p = owned_path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return ref(p)

def same_buffer_module(path, expected, name):
    raw = buffer(dict(path=str(path),sha256=expected),2*1024**2)
    m = types.ModuleType(name); m.__file__=str(path)
    exec(compile(raw,str(path),'exec'),m.__dict__)
    return m

def metadata_rules():
    """Execute only three unchanged accounting functions from the closed source."""
    raw = buffer(dict(path=str(RULES),sha256=RULES_SHA),1024**2)
    names = {'need','known_or_reserved','exhaustive_accounting','prospective_first32'}
    tree = ast.parse(raw)
    selected = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    need({x.name for x in selected}==names, 'exact original accounting API')
    m = types.ModuleType('_unchanged_model_metadata_rules')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(RULES),'exec'),m.__dict__)
    return m

def fresh_absent(identity):
    need(type(identity['pid']) is int and type(identity['startticks']) is int and identity['live'] is False, 'recorded typed owned identity closed')
    try:
        fields=Path('/proc',str(identity['pid']),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:
        return True
    return int(fields[19])!=identity['startticks']

def closed_wait(q):
    w = read(q)
    need(w['waitpid_observed'] is True and w['child_pid']==w['waited_pid'] and type(w['actual_wait_exit']) is int and w['actual_wait_exit']==w['raw_wait_status']==0, 'true owned wait4 zero')
    need(w['owned_identities'] and all(fresh_absent(i) for i in w['owned_identities']) and not w['error'] and not w['resource_abort'] and not w['signals'], 'clean genuinely closed owned process chain')
    need(sha(w['log'])==w['log_sha256'], 'actual log SHA')
    return w

def get(value, keys):
    need(isinstance(keys,list) and keys and all(isinstance(x,str) and x for x in keys), 'literal metadata field path')
    for k in keys:
        need(isinstance(value,dict) and k in value, 'actual schema field '+k);value=value[k]
    return value

def tool0(value, profile):
    profiles={'parent_tool':('actual_exit','session_closed'), 'source_tool':('actual_tool_exit','tool_reaped'), 'terminal_result':('exit_code','session_closed')}
    need(profile in profiles and isinstance(value,dict), 'declared actual tool schema')
    exit_key, closed_key = profiles[profile]
    need(type(value.get(exit_key)) is int and value[exit_key]==0 and value.get(closed_key) is True, 'genuine external tool terminal zero')

def stage_chain(stage, result_ref):
    """Schema paths bind the actual receipt. A nonexistent outer wait stays null."""
    record = read(stage['record'])
    need(get(record,stage['result_field'])==result_ref and get(record,stage['worker_wait_field'])==stage['worker_wait'], 'terminal record owns exact result and worker receipt')
    closed_wait(stage['worker_wait'])
    tool_record=read(stage['tool_record'])
    tool0(get(tool_record,stage['tool_field']),stage['tool_profile'])
    if stage['wrapper_wait'] is not None:
        need(get(record,stage['wrapper_wait_field'])==stage['wrapper_wait'], 'same actual wrapper receipt');closed_wait(stage['wrapper_wait'])
    else:
        need(stage['wrapper_scope']=='ACTUAL_TOOL_TERMINAL_ONLY_NO_SEPARATE_WRAPPER_RECEIPT', 'no fabricated wrapper wait')
    return record

def calibration(report, frame, point0):
    need(report['schema']=='astra.full8192.model_development_replay.v2' and report['synthetic'] is False and report['status']=='EMPIRICAL_MODEL_REPLAY_READY_FOR_REVIEW', 'formal actual runner replay, not preliminary DIAG')
    need(report['frame_binding']==frame and report['calibration_input']==point0 and report['actual_native_states']==128, 'same bound original128 point0')
    need(report['global_error_bound']=='UNKNOWN' and report['formal_holdout'] is False and report['native_admission'] is False and report['diagnostic_tolerance_m']==5e-5, 'unchanged diagnostic threshold and no admission')
    for name in ('calibration','development_verification'):
        x=report['replay'][name]
        need(type(x['states']) is int and x['states']==64 and type(x['sign_disagreements']) is int and x['sign_disagreements']==0, 'fixed64 development half/signs')
        need(all(type(x[k]) in (float,int) and math.isfinite(x[k]) and 0<=x[k]<=5e-5 for k in ('max_abs_error_m','RMS_error_m')), 'finite empirical error; never global bound')
    need(math.isfinite(report['point0_position_max_abs_error_m']) and report['point0_position_max_abs_error_m']>=0, 'recorded native pose diagnostic')
    return report

def validate_row(row, expected_id, rules):
    need(type(row['global_id']) is int and row['global_id']==expected_id and (row['seed'],row['proposal_id'])==divmod(expected_id,2048) and row['family']==row['proposal_id']%4, 'ordered original identity including last8191')
    n=row['finite_distance_count'];neg=row['negative_rows'];status=row['model_status'];gap=row['minimum_model_gap_m']
    need(type(n) is int and 0<=n<=9021 and status in STATES and isinstance(neg,list) and len(neg)==len(set(neg)) and all(type(x) is int and 0<=x<9021 for x in neg), 'all9021 last-row identity/negative indices')
    need(row['known_or_reserved']==((row['seed'],row['proposal_id']) in rules.known_or_reserved()) and row['eligible'] is False and row['native_status']==row['camera_status']=='NOT_TESTED', 'MODEL_ONLY no promotion')
    if n<9021:
        need(status=='MODEL_UNKNOWN' and gap is None, 'unknown remains unknown')
    else:
        need(type(gap) in (float,int) and math.isfinite(gap) and ((bool(neg) and gap<0 and status=='MODEL_NEGATIVE') or (not neg and gap>=0 and status=='MODEL_PREDICTED_INITIAL_CLEAR')), 'literal zero sign threshold and disposition')

def ready_inputs(value):
    need(value['schema']=='astra.report.increment05.inputs.v1' and value['status']=='READY_WITH_CLOSED_REAL_RESULTS' and value['template_only'] is False, 'future result template is not build authority')
    instant=datetime.datetime.fromisoformat(value['snapshot_utc']);need(instant.tzinfo is not None, 'explicit actual snapshot timezone')
    for name in ('formal_calibration_peer','model8192','model8192_terminal','schedule','model8192_DATA_peer','point0_finalize'):
        q=value[name] if name not in ('model8192_terminal','schedule_terminal') else value[name]['record']
        buffer(q,16*1024**2)
    return value
