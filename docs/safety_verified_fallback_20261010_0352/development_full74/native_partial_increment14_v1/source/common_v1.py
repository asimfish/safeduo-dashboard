"""Bound report metadata and original image bytes; no SDK or scientific re-audit."""
import datetime as dt, hashlib, json, os, types
from pathlib import Path
H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent
O=R/'astra_report_increment14_v1'
W=Path('/home/liyufeng/safeduo-dashboard-verified-fallback-20261010-0352')
REL='docs/safety_verified_fallback_20261010_0352/development_full74/'
PREFIX='native_partial_increment14_v1'
HEAD='9ca978756e97e28090a5e837df93d93612edf70e'
PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
BPY='/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/bin/python'
CORE_SHA='82c5ed80a8e1e4f875e2314fc7c2fa465916269311102dfc1549088504260626'
LOADER=H/'astra/owned_discovery_race_v1/load_reviewed_core_v1.py'
LOADER_SHA='8c237366a9cc5a7ab7857472858806ea5c1645263bde5f6ed7e24b5f54a33512'
OBSERVER=H/'astra/full74_admissible_pilot_namespace_v2/observe_namespace_v2.py'
OBSERVER_SHA='65503815d51c14cf94e42e41c01e63c8d01a8a150ee2a0ce77ad990e6d879fcc'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT')
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(path):
    p=Path(path);need(p.is_absolute() and not p.is_symlink(),'regular absolute artifact')
    h=hashlib.sha256()
    with p.open('rb') as f:
        a=os.fstat(f.fileno())
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
        z=os.fstat(f.fileno())
    need((a.st_dev,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'stable bytes/dev/size/mtime/ctime, no inode authority')
    return h.hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def buffer(q,cap=16*1024**2):
    need(set(q)=={'path','sha256'} and len(q['sha256'])==64,'exact artifact reference')
    p=Path(q['path']);need(p.is_absolute() and not p.is_symlink(),'regular bound artifact')
    with p.open('rb') as f:
        a=os.fstat(f.fileno());b=f.read(cap+1);z=os.fstat(f.fileno())
    need(len(b)<=cap and hashlib.sha256(b).hexdigest()==q['sha256'],'bounded exact authenticated buffer: '+str(p))
    need((a.st_dev,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'stable authenticated read')
    return b
def read(q):
    def unique(pairs):
        d={}
        for k,v in pairs:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    return json.loads(buffer(q),object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))
def write(p,j):
    p=Path(p);need(p.resolve().is_relative_to(O) or p.resolve().is_relative_to(D),'own metadata write')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return ref(p)
def copy(q,p,cap=16*1024**2):
    p=Path(p);need(p.resolve().is_relative_to(O),'own immutable overlay')
    raw=buffer(q,cap);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    need(sha(p)==q['sha256'],'original copy bytes');return ref(p)
def module(p,s,n):
    raw=buffer({'path':str(p),'sha256':s},2*1024**2);m=types.ModuleType(n);m.__file__=str(p);exec(compile(raw,str(p),'exec'),m.__dict__);return m
def absent(i):
    need(type(i['pid']) is int and type(i['startticks']) is int,'typed enrolled PID/startticks')
    try:fields=Path('/proc',str(i['pid']),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return True
    return int(fields[19])!=i['startticks']
def wait(q,expected=0,clean=True):
    w=read(q);need(w['waitpid_observed'] is True and w['child_pid']==w['waited_pid'],'actual owned wait4')
    need(type(w['raw_wait_status']) is int and type(w['actual_wait_exit']) is int and os.waitstatus_to_exitcode(w['raw_wait_status'])==w['actual_wait_exit']==expected,'raw/exit agree')
    need(w['owned_identities'] and all(i['live'] is False and absent(i) for i in w['owned_identities']),'registered owned identities freshly absent')
    need(sha(w['log'])==w['log_sha256'],'actual log SHA')
    if clean:need(not w['signals'] and not w['error'] and not w['resource_abort'],'clean actual CPU wait')
    return w
def tool(q,expected=0):
    t=read(q);need(t.get('session_closed') is True and type(t.get('actual_tool_exit')) is int and t['actual_tool_exit']==expected,'actual tool terminal')
    term=t.get('actual_terminal_result');need(isinstance(term,dict) and type(term.get('exit_code')) is int and term['exit_code']==expected,'observed terminal result')
    return t
def sources():
    g=read(ref(D/'SOURCE_GRAPH_V1.json'));need({k:ref(v['path']) for k,v in g['own_source_files'].items()}==g['own_source_files'],'own source exact graph')
    return g['own_source_files']
