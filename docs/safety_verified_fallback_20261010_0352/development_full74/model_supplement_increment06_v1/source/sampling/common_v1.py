"""Selected same-buffer source/data pins; own new outputs, CPU only."""
import os,sys,resource,json,hashlib,types,math,io
from pathlib import Path
H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
HERE=Path(__file__).resolve().parent
OUT=R/'astra_full74_supplemental_development_sampling_v2'
OLD=H/'astra/full8192_model_geometry_runner_v2'
PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
def need(v,s):
 if not v:raise ValueError(s)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def raw(r,limit=32*1024**2):
 need(type(r) is dict and set(r)=={'path','sha256'},'exact actual ref')
 p=Path(r['path']);need(p.is_absolute() and p.resolve()==p and (p.is_relative_to(H) or p.is_relative_to(R)),'local pinned path')
 with p.open('rb') as f:b=f.read(limit+1)
 need(len(b)<=limit and hashlib.sha256(b).hexdigest()==r['sha256'],'bounded SHA '+str(p));return b
def read(r):
 def unique(items):
  d={}
  for k,v in items:need(k not in d,'duplicate JSON');d[k]=v
  return d
 def reject(v):raise ValueError('nonfinite JSON '+v)
 def finite(v):
  x=float(v);need(math.isfinite(x),'nonfinite float');return x
 return json.loads(raw(r),object_pairs_hook=unique,parse_constant=reject,parse_float=finite)
def write(p,d):
 p=Path(p);need(p.resolve()==p and p.is_relative_to(OUT),'new own NAS only')
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 return ref(p)
def npy(p,a):
 import numpy as np
 p=Path(p);need(p.is_relative_to(OUT) and p.resolve()==p,'new own NPY')
 with p.open('xb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
 return ref(p)
def pins():return read(ref(HERE/'DEPENDENCY_PINS_V1.json'))
def protocol():return read(ref(HERE/'PROTOCOL_V1.json'))
def source_graph():
 r=ref(HERE/'SOURCE_GRAPH_V1.json');g=read(r)
 for p,d in g['files'].items():raw(dict(path=p,sha256=d))
 return r
def pinned(p):return dict(path=str(p),sha256=pins()['files'][str(p)])
def module(r,name):
 b=raw(r,2*1024**2)
 if name in sys.modules:
  m=sys.modules[name];need(m.__file__==r['path'] and getattr(m,'__verified_sha256__',None)==r['sha256'],'module identity');return m
 m=types.ModuleType(name);m.__file__=r['path'];m.__verified_sha256__=r['sha256'];sys.modules[name]=m
 try:exec(compile(b,r['path'],'exec'),m.__dict__)
 except BaseException:sys.modules.pop(name,None);raise
 return m
def support():return module(pinned(H/'astra/full74_clock3_reader_operations_v1/common_v1.py'),'_supplement_CPU_support')
def limits():
 need(sys.dont_write_bytecode and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU -B/noCUDA/threads1')
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def name(v):need(v and v.replace('_','').isalnum(),'new own output name');return v
