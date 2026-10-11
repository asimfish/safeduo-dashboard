"""Only authenticated closed metadata and inherited HTML; no native/media reads."""
import hashlib, types
from pathlib import Path
H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent
O=R/'astra_report_increment17_v1'
_p=H/'astra/report_increment16_v1/actual_data_release3/common_v1.py'
_raw=_p.read_bytes()
if hashlib.sha256(_raw).hexdigest()!='1f897053b0cb833c6172409fecbbee26865f2b7d4572cd044a76e44447f131f6':raise ValueError('exact inherited metadata primitive')
_m=types.ModuleType('_closed16_metadata');_m.__file__=str(_p);exec(compile(_raw,str(_p),'exec'),_m.__dict__)
for _n in ('need','sha','ref','buffer','read','module','absent','wait','tool','PY','THREADS','LOADER','LOADER_SHA','CORE_SHA','OBSERVER','OBSERVER_SHA'):globals()[_n]=getattr(_m,_n)
_m.O=O;_m.D=D
write=_m.write
PREFIX='native_attempt_increment17_draft5_v1'
def sources():
    g=read(ref(D/'SOURCE_GRAPH_V1.json'))
    need({n:ref(q['path']) for n,q in g['own_source_files'].items()}==g['own_source_files'],'Report17 own source fixed')
    need(ref(g['inherited_metadata_primitive']['path'])==g['inherited_metadata_primitive'],'inherited metadata primitive unchanged')
    return g
