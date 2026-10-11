"""Report10 local four-mode gate; software browser with exact own wait closure."""
import argparse, hashlib, json, os, types
from pathlib import Path

H=Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
D=Path(__file__).resolve().parent
O=R/'astra_report_increment10_v1/repair1/local_browser_attempt1'
PY='/home/liyufeng/miniforge3/envs/safeduo/bin/python'
BPY='/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/bin/python'
BIND_SHA='bfd5ee3ed2cb6a871ced9be6181a850069402fd72adb0e7b4dc62bd4b2f7ef25'

def module(path,sha,name):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('authenticated exact helper buffer')
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--source-sha256',required=True);ap.add_argument('--build-handoff',required=True);ap.add_argument('--build-sha256',required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    b=module(H/'astra/report_increment05_v4/binding_v1.py',BIND_SHA,'_closed_binding')
    graph=b.read(b.ref(D/'SOURCE_GRAPH_V1.json'))
    b.need(b.sha(__file__)==a.source_sha256 and b.sha(D/'qa_report_v1.py')==graph['own_source_sha256'][str(D/'qa_report_v1.py')] and os.sys.dont_write_bytecode and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in b.THREADS),'exact own source and threads1 software CPU browser')
    q=dict(path=a.build_handoff,sha256=a.build_sha256);h=b.read(q);p=b.read(h['plan'])
    b.need(h['candidate']=='candidate10' and h['base_public_commit']=='8d7c8e864014eb6acba69e41631f9e6e609b67e0' and h['bank_count']==0 and h['safety_acceptance'] is False and h['manifest']==p['manifest'],'frozen exact10 no qualification')
    b.closed_wait(h['actual_worker_wait']);b.closed_wait(h['actual_wrapper_wait']);t=b.read(h['actual_tool_wait']);b.need(t['actual_exit']==0 and t['session_closed'],'real build top tool0')
    if a.run:
        loader=module(H/'astra/owned_discovery_race_v1/load_reviewed_core_v1.py','8c237366a9cc5a7ab7857472858806ea5c1645263bde5f6ed7e24b5f54a33512','_browser07_loader');core=loader.load_reviewed_core('82c5ed80a8e1e4f875e2314fc7c2fa465916269311102dfc1549088504260626')
        argv=[BPY,'-B',str(D/'qa_report_v1.py'),'--overlay',h['overlay'],'--plan',h['plan']['path'],'--plan-sha256',h['plan']['sha256'],'--manifest-sha256',h['manifest']['sha256'],'--mode','local','--out',str(O/'products')]
        w=core.bounded_child(argv,dict(os.environ),O/'worker',backend='CPU_REPORT10_LOCAL_SOFTWARE_BROWSER',max_wall=360,cpu_limit=False)
        b.closed_wait(b.ref(O/'worker/ACTUAL_WAIT.json'));return w['actual_wait_exit']
    O.mkdir(parents=True,exist_ok=False)
    observer=module(H/'astra/full74_admissible_pilot_namespace_v2/observe_namespace_v2.py','65503815d51c14cf94e42e41c01e63c8d01a8a150ee2a0ce77ad990e6d879fcc','_browser07_observer')
    observer.observe([PY,'-B',str(Path(__file__).resolve()),*os.sys.argv[1:],'--run'],O/'external')
    worker=b.ref(O/'worker/ACTUAL_WAIT.json');outer=b.ref(O/'external/ACTUAL_WAIT.json');b.closed_wait(worker);b.closed_wait(outer)
    hand=b.write(O/'CLOSED_LOCAL_BROWSER_V1.json',dict(status='CLOSED_REPORT10_LOCAL_BROWSER_PENDING_TOOL',candidate=q,source=b.ref(__file__),QA_source=b.ref(D/'qa_report_v1.py'),worker_wait=worker,outer_wait=outer,browser_report=b.ref(O/'products/BROWSER_REPORT_V1.json'),browser_AS='reviewed Chromium virtual-address reservation ASunlimited; software disable-gpu; owned360s timeout'))
    print(json.dumps(dict(handoff=hand,worker_wait=worker,wrapper_wait=outer)),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
