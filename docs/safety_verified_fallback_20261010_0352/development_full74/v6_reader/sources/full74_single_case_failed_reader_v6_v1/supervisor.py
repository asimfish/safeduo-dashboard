"""Own only real bounded CPU children, either failure tests or raw failed audit."""
from evidence_io import cpu_limits
cpu_limits()
import argparse,json
from evidence_io import HERE,OUTPUT,Evidence,require,write_new,sha,contained
from cpu_supervisor import wait_cpu
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--mode',choices=['tests','failed-raw'],required=True);p.add_argument('--out',required=True);p.add_argument('--binding');p.add_argument('--binding-sha256');a=p.parse_args()
    out=contained(a.out,OUTPUT)
    if a.mode=='tests':script=HERE/'review_v3.py';args=['--proof-dir',str(out/'proofs')];wall=120
    else:
        require(a.binding and a.binding_sha256,'explicit failed binding path/hash')
        script=HERE/'single_reader.py';args=['--binding',a.binding,'--binding-sha256',a.binding_sha256,'--output',str(out/'FAILED_DIAGNOSTIC.json')];wall=1800
    w=wait_cpu(script,args,out,wall)
    good=w['actual_wait_exit']==w['raw_wait_status']==0 and w['resource_abort'] is None and not w['signals']
    product=out/'proofs/CPU_TESTS.json' if a.mode=='tests' else out/'FAILED_DIAGNOSTIC.json'
    report=dict(schema='astra.single_case.failed_reader_delivery.v6.v1',status='CLOSED_CPU_TESTS' if a.mode=='tests' and good else 'CLOSED_FAILED_NATIVE_DIAGNOSTIC' if good else 'FAILED_CPU_READER',
        actual_wait=dict(path=str(out/'ACTUAL_WAIT.json'),sha256=sha(out/'ACTUAL_WAIT.json')),actual_exit=w['actual_wait_exit'],raw_status=w['raw_wait_status'],
        child_pid=w['child_pid'],startticks=w['child_identity']['startticks'],log_sha256=w['log_sha256'],source_binding=w['source_before'],
        product=dict(path=str(product),sha256=sha(product)) if product.is_file() else None,native_pass=False,data_pass=False,qualification=False,safety_acceptance=False)
    write_new(out/'CLOSED_DELIVERY.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_binding'}),flush=True)
    return 0 if good else 2
if __name__=='__main__':raise SystemExit(main())
