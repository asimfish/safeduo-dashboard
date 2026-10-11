"""New metadata adapter boundaries only; actual failed audit is the positive input."""
import argparse,copy,datetime as dt,json
from pathlib import Path
import report_v1 as r
PACKET={'path': '/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352/astra_report_increment12_v1/inputs/ACTUAL_TERMINAL_INPUT_METADATA_PEER_V12.json', 'sha256': '6b9325abfb42cf9de5fe20e337da4e8c6834ef06fdff17a99cb086203bd15a0f'}
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);a=p.parse_args();b=r.binding()
    out=Path(a.out);b.need(not out.exists() and out.resolve().is_relative_to(r.O),'exclusive source tests')
    packet=b.read(PACKET);outcome=r.terminal_inputs(b,PACKET);h=b.read(r.STATIC['V8_FAILED_METADATA']);m=b.read(h['review']);epoch=dt.datetime.fromisoformat(packet['snapshot_utc'])
    b.need(outcome['V8_independent_FAILED_METADATA']['full_RAW_DATA_PASS'] is False and outcome['V8_independent_FAILED_METADATA']['camera_pixels_decoded']==0,'actual failed metadata-only input')
    checks=['real742c+parent2526+native/outer/tool−9/1/1+reviewer0; metadata-only positive']
    for name,field,value in [('RAW_promotion','full_RAW_DATA_PASS',True),('wrong_native_exit','native_actual_exit',0),('fabricated_group','camera_group_receipts',1)]:
        bad=copy.deepcopy(m);bad[field]=value
        try:r.metadata_scope(b,h,bad,packet,epoch)
        except ValueError:checks.append(name+' denied')
        else:raise ValueError('invalid metadata promotion admitted')
    try:r.metadata_scope(b,h,m,packet,dt.datetime.fromisoformat('2026-10-11T03:19:41.328001+00:00'))
    except ValueError:checks.append('peer after old cutoff denied')
    else:raise ValueError('later metadata backdated')
    out.mkdir(parents=True,exist_ok=False)
    b.write(out/'SOURCE_TEST_REPORT_V1.json',dict(status='PASS_ACTUAL_METADATA_SCOPE_ADAPTER',test_count=len(checks),new_checks=checks,real_input=r.STATIC['V8_FAILED_METADATA'],reused_prior_candidate=packet['prior_frozen_candidate'],synthetic_native_positive=False,new_science_or_math_tests=0,native_GPU_Git_W_actions=0,publication_performed=False))
    print(json.dumps(dict(status='PASS_ACTUAL_METADATA_SCOPE_ADAPTER',new_checks=len(checks))),flush=True)
if __name__=='__main__':main()
