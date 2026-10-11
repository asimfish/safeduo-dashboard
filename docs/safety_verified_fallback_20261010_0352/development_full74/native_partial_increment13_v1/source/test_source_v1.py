"""Three adapter-boundary groups; no native fixture or mathematical safety claim."""
import argparse,copy,json
from pathlib import Path
import report_v1 as r
def rejected(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('invalid boundary accepted')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();b=r.binding()
    packet=b.read(b.ref(r.D/'FROZEN_INPUTS_V1.json'));p=b.read(packet['refs']['V9_parent']);n=b.read(packet['refs']['V9_initial_numeric'])
    r.validate_v9(p)
    for field,value in [('native_actual_exit',0),('completed_camera_group_receipts',1),('actual_physical_prefix_advance_count',12),('entry',{'status':'PASS'})]:
        q=copy.deepcopy(p);q[field]=value;rejected(lambda:r.validate_v9(q))
    scope=r.numeric_scope(n,packet['refs']['V9_parent']);assert scope['contact_freshness']=='UNKNOWN' and not scope['all32_initial_safe_inferred'] and not scope['all32_force_failure_inferred']
    for mutate in ('fresh','uncommitted','raw'):
        q=copy.deepcopy(n)
        if mutate=='fresh':next(x for x in q['fields'] if x['field']=='point_fresh')['max']=1
        elif mutate=='uncommitted':q['uncommitted_allocated_rows_examined']=True
        else:q['independent_sphere_math_recomputed']=True
        rejected(lambda:r.numeric_scope(q,packet['refs']['V9_parent']))
    parsed=[b.read(packet['refs'][key]) for key in ('V9_gallery','V9_initial_numeric','V9_parent')]
    assert all('\\u' not in x['caption_zh'] for x in parsed[0]['images'])
    assert parsed[0]['closed_native']==parsed[1]['closed_native']==packet['refs']['V9_parent']
    epoch=r.dt.datetime.fromisoformat(packet['snapshot_utc']);local=epoch.astimezone(r.dt.timezone(r.dt.timedelta(hours=8)))
    assert epoch.timestamp()==local.timestamp()
    r.inputs(b,b.ref(r.D/'FROZEN_INPUTS_V1.json'))  # actual schemas, including explicit parent live=False
    out=Path(a.out);b.need(not out.exists() and out.resolve().is_relative_to(r.O),'exclusive own tests');out.mkdir(parents=True)
    b.write(out/'SOURCE_TEST_REPORT_V1.json',dict(status='PASS_CPU_ADAPTER_BOUNDARIES_ONLY',groups=4,checks=['failed terminal cannot become complete native','initial contact UNKNOWN and uncommitted rows cannot be promoted','exact caption/ref and UTC/local identity','actual full metadata interface and live=False ownership'],negative_cases=7,science_fixtures_native_PASS=False,GPU_native_runs=0))
    print(json.dumps(dict(groups=4,negative_cases=7,status='PASS_CPU_ONLY')),flush=True)
if __name__=='__main__':main()
