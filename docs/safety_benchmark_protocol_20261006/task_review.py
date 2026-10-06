"""Recompute old outcome-table counts without relabelling or raw reanalysis."""
from pathlib import Path
import json,hashlib,csv
HERE=Path(__file__).resolve().parent
def main():
    inventory=json.loads((HERE/'EVIDENCE_INVENTORY.json').read_text())
    binding=next(i for i in inventory['source_bindings'] if i['path'].endswith('/analysis.json'))
    source=Path(binding['path']);blob=source.read_bytes();assert hashlib.sha256(blob).hexdigest()==binding['sha256']
    data=json.loads(blob);rows=data['outcomes'];assert len(rows)==192
    failures={o:{k:0 for k in rows[0]['objects'][o]['gates']} for o in rows[0]['objects']}
    exported=[];success=0
    for r in rows:
        objects=r['objects']
        for name,obj in objects.items():
            assert obj['pass']==all(obj['gates'].values())
            for key,value in obj['gates'].items():failures[name][key]+=int(not value)
        computed=all(v['pass'] for v in objects.values()) and r['official_object_qa_pass']
        assert r['task_pass']==computed;success+=int(computed)
        out=dict(id=f"b{r['block']}_env{r['slot']:03d}",split=r['split'],kind=r['kind'],task_pass=computed,
            official_qa_pass=r['official_object_qa_pass'],sphere_violation_steps=r['violation_steps'])
        out.update(r['parameters'])
        for name,obj in objects.items():
            out.update({name+':'+key:value for key,value in obj['gates'].items()})
            for key in ('final_xy_mm','final_tilt_deg','max_lift_mm','final_dz_mm'):out[name+':'+key]=obj[key]
        exported.append(out)
    assert success==data['totals']['pass']==19 and failures==data['totals']['failed_gates']
    assert source.read_bytes()==blob
    with (HERE/'prior_task_cases.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]),lineterminator='\n');writer.writeheader();writer.writerows(exported)
    protocol=Path(binding['path']).parent/'protocol.json';pr=json.loads(protocol.read_text())
    summary=dict(status='PASS_PRIOR_OUTCOME_TABLE_CROSSCHECK',scope='known historical outcome table, not new experiments or reanalysis of dense physics',
        source_sha256=binding['sha256'],tasks=192,task_pass=success,task_fail=192-success,
        failures_overlap=True,failed_gates=failures,old_design_factors=pr['factors'],
        old_pair_exposed_episodes=data['totals']['pair_exposed_episodes'],
        by_split=data['totals']['by_split'],by_kind=data['totals']['by_kind'],
        statistical_scope=data['statistics'],measurement_uncalibrated=True)
    with (HERE/'PRIOR_TASK_REVIEW.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    print('RECOMPUTED_PRIOR_TABLE',success,'/',len(rows),failures)
if __name__=='__main__':main()
