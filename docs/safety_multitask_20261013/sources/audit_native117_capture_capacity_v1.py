from pathlib import Path
import hashlib,json
import numpy as np

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    paths={v:Path(read(H/f'fourhand_native_probe_v{v}/RESOURCE_EXECUTION_V1.json')['out']) for v in [115,117]}
    reg=read(H/'fourhand_native_probe_v117/REGISTRATION_V1.json');sources={}
    for f,h in reg['source_sha256'].items():assert sha(f)==h,f
    compared_fields=0;compared_values=0
    for name in ['native_parameters.npz','actual_initial_seed_binding.npz']:
        with np.load(paths[115]/name) as x,np.load(paths[117]/name) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:
                assert x[key].dtype==y[key].dtype and x[key].shape==y[key].shape and x[key].tobytes()==y[key].tobytes(),key
                compared_fields+=1;compared_values+=x[key].size
    indexes={v:read(paths[v]/'physics_chunks.json') for v in paths}
    assert len(indexes[115])==len(indexes[117]);events=0
    for a,b in zip(indexes[115],indexes[117]):
        assert (a['start'],a['stop'])==(b['start'],b['stop'])
        with np.load(paths[115]/a['path']) as x,np.load(paths[117]/b['path']) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:
                assert x[key].dtype==y[key].dtype and x[key].shape==y[key].shape and x[key].tobytes()==y[key].tobytes(),(a['start'],key)
                compared_fields+=1;compared_values+=x[key].size
        sources[str(paths[117]/b['path'])]=sha(paths[117]/b['path']);events+=b['stop']-b['start']
    p=read(paths[117]/'fourhand_protocol.json');old=read(paths[115]/'fourhand_protocol.json')
    assert events==832 and p['model_qualification_failure']==old['model_qualification_failure']
    ledger=paths[117]/'owned_native_sweep_launches_v1.jsonl';rows=ends=0
    with ledger.open() as f:
        for line in f:
            r=json.loads(line);rows+=1
            if r['kind']=='event_end':
                assert r['event']==ends and r['pass'] and r['native_mimic']==r['native_internal']==33 and r['extra']==528 and r['errors']==0
                ends+=1
    assert ends==events
    result={'status':'CAPTURE_CAPACITY_NATIVE_COMPATIBILITY_PASS','version':117,'native_exit':0,
        'baseline_version':115,'byte_equal_physics_events':events,'byte_equal_array_fields':compared_fields,
        'byte_equal_array_values':compared_values,'all_targets_states_contacts_and_raw_geometry_byte_equal':True,
        'native_ledger_event_ends':ends,'native_ledger_rows':rows,'original_failure_preserved':p['model_qualification_failure'],
        'capture_capacity':65536,'events_above8192_physically_executed':0,'long_native_duration_qualified':False,
        'original_model_gate_pass':False,'complete_task_successes':0,'fullSystem0_accepted':False,
        'source_sha256':{**sources,str(ledger):sha(ledger),str(H/'fourhand_native_probe_v117/REGISTRATION_V1.json'):sha(H/'fourhand_native_probe_v117/REGISTRATION_V1.json')}}
    output=K/'NATIVE117_CAPTURE_CAPACITY_COMPATIBILITY_V1.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
