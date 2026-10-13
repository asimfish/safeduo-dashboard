from pathlib import Path
import ast,datetime,hashlib,json,shutil

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'spike/native_mimic_long_duration_v1'
N=K/'capture_capacity65536_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if N.exists():
        # Only a documented source-preparation assertion failed; no compilation
        # or native execution occurred. Verify its exact partial output.
        assert {p.name for p in N.iterdir()} == {'bridge.cpp'}
        assert (N/'bridge.cpp').read_text() == (B/'bridge.cpp').read_text().replace('8192','65536')
        (K/'CAPTURE_PREPARATION_FAILURE_V1.json').write_text(json.dumps({
            'actual_exit':1,'chunk_id':'1edc9e','reason':'Expected three capacity literals in CPU test; original has two',
            'native_launches':0,'partial_output_verified':True},indent=2)+'\n')
    contract={'iteration':'continuous_task_capture_capacity_20261013','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'capacity_before':8192,'capacity_after':65536,'maximum_implementation_candidates':2,'selected_implementation_candidates':1,
        'extra_native_passes_unchanged':16,'solver_algorithm_changes':0,'physical_threshold_changes':0,
        'native_validation':'Replay115 full832-event physical trace; known safety/model failure must remain',
        'long_native_duration_qualified':False,'fullSystem0_accepted':False}
    (K/'CAPTURE_CAPACITY_PROSPECTIVE_V1.json').write_text(json.dumps(contract,indent=2)+'\n')
    N.mkdir(exist_ok=True)
    old=(B/'bridge.cpp').read_text();assert old.count('8192')==5
    (N/'bridge.cpp').write_text(old.replace('8192','65536'))
    s=(B/'test_boundary.py').read_text();assert s.count('8192')==2
    (N/'test_boundary.py').write_text(s.replace('8192','65536'))
    ast.parse((N/'test_boundary.py').read_text())
    for name in ['fake_driver.cpp','native_boundary.py']:shutil.copyfile(B/name,N/name)
    record={**contract,'source_diff':'Only five8192 capacity literals become65536; reverse replacement byte-equals original bridge.cpp',
        'reverse_replacement_byte_equal':(N/'bridge.cpp').read_text().replace('65536','8192')==old,
        'source_sha256':{str(p):sha(p) for p in [B/'bridge.cpp',B/'test_boundary.py',N/'bridge.cpp',N/'test_boundary.py',K/'ADR_CONTINUOUS_TASK_CAPTURE_V1.md']}}
    assert record['reverse_replacement_byte_equal']
    (N/'CAPACITY_ONLY_SOURCE_DIFF_V1.json').write_text(json.dumps(record,indent=2)+'\n')
    print('CAPTURE_CAPACITY65536_PREPARED;physical algorithm unchanged;CPU and native gates pending',flush=True)


if __name__=='__main__':main()
