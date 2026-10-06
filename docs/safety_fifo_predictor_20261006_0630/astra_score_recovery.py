"""Scheduling-only recovery of interrupted independent scoring; no parent math imports.

Each condition is freshly computed by the exact frozen independent score_cell.
The unchanged run_score routes calls to these futures in original order and
performs its original full-ledger recheck/aggregation. Checkpoints are outputs,
not cached inputs; this attempt does not reuse prior partial outcomes.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,importlib,json,os,sys,time
H=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(8*1024*1024):h.update(data)
    return h.hexdigest()

def merge_hashes(ledger,hashes):
    for path,digest in hashes.items():ledger._remember(Path(path),digest,None)

def routed_cell(module,ledger,entries,futures):
    def route(actual_ledger,plan,job,record):
        module.require(actual_ledger is ledger,'recovery ledger mismatch')
        key=(plan['output_root'],job['id'])
        module.require(key in entries,'recovery unknown condition')
        expected=entries[key]
        module.require((plan,job,record)==expected,'recovery arguments differ from registered future')
        cell,hashes,error=futures[key].result()
        merge_hashes(ledger,hashes)
        if error is not None:raise error
        return cell
    return route

def await_memory():
    while True:
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        if available>=25*1024**3:return
        print('RESOURCE_WAIT',available,flush=True);time.sleep(10)

def main():
    plan_path=H/'ASTRA_FINAL_RECOVERY_PLAN.json';reg=json.loads(plan_path.read_text())
    assert reg['status']=='REGISTERED_RECOVERY_BEFORE_ANY_RECOVERY_OUTCOMES'
    assert reg.get('workers')==4,'unregistered recovery worker bound'
    assert all(sha(p)==s for p,s in reg['sources'].items())
    assert not Path('/proc/1604874').exists(),'original scorer still exists'
    assert not (H/'ASTRA_FINAL_SCORE.json').exists(),'canonical score already exists'
    with (H/'astra_score_recovery_writer.lock').open('x') as f:json.dump(dict(pid=os.getpid(),source_sha256=sha(Path(__file__))),f)
    s=importlib.import_module('astra_final_score')
    s.require(sys.executable==reg['interpreter'] and s.np.__version__==reg['numpy'],'recovery runtime differs')
    ledger=s.Ledger();ledger.bind(plan_path)
    for path,digest in reg['sources'].items():ledger.bind(path,digest)
    design,plans=s.registration(ledger);s.bind_scoring_software(ledger)
    entries={}
    for plan in plans:
        for p,d in plan['source_sha256'].items():ledger.bind(Path(plan['cwd'])/p,d)
        for p,d in plan['research_source_sha256'].items():ledger.bind(p,d)
        ledger.bind(plan['checkpoint_path'],plan['checkpoint_sha256'])
        campaign=ledger.json(Path(plan['output_root'])/'campaign.json')
        s.require(campaign['plan']==plan and campaign['status']=='complete','noncanonical or reopened campaign')
        records={r['id']:r for r in campaign['jobs']}
        s.require(len(records)==len(campaign['jobs'])==4,'campaign job inventory')
        for job in plan['jobs']:
            record=records[job['id']];s.require(record['status']=='complete' and record['exit_code']==0,'condition incomplete')
            entries[plan['output_root'],job['id']]=(plan,job,record)
    s.require(len(entries)==12,'recovery inventory')
    original=s.score_cell;original_code=original.__code__;aggregation_code=s.run_score.__code__
    def compute(index,arg):
        await_memory();local=s.Ledger();started=time.monotonic();cell=None;error=None
        try:cell=original(local,*arg)
        except Exception as e:error=e
        snapshot=dict(index=index,id=arg[1]['id'],status='COMPUTED' if error is None else 'FAILED',result=cell,input_sha256=local.hashes,error=None if error is None else f'{type(error).__name__}: {error}',seconds=time.monotonic()-started)
        s.write_owned(f'astra_recovery_cell_{index:02d}.json',snapshot)
        print('RECOVERY_CELL_CLOSED',index,arg[1]['id'],snapshot['status'],flush=True)
        return cell,local.hashes,error
    started=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=reg['workers']) as pool:
            futures={key:pool.submit(compute,i,arg) for i,(key,arg) in enumerate(entries.items())}
            s.score_cell=routed_cell(s,ledger,entries,futures)
            try:result=s.run_score(design,plans,ledger)
            finally:s.score_cell=original
        s.require(original.__code__ is original_code and s.run_score.__code__ is aggregation_code,'original function code object changed')
        s.require(all(sha(p)==digest for p,digest in reg['sources'].items()),'recovery sources changed')
        s.write_owned('ASTRA_FINAL_RECOVERY_EXECUTION.json',dict(status='PASS_EXACT_FROZEN_INDEPENDENT_RECOVERY' if result['status']=='PASS_COMPLETE_INDEPENDENT_NUMERIC' else 'INCOMPLETE_INDEPENDENT_RECOVERY',utc=s.now(),numeric_status=result['status'],workers=reg['workers'],registered_order=[arg[1]['id'] for arg in entries.values()],registration_sha256=sha(plan_path),source_sha256=reg['sources'],original_score_cell_and_run_score_code_objects_unchanged=True,all12_fresh_condition_reads=True,previous_partial_outcomes_reused=False,private_ledgers_merged_without_conflict=True,original_full_raw_hash_recheck_preserved=True,original_attempt_complete_claimed=False,raw_physics_mutation=False,seconds=time.monotonic()-started))
        print('RECOVERY_COMPLETE',result['status'],flush=True)
    except Exception as e:
        s.write_owned('ASTRA_FINAL_RECOVERY_ERROR.json',dict(status='FAILED_RECOVERY',utc=s.now(),error=f'{type(e).__name__}: {e}',final_completion_claimed=False))
        raise
if __name__=='__main__':main()
