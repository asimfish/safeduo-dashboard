"""Freeze exact sources, commands and input hashes before prospective physics."""
from pathlib import Path
import json,hashlib,secrets,copy,sys
from datetime import datetime,timezone
import numpy as np
from random_input import make_recipe
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
OLD=HERE.parent/'safety_feasible_guard_20261005_2100'/'holdout_v2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
def main():
    sys.path.insert(0,str(HERE.parent.parent.parent/'src'))
    from safeduo.eval.research_battery import cell_seed
    directory=HERE/'registered';directory.mkdir(exist_ok=False);ROOT.mkdir(exist_ok=False)
    sources={str(p):sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.log')}
    rows=[]
    for block in range(3):
        old=json.loads((OLD/f'holdout_{block}_plan.json').read_text())
        for rel,h in old['source_sha256'].items():assert sha(Path(old['cwd'])/rel)==h,rel
        for path,h in old['research_source_sha256'].items():assert sha(path)==h,path
        assert sha(old['checkpoint_path'])==old['checkpoint_sha256']
        seed=secrets.randbelow(2**31-1);source_seed=cell_seed(seed,'risk_burst',.05)
        tape,info=make_recipe(source_seed);file=directory/f'recipe_{block}.npz'
        np.savez_compressed(file,tape=tape,**{k:v for k,v in info.items() if isinstance(v,np.ndarray)})
        root=ROOT/f'block{block}';jobs=[]
        for mode in ('admission_full','joint_reference'):
            job=copy.deepcopy(next(j for j in old['jobs'] if j['id'].startswith(mode+'_')))
            job['id']=f'{mode}_{seed}';job['mode']=mode
            argv=job['argv'];argv[1]=str(HERE/'observer_runner.py')
            argv[argv.index('--seeds')+1]=str(seed)
            device=f'cuda:{block%2}';argv[argv.index('--device')+1]=device
            argv+=['--out',str(root/job['id'])]
            job['expected']['seed']=seed;job['expected_args']['device']=device
            job['env']['SAFEDUO_JOINT_MODE']=mode
            jobs.append(job)
        plan={**{k:old[k] for k in ('cwd','source_sha256','checkpoint_path','checkpoint_sha256','env')},
              'research_source_sha256':{**old['research_source_sha256'],**sources,str(file):sha(file)},
              'jobs':jobs,'output_root':str(root),'command_seed':seed,'source_seed':source_seed,
              'command_tape_sha256':hashlib.sha256(tape.tobytes()).hexdigest(),
              'registered_recipe_path':str(file),'registered_recipe_sha256':sha(file),
              'registered_utc':datetime.now(timezone.utc).isoformat(),
              'new_commands':64,'new_initials':0,'observational_predictors':4,'control_enabled':False,
              'shared_machine':True,'unchanged_controller_implementation':True,'bitexact_repeatability_claimed':False}
        write(directory/f'block{block}.json',plan)
        rows.append(dict(block=block,command_seed=seed,source_seed=source_seed,device=device,tape_sha256=plan['command_tape_sha256']))
    assert len({r['command_seed'] for r in rows})==3
    write(directory/'SUMMARY.json',dict(rows=rows,new_commands=192,new_initials=0,method_windows=384,registered_utc=datetime.now(timezone.utc).isoformat()))
    print('FROZEN',rows,flush=True)
if __name__=='__main__':main()
