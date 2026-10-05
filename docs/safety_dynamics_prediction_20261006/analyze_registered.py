"""Original frozen metric oracle, corrected launch-plan namespace only."""
from pathlib import Path
import json
from analyze import HERE,sha,inspect
def main():
    plans=[json.loads(f.read_text()) for f in sorted((HERE/'registered_launch').glob('block*.json'))]
    results=[];assert len(plans)==3
    for plan in plans:
        for path,h in plan['research_source_sha256'].items():assert sha(path)==h,path
        for rel,h in plan['source_sha256'].items():assert sha(Path(plan['cwd'])/rel)==h,rel
        root=Path(plan['output_root']);campaign=json.loads((root/'campaign.json').read_text())
        assert campaign['status']=='complete'
        inputs=[]
        for job in plan['jobs']:
            r,inp=inspect(root,job);results.append(r);inputs.append(inp)
            assert __import__('hashlib').sha256(inp['tape'].tobytes()).hexdigest()==plan['command_tape_sha256']
            print('AUDITED',job['id'],r['violations'],flush=True)
        import numpy as np
        assert np.array_equal(inputs[0]['q_initial'],inputs[1]['q_initial'])
        assert np.array_equal(inputs[0]['tape'],inputs[1]['tape'])
    out=dict(status='complete',method_windows=384,new_command_tapes=192,new_initials=0,rows=results,
             model_sha256=sha(HERE/'model.json'),control_enabled=False,
             scope='prospective passive point predictions; no conservative or physical-safety certification')
    with (HERE/'results.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print('COMPLETE',out['method_windows'],flush=True)
if __name__=='__main__':main()
