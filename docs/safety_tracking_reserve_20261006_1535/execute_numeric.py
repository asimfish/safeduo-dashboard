"""Two owned GPU lanes, sequential fresh processes per lane."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE.parent/'safety_random_space_20261004'))
from isolated_campaign_v2 import run_campaign

if __name__=='__main__':
    lane=int(sys.argv[1]);blocks=[lane]
    registration=json.loads((HERE/'NUMERIC_REGISTRATION.json').read_text())
    for block in blocks:
        path=HERE/'plans'/f'holdout_{block}_plan.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==registration['plans'][str(path)]
        plan=json.loads(path.read_text())
        result=run_campaign(plan,plan['output_root'])
        print('CLOSED',block,result['status'],flush=True)
