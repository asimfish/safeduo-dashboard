import sys,numpy as np,torch
from pathlib import Path
from risk_source import prepend_zero as fixed
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'safety_risk_strata_20261004'))
from risk_recipe import prepend_zero as old
for n in [2,3,32,59,60,61,902]:
 tape=torch.arange(n*2,dtype=torch.float32).reshape(n,1,2)
 info={k:np.arange(n*2).reshape(n,1,2) for k in ['updates','holds','segment_amplitudes']}
 previous,_=old(tape,info);current,meta=fixed(tape,info)
 assert len(current)==n+60 and not current[:60].any() and torch.equal(current[60:],tape)
 if n>=60:assert torch.equal(previous,current)
 else:assert len(previous)<len(current)
 for k in info:assert len(meta[k])==n+60 and np.array_equal(meta[k][60:],info[k])
print('PASS7 short/long lengths; old short negative control reproduced; full902 unchanged')
