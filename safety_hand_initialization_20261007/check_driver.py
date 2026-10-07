import json, sys, torch, types
from pathlib import Path
p=Path(__file__).resolve().parent
for f in p.glob('*.py'): compile(f.read_text(),str(f),'exec')
sys.path.insert(0,str(p))
from absolute_hand_targets import AbsoluteHandTargets
meta=json.loads((p/'legacy_native_metadata.json').read_text())
class Art:
    def __init__(self,m):
        self.joint_names=m['joint_names']
        self.data=types.SimpleNamespace(soft_joint_pos_limits=torch.tensor(m['soft_limits_rad'][:1]))
        self.last=None
    def set_joint_position_target(self,q,joint_ids):self.last=q.clone()
env=types.SimpleNamespace(device='cpu',num_envs=1,_arms={arm:Art(m) for arm,m in meta['arms'].items()})
driver=AbsoluteHandTargets(env,.008333,meta,.35)
checks=[]
for fraction,thumb in [(.45,None),(.4769,.7569),(.4746,.7546),(.5264,None),(.5182,None)]:
    for arm in env._arms: driver.command(arm,fraction,.6,thumb)
    for _ in range(74):driver.step()
    for arm,m in meta['arms'].items():
        ids=m['hand_ids'];lim=env._arms[arm].data.soft_joint_pos_limits[0,ids]
        old=torch.tensor(m['hand_default_rad'][0])
        far=torch.where((lim[:,1]-old).abs()>=(lim[:,0]-old).abs(),lim[:,1],lim[:,0])
        fs=torch.tensor([thumb if thumb is not None and 'thumb' in m['joint_names'][i] else fraction for i in ids])
        expected=old+fs*(far-old)
        assert torch.equal(env._arms[arm].last[0],expected),(arm,fraction)
        checks.append(dict(arm=arm,fraction=fraction,thumb=thumb,closed_endpoint_bitexact=True))
for arm in env._arms:driver.command(arm,0,.6)
for _ in range(74):driver.step()
for arm in env._arms:assert torch.equal(env._arms[arm].last[0],driver.open_q[arm])
(p/'DRIVER_ENDPOINT_CHECK.json').write_text(json.dumps(dict(status='PASS_20_CLOSED_ENDPOINTS_AND_FOUR_OPEN_ENDPOINTS',checks=checks),indent=2)+'\n')
print('PASS: 20 absolute closed endpoints and 4 open endpoints bitexact; source syntax valid')
