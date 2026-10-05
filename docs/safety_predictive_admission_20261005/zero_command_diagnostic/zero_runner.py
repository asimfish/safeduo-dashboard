"""Long neutral control on reused registered initial banks, no new random tapes."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'safety_mechanism_20261005'))
from mechanism_runner import MechanismTrace,risk

OriginalSource=risk.RiskSource
class ZeroSource(OriginalSource):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tape.zero_()
        for k in ('updates','holds','segment_amplitudes'):self.info[k][...]=0
        self.info.update(kind='constant_zero',hold_choices=[],amplitude_choices=[0.],hold_counts={},amplitude_counts={},
                         streams='no random command stream used; exact constant-zero tape',
                         tape_sha256=hashlib.sha256(self.tape.cpu().numpy().tobytes()).hexdigest())

class ZeroTrace(MechanismTrace):
    def start(self,env):
        super().start(env)
        assert not env._delta_src.tape.any()
        manifest=json.loads((self.out/'random_manifest.json').read_text())
        manifest.update(neutral_diagnostic=True,actual_input='all960 commands zero; legacy CLI amp.05 unused',
                        primary_endpoint='all16s; no pressure phase; keep all failures',
                        initial_banks_reused=True,new_random_command_windows=0,
                        neutral_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (self.out/'random_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

original_design=risk.base.battery.build_design
def design(args):
    out=original_design(args)
    for cell in out:cell['temporal_sampling']='constant_zero_all960_steps'
    return out

risk.RiskSource=ZeroSource
risk.base.battery.build_design=design
risk.base.battery.EpisodeTrace=ZeroTrace
if __name__=='__main__':risk.base.battery.main()
