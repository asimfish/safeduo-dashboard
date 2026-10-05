"""Passive actual-FIFO one-step predictions; frozen preceding controllers."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE.parent/'safety_feasible_guard_20261005_2100'))
import guard_runner_v2 as base
from predictors import Predictor,NAMES
from random_input import make_recipe
from safeduo.baselines.base import stack_robot
from safeduo.safety.types import ARM_KEYS

class Source(base.admission.risk.RiskSource):
    def __init__(self,env,kind,amp,seed,steps):
        super().__init__(env,kind,amp,seed,steps)
        assert steps==960 and env.num_envs==64
        tape,info=make_recipe(seed)
        self.tape=torch.from_numpy(tape).to(env.device);self.info=info
        self.info['tape_sha256']=hashlib.sha256(tape.tobytes()).hexdigest()
        self.initial_meta['kind']='reused_qualified_192_initials_new_balanced_commands'

def make_flow(kind,env,amp,env_yaml):
    master=int(sys.argv[sys.argv.index('--seeds')+1])
    assert kind=='risk_burst'
    return Source(env,kind,amp,base.admission.risk.base.battery.cell_seed(master,kind,amp),960)

class Observer(base.GuardTrace):
    def start(self,env):
        super().start(env)
        assert self.mode in ('admission_full','joint_reference')
        model=json.loads((HERE/'model.json').read_text())
        self.models=[Predictor(n,model['coefficients']) for n in NAMES]
        self.pred_chunk=[];self.pred_receipts=[];self.pred_start=0
        for k in ('predicted_delta_q','delivered_model_target','passive_model_q','passive_model_qd'):
            self.frames[k]=[]
        old_rows=env._provider.rows_from
        def rows(out,body):
            value=old_rows(out,body)
            if out.active_idx.shape[1]==9021:self.full_J=value.J
            return value
        env._provider.rows_from=rows
        old_safety=env.safety_dist_out
        def observe():
            out=old_safety();state=env.scene_state()
            q=torch.cat([state.q[a] for a in ARM_KEYS],-1)
            v=torch.cat([state.qd[a] for a in ARM_KEYS],-1)
            delivered=torch.cat([env._evaluation_actuator_delay.queue.pending[0][a] for a in ARM_KEYS],-1)
            dt=state.dt;assert abs(dt-model['dt'])<1e-12
            dq=torch.stack([m.predict(q,v,delivered,dt) for m in self.models],1)
            rate=sum(torch.einsum('nmd,nd->nm',self.full_J[r],stack_robot(state.qd,r)) for r in ('F','U'))
            remainder=(-env._last_out.closing-rate)*dt
            pred=[]
            for i in range(4):
                delta=torch.einsum('nmd,nd->nm',self.full_J['F'],dq[:,i,:14])+torch.einsum('nmd,nd->nm',self.full_J['U'],dq[:,i,14:])
                pred.append(env._last_out.dists+delta+remainder)
            self.pred=torch.stack(pred,-1)
            assert torch.isfinite(self.pred).all()
            self.pred_pre_exempt=env._last_out.full_viol_exempt.detach().cpu().numpy().copy()
            self.pred_pre_d=env._last_out.dists.detach().cpu().numpy().copy()
            self.pred_J=self.full_J
            self.model_state=dict(predicted_delta_q=dq.detach().clone(),delivered_model_target=delivered.detach().clone(),
                                  passive_model_q=q.detach().clone(),passive_model_qd=v.detach().clone())
            self.pred_remainder=remainder.detach().cpu().numpy().copy()
            return out
        env.safety_dist_out=observe
        (self.out/'prediction_manifest.json').write_text(json.dumps(dict(
            models=NAMES,model_sha256=hashlib.sha256((HERE/'model.json').read_bytes()).hexdigest(),
            control_enabled=False,target_input='oldest actual pre-step FIFO item',
            full_body_remainder='(-full.closing-controlled_Jqd)*dt; constant-rate approximation',
            raw_future_state_used=False,new_initials=0,new_commands_per_bank=64),indent=2)+'\n')
        np.savez_compressed(self.out/'regime_recipe.npz',**{k:v for k,v in env._delta_src.info.items() if isinstance(v,np.ndarray)})

    def flush_predictions(self):
        if not self.pred_chunk:return
        root=self.out/'prediction_chunks';root.mkdir(exist_ok=True)
        stop=self.pred_start+len(self.pred_chunk);file=root/f'steps_{self.pred_start:04d}_{stop:04d}.npz'
        np.savez_compressed(file,**{k:np.stack([r[k] for r in self.pred_chunk]) for k in self.pred_chunk[0]})
        self.pred_receipts.append(dict(path=str(file.relative_to(self.out)),start=self.pred_start,stop=stop,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        self.pred_start=stop;self.pred_chunk=[]

    def step(self,env,t):
        super().step(env,t)
        for k,value in self.model_state.items():self.frames[k].append(value)
        before=self.frames['passive_model_q'][-1];actual=self.frames['q'][-1]-before
        delta=torch.einsum('nmd,nd->nm',self.pred_J['F'],actual[:,:14])+torch.einsum('nmd,nd->nm',self.pred_J['U'],actual[:,14:])
        self.pred_chunk.append(dict(predicted=self.pred.detach().cpu().numpy(),pre_d=self.pred_pre_d,
            pre_exempt=np.packbits(self.pred_pre_exempt,-1),post_d=env._last_out.dists.detach().cpu().numpy().copy(),
            post_exempt=np.packbits(env._last_out.full_viol_exempt.detach().cpu().numpy(),-1),
            hindsight_controlled_delta=delta.detach().cpu().numpy(),passive_rate_delta=self.pred_remainder))
        if len(self.pred_chunk)==32:self.flush_predictions()

    def write(self,*args,**kwargs):
        self.flush_predictions()
        (self.out/'prediction_receipts.json').write_text(json.dumps(dict(steps=self.pred_start,rows=9021,models=NAMES,chunks=self.pred_receipts),indent=2)+'\n')
        return super().write(*args,**kwargs)

original_design=base.admission.risk.base.battery.build_design
def build_design(args):
    result=original_design(args)
    for cell in result:
        cell.update(initial_sampling='reused_qualified_bank',temporal_sampling='zero60_balanced_mixed450_iid450')
    return result
base.admission.risk.base.battery.build_design=build_design
base.admission.risk.base.battery.make_flow=make_flow
base.admission.risk.base.battery.EpisodeTrace=Observer
if __name__=='__main__':base.admission.risk.base.battery.main()
