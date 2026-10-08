"""Describe observed conditioned input coverage, without extrapolating safety probability."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
H=Path(__file__).resolve().parent
def load(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    reg=json.loads((H/'ACCEPTANCE_REGISTRATION.json').read_text());blocks=[];allq=[];alltape=[]
    for b in range(2):
        job=next(j for j in reg['jobs'] if j['id']==f'b{b}_zero_inclusive');root=Path(job['out']);x=load(root/'input_recipe.npz');q=x['sampled_initial'];limits=x['joint_soft_limits'];assert q.shape==(64,26)
        assert np.isfinite(q).all() and (q>=limits[...,0]).all() and (q<=limits[...,1]).all()
        normalized=(q-limits[...,0])/(limits[...,1]-limits[...,0]);hist=[np.histogram(normalized[:,j],bins=np.linspace(0,1,13))[0].tolist() for j in range(26)]
        tape=x['tape'][:960];assert tape.shape==(960,64,26) and not tape[:60].any();allq.append(q);alltape.append(tape)
        amp=x['segment_amplitudes'][:960];holds=x['holds'][:960];updates=x['updates'][:960];assign=load(root/'bank_assignment.npz')['risk_pair_index'];arms=[]
        for a,(start,stop) in enumerate([(0,7),(7,14),(14,20),(20,26)]):
            u=tape[60:,:,start:stop];h=holds[60:,:,a][updates[60:,:,a]];v=amp[60:,:,a][updates[60:,:,a]]
            arms.append(dict(arm=['F_L','F_R','U_L','U_R'][a],positive_fraction=float((u>0).mean()),negative_fraction=float((u<0).mean()),zero_fraction=float((u==0).mean()),command_abs_max=float(np.abs(u).max()),command_abs_quantiles=np.quantile(np.abs(u),[0,.25,.5,.75,.95,1]).tolist(),updated_segments=int(updates[60:,:,a].sum()),observed_hold_steps_unique=np.unique(h).tolist(),segment_amplitude_quantiles=np.quantile(v,[0,.25,.5,.75,.95,1]).tolist()))
        blocks.append(dict(block=b,bank_path=job['env']['SAFEDUO_INITIAL_BANK_NPZ'],bank_sha256=sha(job['env']['SAFEDUO_INITIAL_BANK_NPZ']),input_recipe_sha256=sha(root/'input_recipe.npz'),command_seed=int(job['argv'][job['argv'].index('--seeds')+1]),strata=[dict(label=int(l),cases=int((assign==l).sum())) for l in np.unique(assign)],initial_q_min_rad=q.min(0).tolist(),initial_q_max_rad=q.max(0).tolist(),initial_normalized_min=normalized.min(0).tolist(),initial_normalized_max=normalized.max(0).tolist(),initial_joint_histogram_12_bins=hist,initial_joint_bins_occupied=[sum(v>0 for v in row) for row in hist],initial_ee_world_or_local_raw_field='ee_initial from raw producer; no workspace-volume certification',initial_ee_min_m=x['ee_initial'].min(0).tolist(),initial_ee_max_m=x['ee_initial'].max(0).tolist(),command_arms=arms))
    q=np.concatenate(allq);assert len(np.unique(q,axis=0))==128
    out=dict(status='COMPLETE_OBSERVED_INPUT_COVERAGE',unique_requested_initials=128,independent_initial_banks=2,method_windows=512,random_control_env_frames=128*900,all26_joint_requested_span_rad=np.ptp(q,axis=0).tolist(),blocks=blocks,limitations=['initials conditioned on original sphere margins and raw-zero stability, not unconditional IID','no hand target randomization','mass/friction/gains/tablelayout/FIFO6 fixed, not domain randomization','only2 independent random seed blocks; correlated envs are not512 independent safety trials','joint-bin occupation and EE min/max are descriptive, not full workspace-volume coverage'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/'COVERAGE_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(out['status'],128,flush=True)
if __name__=='__main__':main()
