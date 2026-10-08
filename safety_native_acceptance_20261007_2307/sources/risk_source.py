"""Evaluation-local exact zero prefix, also valid for a short integration tape."""
import hashlib,numpy as np,torch
PREFIX=60
def prepend_zero(tape,info):
 prefix=torch.zeros((PREFIX,*tape.shape[1:]),dtype=tape.dtype,device=tape.device)
 tape=torch.cat([prefix,tape],0)
 out={**info,'kind':'zero_prefix_then_mixed_hold','zero_prefix_steps':PREFIX,'random_steps':len(tape)-PREFIX-2}
 for key in ['updates','holds','segment_amplitudes']:
  value=info[key];out[key]=np.concatenate([np.zeros((PREFIX,*value.shape[1:]),dtype=value.dtype),value],0)
 out['tape_sha256']=hashlib.sha256(tape.cpu().numpy().tobytes()).hexdigest()
 return tape,out
