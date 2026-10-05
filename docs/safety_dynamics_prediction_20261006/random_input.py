"""State-independent balanced mixed/IID pressure; exact per-arm metadata."""
import numpy as np
WIDTHS=(7,7,6,6)
AMPS=np.array([.005,.015,.025,.05],np.float32)
HOLDS=np.array([1,4,15,30,90,180])
def make_recipe(seed):
    rng=np.random.default_rng(seed);n=64;length=450
    tape=np.zeros((962,n,26),np.float32);updates=np.zeros((962,n,4),bool)
    holds=np.zeros((962,n,4),np.int32);amplitudes=np.zeros((962,n,4),np.float32)
    regime=np.zeros((962,n),np.int8);order=rng.permutation(np.repeat([0,1],32))
    for e in range(n):
        for block in range(2):
            mixed=block==order[e];start=60+block*length
            regime[start:start+length,e]=1 if mixed else 2
            offset=0
            for arm,width in enumerate(WIDTHS):
                t=0
                while t<length:
                    hold=int(rng.choice(HOLDS)) if mixed else 1
                    stop=min(length,t+hold);amp=rng.choice(AMPS)
                    direction=rng.uniform(-1,1,width).astype(np.float32)
                    tape[start+t:start+stop,e,offset:offset+width]=direction*amp
                    updates[start+t,e,arm]=True
                    holds[start+t:start+stop,e,arm]=hold
                    amplitudes[start+t:start+stop,e,arm]=amp
                    t=stop
                offset+=width
    for value in (tape,updates,holds,amplitudes,regime):value[960:]=value[959]
    return tape,dict(updates=updates,holds=holds,segment_amplitudes=amplitudes,
                    regime=regime,order=order,kind='balanced_mixed450_iid450',
                    zero_prefix_steps=60,random_steps=900,seed=int(seed),
                    amplitudes_rad=AMPS.tolist(),mixed_holds=HOLDS.tolist(),state_feedback=False)
