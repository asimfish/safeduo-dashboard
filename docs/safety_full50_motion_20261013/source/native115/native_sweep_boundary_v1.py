"""Small ctypes boundary; all numerical choices belong to the frozen registration."""
import ctypes
class NativeSweepBoundary:
    def __init__(self,config,path):
        self.library=ctypes.CDLL(config['library'])
        self.library.safeduo_sweep_start.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
        for name in ['safeduo_sweep_prepare','safeduo_sweep_abort']:
            getattr(self.library,name).argtypes=[]
        for name in ['safeduo_sweep_event_begin','safeduo_sweep_event_end']:
            getattr(self.library,name).argtypes=[ctypes.c_int]
        assert ctypes.cast(ctypes.CDLL(None).cuLaunchKernel,ctypes.c_void_p).value==ctypes.cast(self.library.cuLaunchKernel,ctypes.c_void_p).value,'Owned preload not globally bound'
        assert self.library.safeduo_sweep_start(str(path).encode(),config['extra_pass_count'],config['expected_native_sweeps_per_event'])==0
    def prepare(self):
        assert self.library.safeduo_sweep_prepare()==0,'Actual installed kernel identities and mimic layout not qualified'
    def begin(self,index):
        assert self.library.safeduo_sweep_event_begin(index)==0
    def end(self,index):
        assert self.library.safeduo_sweep_event_end(index)==0,'Native launch coverage/count/return gate failed'
    def close(self):
        assert self.library.safeduo_sweep_abort()==0
