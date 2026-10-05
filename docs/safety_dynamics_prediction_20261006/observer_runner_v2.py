"""Capture the callable actually dereferenced by the frozen guard closure."""
import observer_runner as original
class Observer(original.Observer):
    def start(self,env):
        super().start(env)
        self.unwrapped_rows=self.original_rows
        def capture(out,body):
            value=self.unwrapped_rows(out,body)
            if out.active_idx.shape[1]==9021:self.full_J=value.J
            return value
        # GuardTrace.checked_rows calls this attribute, bypassing provider.rows_from.
        self.original_rows=capture
    def write(self,*args,**kwargs):
        try:return super().write(*args,**kwargs)
        finally:self.env._provider.rows_from=self.unwrapped_rows
original.base.admission.risk.base.battery.EpisodeTrace=Observer
if __name__=='__main__':original.base.admission.risk.base.battery.main()
