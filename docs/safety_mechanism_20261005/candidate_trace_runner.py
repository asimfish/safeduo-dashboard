"""Passive exact replay of the frozen candidate on the old development seed."""
from reference_envelope import install
from trace_runner import MechanismTrace, risk


class CandidateTrace(MechanismTrace):
    def start(self, env):
        install(env, 'envelope_050')
        super().start(env)


risk.base.battery.EpisodeTrace=CandidateTrace
if __name__=='__main__':
    risk.base.battery.main()
