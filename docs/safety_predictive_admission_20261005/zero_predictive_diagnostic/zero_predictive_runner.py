"""Combine unchanged frozen admission control with the registered zero source."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_zero_command_20261005'))
import zero_runner as neutral
sys.path.insert(0, str(HERE.parent / 'safety_predictive_admission_20261005'))
import predictive_runner as candidate


class NeutralPredictiveTrace(candidate.PredictiveTrace):
    def start(self, env):
        super().start(env)
        assert not env._delta_src.tape.any()
        manifest = json.loads((self.out / 'random_manifest.json').read_text())
        manifest.update(neutral_diagnostic=True, actual_input='all960 commands zero; legacy CLI amp.05 unused',
                        primary_endpoint='all16s; no pressure phase; keep all failures',
                        initial_banks_reused=True, new_random_command_windows=0,
                        neutral_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        unchanged_zero_source_sha256=hashlib.sha256(Path(neutral.__file__).read_bytes()).hexdigest())
        (self.out / 'random_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


candidate.risk.base.battery.EpisodeTrace = NeutralPredictiveTrace
if __name__ == '__main__':
    candidate.risk.base.battery.main()
