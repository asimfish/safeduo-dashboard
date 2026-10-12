"""Versioned guard candidate; preserves V1 and changes command continuity only."""
from pathlib import Path
import ast
import hashlib
import json

P=Path(__file__).resolve().parent


def once(s,old,new):
    assert s.count(old)==1,old[:80]
    return s.replace(old,new)


def main():
    dest=P/'actual_state_arm_guard_v2.py';assert not dest.exists()
    source=P/'actual_state_arm_guard_v1.py';s=source.read_text()
    s=once(s,'command_rate_rad_s=.25, backup_rate_rad_s=.15, reserve_m=.0005,',
             'command_rate_rad_s=.10, backup_rate_rad_s=.05, command_acceleration_rad_s2=.5, reserve_m=.0005,')
    s=once(s,'        self.dt,self.fifo=control_dt_s,fifo_controls',
             '        self.dt,self.fifo=control_dt_s,fifo_controls\n        if not np.isfinite(command_acceleration_rad_s2) or command_acceleration_rad_s2<=0:raise ValueError("invalid command acceleration")\n        self.acceleration=command_acceleration_rad_s2')
    s=once(s,'    def decide(self, full_q, full_qd, applied_from_fifo, pending_fifo, actual_gaps):',
             '    def decide(self, full_q, full_qd, applied_from_fifo, pending_fifo, actual_gaps, previous_command_target, previous_command_velocity):')
    anchor='        source_now=np.asarray(self.gap_function({a:root[a][None] for a in ARMS}),float)'
    s=once(s,anchor,'''        previous={a:np.asarray(previous_command_target[a],float) for a in ARMS}
        previous_velocity={a:np.asarray(previous_command_velocity[a],float) for a in ARMS}
        for a in ARMS:
            if (previous[a].shape!=root[a].shape or previous_velocity[a].shape!=root[a].shape or
                    not np.isfinite(previous[a]).all() or not np.isfinite(previous_velocity[a]).all()):
                raise ValueError('exact previous actuator command and command velocity required')
        def smooth_target(goal, rate):
            target={}
            for a in ARMS:
                wanted=np.clip((goal[a]-previous[a])/self.dt,-rate,rate)
                velocity=np.clip(wanted,previous_velocity[a]-self.acceleration*self.dt,
                                  previous_velocity[a]+self.acceleration*self.dt)
                # Saturation is final, including invalid prior velocity diagnostics.
                velocity=np.clip(velocity,-self.rate,self.rate)
                target[a]=previous[a]+velocity*self.dt
            return target
'''+anchor)
    old="            target={a:root[a]+fraction*np.clip(applied[a]-root[a],-self.rate*self.dt,self.rate*self.dt) for a in ARMS}"
    new="            goal={a:previous[a]+fraction*(applied[a]-previous[a]) if fraction else root[a] for a in ARMS}\n            target=smooth_target(goal,self.rate)"
    s=once(s,old,new)
    s=once(s,"        candidates.append({a:root[a]+np.clip(self.retreat[a]-root[a],-self.backup_rate*self.dt,self.backup_rate*self.dt) for a in ARMS})",
             "        candidates.append(smooth_target(self.retreat,self.backup_rate))")
    s=once(s,"            target=root;mode='UNQUALIFIED_BRAKE';alpha=0.;forecast=float(minima[5])",
             "            target=candidates[5];mode='UNQUALIFIED_BRAKE';alpha=0.;forecast=float(minima[5])")
    # Restore float32 actuator quantization after the bounded double-precision step.
    s=s.replace('Online simulation candidate after the real FIFO, not a safety certificate.',
                'Online V2: previous-command slew/acceleration plus actual state and FIFO; not a safety certificate.')
    ast.parse(s);dest.write_text(s)
    record={'changed_scope':'Post-FIFO command continuity and lower arm command delta; forecast and original native gates unchanged',
            'command_rate_rad_s':.10,'backup_rate_rad_s':.05,'command_acceleration_rad_s2':.5,
            'old_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'new_source_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'native_executed':0,'hypothesis_not_proven':'Reduced discontinuity may improve loaded mimic qd convergence; native109 thumb spike cause is not established',
            'fullSystem0_accepted':False}
    (P/'SLEW_GUARD_CHANGE_SCOPE_V2.json').write_text(json.dumps(record,indent=2)+'\n')
    print('SLEW_GUARD_V2_PREPARED_ORIGINAL_V1_PRESERVED',flush=True)


if __name__=='__main__':main()
