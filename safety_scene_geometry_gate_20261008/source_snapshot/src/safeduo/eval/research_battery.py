"""Frozen, matched-method simulation campaign with per-episode evidence.

Pure uniform noise uses a fixed command tape, identical across methods.
Workspace and conflict sources are state-feedback policies: their random seed
is matched, but their commands may diverge after the methods change the state.
No contact sensor or task-success claim is made by this sphere-layer battery.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from safeduo.eval.random_battery import (
    ARM_KEYS, CLASS_KEYS, aggregate, make_flow, run_window, summarize_window,
)
from safeduo.eval.coverage_design import build_schedule
from safeduo.eval.coverage_source import PairStratifiedDelta
from safeduo.eval.perturbations import InitialPoseSource, TargetDelayAdapter
from safeduo.safety.types import DOF_OF, DeltaCmd


METHODS = ('raw', 'backstop_only', 'system0')
FLOWS = ('uniform_random', 'held_random', 'l1_full', 'directed_all', 'pair_stratified')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_seed(seed, flow, amp):
    """Stable across method ordering, subsets, Python hash randomization and restart."""
    key = f'research-battery-v1|{seed}|{flow}|{float(amp):.9g}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little') % (2**31 - 1)


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')
    tmp.replace(path)


def verify_backstop_config(declared, actual):
    """Reject silently ignored declared solver options before any campaign cell."""
    effective=asdict(actual)
    for key,value in declared.items():
        if key not in effective or value!=effective[key]:
            raise ValueError(f'backstop runtime config mismatch: {key}={value!r}, effective={effective.get(key)!r}')
    return effective


class UniformRandomTape:
    """Uniform increments for all joints; optional hold, without goal guidance."""
    def __init__(self, n_envs, steps, amp, seed, device='cpu', hold_steps=1):
        if hold_steps < 1 or int(hold_steps) != hold_steps:
            raise ValueError('hold_steps must be a positive integer')
        gen = torch.Generator(device='cpu').manual_seed(seed)
        length = steps + 2
        draws = (torch.rand((length + hold_steps - 1) // hold_steps, n_envs,
                            sum(DOF_OF.values()), generator=gen) * 2 - 1) * amp
        self.tape = draws.repeat_interleave(hold_steps, dim=0)[:length]
        self.sha256 = hashlib.sha256(self.tape.numpy().tobytes()).hexdigest()
        self.tape = self.tape.to(device)
        self.index = 0

    def reset(self, env_ids, generator=None):
        if env_ids.numel() != self.tape.shape[1]:
            raise ValueError('research command tape only supports full-window resets')
        self.index = 0

    def sample(self, state):
        values = self.tape[self.index]
        self.index += 1
        return DeltaCmd(delta_q=dict(zip(ARM_KEYS, values.split([DOF_OF[a] for a in ARM_KEYS], dim=-1))))


class EpisodeTrace:
    """Log executed commands and measured state on GPU; transfer once after each cell."""
    def __init__(self, causal=False, kinematic=False, spheres=False):
        self.causal = causal
        self.kinematic = kinematic
        self.spheres = spheres
        if kinematic and not causal:
            raise ValueError('kinematic trace requires causal trace')

    def start(self, env):
        state = env.scene_state()
        self.q0 = torch.cat([state.q[a] for a in ARM_KEYS], -1).clone()
        self.limits = torch.cat([env._q_soft_limits[a] for a in ARM_KEYS], -2).clone()
        self.initial_violation = env._last_out.violation.clone()
        delay=getattr(env,'_evaluation_actuator_delay',None)
        if delay is not None:
            delay.reset(env._targets)
        self.frames = {key: [] for key in ('q', 'ee', 'cmd', 'exec', 'margins', 'official_margins',
                                         'official_deep', 'alpha', 'bs_active')}
        if delay is not None and delay.queue.steps:
            self.frames['actuator_target'] = []
            self.frames['controller_target'] = []
        if self.causal:
            for key in ('pre_q', 'pre_qd', 'pre_target', 'pre_row_d', 'pre_row_dmin',
                        'pre_row_rate', 'pre_row_backlog', 'pre_row_cls', 'pre_row_valid',
                        'pre_row_id', 'pre_row_exempt', 'solver_residual',
                        'projected_margin_delta', 'applied_margin_delta'):
                self.frames[key] = []
        if self.kinematic:
            from safeduo.safety.geometry import IsaacGeometryProvider
            self.full_joint_idx = {
                a: torch.arange(env._arms[a].num_joints, device=env.device)
                for a in ARM_KEYS
            }
            self.full_provider = IsaacGeometryProvider(
                env._arms, self.full_joint_idx, env._sph, env.device, jacobian_reference='link')
            for key in ('pre_row_body_rate', 'pre_row_full_rate', 'pre_row_com_rate',
                        'pre_joint_qd_all', 'pre_sphere_centers'):
                self.frames[key] = []
        if self.kinematic or self.spheres:
            self.frames['sphere_centers'] = []
            self.kinematic_meta = {
                'joint_names': {a: env._arms[a].joint_names for a in ARM_KEYS},
                'controlled_joint_idx': {a: env._joint_idx[a].tolist() for a in ARM_KEYS},
                'pair_sphere_idx': env._sph.pair_table.cpu().tolist(),
                'sphere_names': env._sph.qualified_names,
                'sphere_radii_m': env._sph.radii.cpu().tolist(),
                'robot_pair_count': env._sph._slice_table.start,
                'sphere_arm_id': env._sph.arm_id.cpu().tolist(),
            }

    def before_step(self, env, t):
        if not self.causal:
            return
        from safeduo.baselines.base import stack_robot
        state = env.scene_state()
        out = env.safety_dist_out()
        rows = env._provider.rows_from(out, env._body_pos_cache)
        self.pre_rows = rows
        self.pre_targets = {a: env._targets[a].detach().clone() for a in ARM_KEYS}
        backlog = {a: env._targets[a] - state.q[a] for a in ARM_KEYS}
        rate = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(state.qd, r))
                   for r in ('F', 'U'))
        stored = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(backlog, r))
                     for r in ('F', 'U'))
        snapshot = {
            'pre_q': torch.cat([state.q[a] for a in ARM_KEYS], -1),
            'pre_qd': torch.cat([state.qd[a] for a in ARM_KEYS], -1),
            'pre_target': torch.cat([env._targets[a] for a in ARM_KEYS], -1),
            'pre_row_d': rows.d, 'pre_row_dmin': rows.d_min,
            'pre_row_rate': rate, 'pre_row_backlog': stored,
            'pre_row_cls': rows.cls, 'pre_row_valid': rows.valid,
            'pre_row_id': out.active_idx,
            'pre_row_exempt': out.viol_exempt,
        }
        if self.kinematic:
            full_rows = self.full_provider.rows_from(out, env._body_pos_cache)
            com_rows = self.full_provider.rows_from(out, {
                a: env._arms[a].data.body_com_pos_w for a in ARM_KEYS})
            full_qd = {a: env._arms[a].data.joint_vel for a in ARM_KEYS}
            snapshot.update(
                pre_row_body_rate=-out.active_pairs[..., 1],
                pre_row_full_rate=sum(torch.einsum(
                    'nmd,nd->nm', full_rows.J[r], stack_robot(full_qd, r)) for r in ('F', 'U')),
                pre_row_com_rate=sum(torch.einsum(
                    'nmd,nd->nm', com_rows.J[r], stack_robot(full_qd, r)) for r in ('F', 'U')),
                pre_joint_qd_all=torch.cat([full_qd[a] for a in ARM_KEYS], -1),
                pre_sphere_centers=env._sph.last_centers,
            )
        for key, value in snapshot.items():
            self.frames[key].append(value.detach().clone())

    def step(self, env, t):
        state, cache = env.scene_state(), env._step_cache
        table = env._sph.last_table_margin.masked_fill(env._sph.last_table_viol_exempt, float('inf')).amin(-1)
        margins = torch.stack([env._last_out.min_margin[k] for k in CLASS_KEYS], -1)
        official = margins.clone()
        official[:, -1] = table
        frame = {
            'q': torch.cat([state.q[a] for a in ARM_KEYS], -1),
            'ee': torch.stack([state.ee_pos[a] for a in ARM_KEYS], 1),
            'cmd': torch.cat([cache['cmd'].delta_q[a] for a in ARM_KEYS], -1),
            'exec': torch.cat([cache['exec'].delta_q[a] for a in ARM_KEYS], -1),
            'margins': margins,
            'official_margins': official,
            'official_deep': official < -0.005,
            'alpha': cache['alpha'], 'bs_active': cache['bs_active'],
        }
        for key, value in frame.items():
            self.frames[key].append(value.detach().clone())
        if 'actuator_target' in self.frames:
            delay=env._evaluation_actuator_delay
            self.frames['actuator_target'].append(torch.cat([delay.applied[a] for a in ARM_KEYS],-1).clone())
            self.frames['controller_target'].append(torch.cat([env._targets[a] for a in ARM_KEYS],-1).clone())
        if self.causal:
            from safeduo.baselines.base import stack_robot
            target_delta = {a: env._targets[a] - self.pre_targets[a] for a in ARM_KEYS}
            for key, delta in [('projected_margin_delta', cache['exec'].delta_q),
                               ('applied_margin_delta', target_delta)]:
                value = sum(torch.einsum('nmd,nd->nm', self.pre_rows.J[r], stack_robot(delta, r))
                            for r in ('F', 'U'))
                self.frames[key].append(value.detach().clone())
            self.frames['solver_residual'].append(torch.stack(
                [cache['backstop_residual_F'], cache['backstop_residual_U']], -1).detach().clone())
        if self.kinematic or self.spheres:
            self.frames['sphere_centers'].append(env._sph.last_centers.detach().clone())

    def write(self, path, acc, meta, episode_meta=None):
        data = {key: torch.stack(frames).cpu().numpy() for key, frames in self.frames.items()}
        data['q_initial'] = self.q0.cpu().numpy()
        data['joint_soft_limits'] = self.limits.cpu().numpy()
        data['initial_violation'] = self.initial_violation.cpu().numpy()
        for key in ('q', 'ee', 'cmd', 'exec', 'margins', 'alpha'):
            if not np.isfinite(data[key]).all():
                raise ValueError(f'non-finite {key}: invalidate cell {meta}')
        data['meta_json'] = np.array(json.dumps(meta))
        if self.kinematic or self.spheres:
            data['kinematic_meta_json'] = np.array(json.dumps(self.kinematic_meta))
        np.savez_compressed(path, **data)
        cmd, exe, q = data['cmd'], data['exec'], data['q']
        intervention = np.max(np.abs(cmd - exe), -1) > 1e-7
        active_cmd = np.linalg.norm(cmd, axis=-1) > 1e-7
        zero_exec = np.linalg.norm(exe, axis=-1) <= 1e-7
        q_full = np.concatenate([data['q_initial'][None], q], axis=0)
        joint_path = np.abs(np.diff(q_full, axis=0)).sum(axis=(0, 2))
        rows = []
        limits = data['joint_soft_limits']
        spans = limits[..., 1] - limits[..., 0]
        episode_meta = episode_meta or [{} for _ in range(q.shape[1])]
        if len(episode_meta) != q.shape[1]:
            raise ValueError('episode metadata must have one row per environment')
        for e in range(q.shape[1]):
            rows.append({
                **meta, 'env_id': e,
                **episode_meta[e],
                'initial_violation': bool(data['initial_violation'][e]),
                'initial_q_sha256': hashlib.sha256(data['q_initial'][e].tobytes()).hexdigest(),
                'violation': bool(acc['viol_any'][e] > 0),
                'violation_steps': int(acc['viol_any'][e]),
                'violation_by_class': {k: bool(acc['viol_cls'][k][e] > 0) for k in CLASS_KEYS},
                'damaging_by_class': dict(zip(CLASS_KEYS, data['official_deep'][:, e].any(0).tolist())),
                'damaging': bool(data['official_deep'][:, e].any()),
                'min_margin_m': dict(zip(CLASS_KEYS, data['margins'][:, e].min(0).tolist())),
                'min_nonexempt_margin_m': dict(zip(CLASS_KEYS, data['official_margins'][:, e].min(0).tolist())),
                'pair_warn_steps': acc['pair_warn'][e].cpu().tolist(),
                'min_pair_margin_m': acc['min_pair'][e].cpu().tolist(),
                'intervention_step_rate': float(intervention[:, e].mean()),
                'backstop_step_rate': float(data['bs_active'][:, e].any(-1).mean()),
                'zero_exec_with_intent_rate': float((active_cmd[:, e] & zero_exec[:, e]).mean()),
                'command_l2_sum': float(np.linalg.norm(cmd[:, e], axis=-1).sum()),
                'executed_l2_sum': float(np.linalg.norm(exe[:, e], axis=-1).sum()),
                'measured_joint_path_rad': float(joint_path[e]),
                'joint_range_fraction_mean': float(((q_full[:, e].max(0) - q_full[:, e].min(0)) / (spans[e] if spans.ndim == 2 else spans)).mean()),
                'ee_aabb_min_m': data['ee'][:, e].min(0).tolist(),
                'ee_aabb_max_m': data['ee'][:, e].max(0).tolist(),
                'command_sha256': hashlib.sha256(cmd[:, e].tobytes()).hexdigest(),
            })
        return rows


def build_design(args):
    return [{'method': method, 'flow': flow, 'amp': amp, 'seed': seed,
             'initial_jitter_rad': jitter, 'actuator_delay_steps': delay,
             'initial_seed': cell_seed(seed,'initial_pose',jitter),
             'source_seed': cell_seed(seed, flow, amp)}
            for flow, amp, seed, jitter, delay, method in itertools.product(
                args.flows,args.amps,args.seeds,args.init_jitter_rad,args.actuator_delay_steps,args.methods)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--env-yaml', default='duo_env_a31.yaml')
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--duration-s', type=float, default=10.)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--amps', nargs='+', type=float, default=[0.005, 0.015])
    parser.add_argument('--flows', nargs='+', choices=FLOWS, default=list(FLOWS))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--out', required=True)
    parser.add_argument('--log-every', type=int, default=600)
    parser.add_argument('--causal-trace', action='store_true',
                        help='record pre-step target/velocity/rows and post-projection residuals')
    parser.add_argument('--kinematic-trace', action='store_true',
                        help='also compare selected/all joint Jacobian rates with body velocity rates')
    parser.add_argument('--hold-steps', type=int, default=30,
                        help='held_random repeats each independently sampled increment for this many control steps')
    parser.add_argument('--sphere-trace', action='store_true',
                        help='record measured sphere centers without full kinematic diagnostics')
    parser.add_argument('--init-jitter-rad',nargs='+',type=float,default=[0.],
                        help='uniform +/- per arm joint about default pose, soft-limit clipped; no collision rejection')
    parser.add_argument('--actuator-delay-steps',nargs='+',type=int,default=[0],
                        help='delay post-safety target delivery by this many control steps; measurements remain current')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.kinematic_trace and not args.causal_trace:
        parser.error('--kinematic-trace requires --causal-trace')
    if args.num_envs <= 0 or args.duration_s <= 0 or min(args.amps) <= 0 or args.hold_steps < 1:
        parser.error('environment count, duration and amplitudes must be positive')
    if min(args.init_jitter_rad)<0 or min(args.actuator_delay_steps)<0:
        parser.error('jitter and actuator delay must be nonnegative')
    if max(args.init_jitter_rad)>0 and any(f not in ('uniform_random','held_random') for f in args.flows):
        parser.error('initial-pose perturbation currently requires an exact random command tape')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        parser.error('output already contains a protocol; use a new directory to preserve evidence')
    from safeduo.configs import load_config, repo_root
    config = load_config(args.env_yaml)
    source_files = sorted((repo_root() / 'src/safeduo').rglob('*.py'))
    source_files += sorted((repo_root() / 'src/safeduo/configs').glob('*.yaml'))
    protocol = {'schema': 'safeduo.research_battery.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'args': vars(args), 'checkpoint_sha256': digest(args.ckpt), 'resolved_config': config,
                'source_sha256': {str(p.relative_to(repo_root())): digest(p) for p in source_files},
                'python': platform.python_version(), 'torch': torch.__version__,
                'design': build_design(args), 'completed_cells': 0, 'status': 'planned',
                'statistics': 'empirical window counts; seed/distribution strata; no IID safety certification',
                'measurement': 'sphere layer, table exemption-aware; damaging = nonexempt margin < -0.005m',
                'perturbation': 'uniform independent joint jitter, no collision rejection; post-safety target latency, current measurements; initial violations reported separately',
                'comparison': 'uniform_random/held_random exact command tape; other flows matched random seeds with state feedback'}
    atomic_json(out / 'protocol.json', protocol)
    app = AppLauncher(args).app
    from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
    from safeduo.eval.block1_harness import PolicyDriver, PassthroughDriver
    from safeduo.eval.clutch import wrap_clutch
    from safeduo.eval.clutch_eval import SAFE_LINES
    from safeduo.eval.endurance_eval import _RawShim, ckpt_arm_aware, ckpt_p2_obs
    cfg = make_duo_env_cfg(num_envs=args.num_envs, device=args.device, yaml_name=args.env_yaml,
                          coordinator=True, arm_aware_obs=ckpt_arm_aware(args.ckpt), p2_obs=ckpt_p2_obs(args.ckpt))
    cfg.coordinator['terminate_on_violation'] = False
    cfg.episode_length_s = args.duration_s + 5
    cfg.seed = 0
    env = DuoEnv(cfg)
    dt = cfg.sim.dt * cfg.decimation
    steps = round(args.duration_s / dt)
    if env._target_guard_enabled or env._target_rebase_on_safety:
        raise ValueError('raw ablation requires target guard/rebase disabled; freeze a matching configuration')
    backstop = env._backstop
    protocol['effective_backstop']=verify_backstop_config(config['safety']['backstop'],backstop.cfg)
    default_positions={a:env._arms[a].data.default_joint_pos[:,env._joint_idx[a]].clone() for a in ARM_KEYS}
    base = PolicyDriver(args.ckpt, device=str(env.device))
    protocol.update(status='running', dt=dt, steps=steps,
                    effective_coordinator=dict(cfg.coordinator),
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')
    atomic_json(out / 'protocol.json', protocol)
    windows, all_episodes = [], []
    try:
        for index, cell in enumerate(protocol['design']):
            method = cell['method']
            env._backstop = _RawShim(backstop) if method == 'raw' else backstop
            torch.manual_seed(cell['source_seed'])
            env._gen.manual_seed(cell['source_seed'])
            env._pending_cmd = None
            env._delta_src = (UniformRandomTape(env.num_envs, steps, cell['amp'], cell['source_seed'], env.device,
                                              hold_steps=args.hold_steps if cell['flow'] == 'held_random' else 1)
                              if cell['flow'] in ('uniform_random', 'held_random') else (
                                  PairStratifiedDelta(env.num_envs, cell['amp'], cell['seed'],
                                                      env_yaml=args.env_yaml, device=env.device,
                                                      cells=build_schedule(env.num_envs, cell['amp'], cell['seed']))
                                  if cell['flow'] == 'pair_stratified' else
                                  make_flow(cell['flow'], env, cell['amp'], args.env_yaml)))
            if cell['initial_jitter_rad']:
                env._delta_src=InitialPoseSource(env._delta_src,default_positions,env._q_soft_limits,
                                                cell['initial_jitter_rad'],cell['initial_seed'])
            previous_delay=getattr(env,'_evaluation_actuator_delay',None)
            if previous_delay is not None:
                previous_delay.close()
            env._evaluation_actuator_delay=TargetDelayAdapter(env,cell['actuator_delay_steps'])
            driver = (wrap_clutch(base, True, .5, .2, env.num_envs, env.device)
                      if method == 'system0' else PassthroughDriver())
            trace = EpisodeTrace(causal=args.causal_trace, kinematic=args.kinematic_trace, spheres=args.sphere_trace)
            print(f"[research] cell {index + 1}/{len(protocol['design'])} {cell}", flush=True)
            acc = run_window(env, driver, steps, dict(SAFE_LINES), .08, .005,
                             log_every=args.log_every, observer=trace)
            meta = {**cell, 'dt': dt, 'cell_id': index + 1, 'env_yaml': args.env_yaml,
                    'checkpoint_sha256': protocol['checkpoint_sha256']}
            if cell['flow'] in ('uniform_random', 'held_random'):
                meta['hold_steps'] = args.hold_steps if cell['flow'] == 'held_random' else 1
                meta['hold_duration_s'] = meta['hold_steps'] * dt
            row = summarize_window(acc, meta, 3)
            episode_meta = (env._delta_src.coverage_metadata()
                            if hasattr(env._delta_src, 'coverage_metadata') else None)
            ep = trace.write(out / f'cell_{index + 1:03d}.npz', acc, meta, episode_meta)
            if method == 'raw' and any(x['intervention_step_rate'] > 0 for x in ep):
                raise ValueError('raw command/execute mismatch: ablation is not a true bypass')
            atomic_json(out / f'cell_{index + 1:03d}.json', {'window': row, 'episodes': ep})
            windows.append(row)
            all_episodes.extend(ep)
            protocol['completed_cells'] = index + 1
            atomic_json(out / 'progress.json', {'completed_cells': index + 1, 'expected_cells': len(protocol['design']),
                        'by_method': {m: aggregate([r for r in windows if r['method'] == m])
                                      for m in args.methods if any(r['method'] == m for r in windows)}})
            print(f"[research] completed {index + 1}: violations={row['violation_episodes']}/{env.num_envs}", flush=True)
        protocol['status'] = 'complete'
        atomic_json(out / 'episodes.json', all_episodes)
        atomic_json(out / 'summary.json', {'protocol': protocol, 'windows': windows,
                    'by_method': {m: aggregate([r for r in windows if r['method'] == m]) for m in args.methods}})
    except BaseException as error:
        protocol.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        protocol['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out / 'protocol.json', protocol)
        env._backstop = backstop
        delay=getattr(env,'_evaluation_actuator_delay',None)
        if delay is not None:
            delay.close()
        env.close()
        app.close()


if __name__ == '__main__':
    main()
