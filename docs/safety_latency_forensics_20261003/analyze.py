"""Independent self/table feasibility oracle and failed-pair event histories."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'safety_perturbation_20261003'
CLASS = ['cross', 'self_F', 'self_U', 'table']
ARM = ['F_L', 'F_R', 'U_L', 'U_R']
SLICES = [slice(0, 7), slice(7, 14), slice(14, 20), slice(20, 26)]


def constraints(d, cfg, t, env, robot, debit_override=None, include_cross=False):
    """Exact noncross subset; p was not recorded, so cross feasibility is unclaimed."""
    r = 'F' if robot == 0 else 'U'
    sl = slice(0, 14) if robot == 0 else slice(14, 26)
    G = -d[f'pre_row_J_{r}'][t, env].astype(float)
    cap = d['solver_cap'][t, env].astype(float)
    dm = d['pre_row_dmin'][t, env]
    deff = cap / (cfg['gamma'] * float(d['meta']['dt'])) + dm
    rel = (d['pre_row_valid'][t, env]
           & d['pre_row_arm_mask'][t, env, :, robot*2:robot*2+2].any(-1))
    if not include_cross: rel &= d['pre_row_cls'][t,env] != 0
    if cfg['engage_dist'] is not None:
        rel &= (deff < cfg['engage_dist']) | (cap < 0)
    if cfg['exempt_structural_rows']:
        structural = d['pre_struct_exempt'][t, env].copy()
        if cfg['retain_conditional_rows']:
            structural &= ~d['pre_row_exempt'][t, env]
        drop = structural & (cap >= 0)
        if cfg['struct_engage_dist'] is not None:
            drop &= d['pre_row_d'][t, env] >= cfg['struct_engage_dist']
        rel &= ~drop
    box = cfg['vmax'] * float(d['meta']['dt'])
    target = d['pre_target'][t, env, sl]
    q = d['pre_q'][t, env, sl]
    limits = d['joint_soft_limits'][env, sl]
    lo = np.maximum(-box, limits[:, 0] - target)
    hi = np.minimum(box, limits[:, 1] - target)
    debit = G @ np.clip(target - q, -box, box)
    for past in d['pre_pending_targets'][t, env, :, sl]:
        debit = np.maximum(debit, G @ np.clip(past - q, -box, box))
    aware = cfg['backlog_aware'] if debit_override is None else debit_override
    h0 = cap.copy()
    if include_cross:
        priority=float(d['solver_priority_p'][t,env])
        budget=(1+priority)*.5 if robot==0 else (1-priority)*.5
        cross=d['pre_row_cls'][t,env]==0
        h0=np.where(cross & (cap>=0),budget*cap,cap)
    if aware: h0=h0-debit
    authority = np.where(G >= 0, G*lo, G*hi).sum(-1)
    h = np.maximum(h0, authority * .9)
    c = np.clip(d['cmd'][t, env, sl], -box, box)
    aG, ah = [], []
    off = 0
    for i in range(robot*2, robot*2+2):
        width = SLICES[i].stop - SLICES[i].start
        norm = np.linalg.norm(c[off:off+width])
        alpha = d['alpha'][t, env, i]
        if norm > 1e-9 and alpha < 1 - 1e-6:
            row = np.zeros(c.shape)
            row[off:off+width] = c[off:off+width] / norm
            aG.append(row); ah.append(alpha * norm)
        off += width
    aG = np.array(aG).reshape(-1, len(c))
    ah = np.array(ah)
    return G, h, rel, lo, hi, aG, ah, debit, h0


def feasibility(d, cfg, t, env, robot):
    G, h, rel, lo, hi, aG, ah, debit, h0 = constraints(d, cfg, t, env, robot)
    g = G[rel]; b = h[rel]
    nd = len(lo)
    # Safety-row common slack, alpha and position/speed box remain hard.
    aug = np.vstack([np.column_stack([g, -np.ones(len(g))]),
                     np.column_stack([aG, np.zeros(len(aG))])])
    rhs = np.concatenate([b, ah])
    result = linprog(np.r_[np.zeros(nd), 1.], A_ub=aug, b_ub=rhs,
                     bounds=list(zip(lo, hi)) + [(0, None)], method='highs')
    assert result.success, result.message
    u = d['exec'][t, env, :14] if robot == 0 else d['exec'][t, env, 14:]
    actual_resid = np.maximum(g @ u - b, 0).max(initial=0)
    return dict(step=t, slack_mm=float(result.x[-1]*1000),
                residual_mm=float(actual_resid*1000),
                full_solver_residual_mm=float(d['solver_residual'][t, env, robot]*1000),
                safety_rows=int(rel.sum()),
                authority_clamped_rows=int((rel & (h > h0 + 1e-8)).sum()),
                bypass=bool(d['pre_bypass_arm'][t, env, robot*2:robot*2+2].any()))


def load(path):
    n = np.load(path)
    d = {k:n[k] for k in n.files}
    d['meta'] = json.loads(str(n['meta_json']))
    d['km'] = json.loads(str(n['kinematic_meta_json']))
    return d


def describe(pid, km):
    i, j = km['pair_sphere_idx'][pid]
    a = km['sphere_names'][i]
    b = km['sphere_names'][j] if pid < km['robot_pair_count'] else f'table[{j}]'
    return [a, b]


def failed_pair(d, t, env, channel):
    km = d['km']; count = km['robot_pair_count']
    if channel == 3:
        values = d['full_table_distance'][t, env].copy()
        values[d['full_table_exempt'][t, env]] = np.inf
        col = int(np.argmin(values))
        return count + col, float(values[col])
    pairs = np.array(km['pair_sphere_idx'][:count]); arms = np.array(km['sphere_arm_id'])
    left = arms[pairs[:,0]] // 2; right = arms[pairs[:,1]] // 2
    take = left != right if channel == 0 else ((left == right) & (left == channel - 1))
    idx = np.flatnonzero(take)
    centers = d['sphere_centers'][t, env]; radii = np.array(km['sphere_radii_m'])
    i, j = pairs[idx,0], pairs[idx,1]
    values = np.linalg.norm(centers[i] - centers[j], axis=-1) - radii[i] - radii[j]
    col = int(np.argmin(values))
    return int(idx[col]), float(values[col])


def pair_history(d, pid, env, stop):
    frames = []
    for t in range(max(0, stop-35), min(stop+8, len(d['q']))):
        col = np.flatnonzero((d['pre_row_id'][t, env] == pid) & d['pre_row_valid'][t, env])
        if not col.size:
            frames.append(dict(step=t, selected=False)); continue
        k = int(col[0])
        frames.append(dict(step=t, selected=True,
            distance_mm=float(d['pre_row_d'][t,env,k]*1000),
            dmin_mm=float(d['pre_row_dmin'][t,env,k]*1000),
            exempt=bool(d['pre_row_exempt'][t,env,k]),
            cap_mm=float(d['solver_cap'][t,env,k]*1000),
            arm_rate_mm_s=float(d['pre_row_rate'][t,env,k]*1000),
            body_rate_mm_s=float(d['pre_row_body_rate'][t,env,k]*1000),
            full_com_rate_mm_s=float(d['pre_row_com_rate'][t,env,k]*1000),
            current_backlog_mm=float(d['pre_row_backlog'][t,env,k]*1000),
            issued_backlog_mm=float(d['issued_row_backlog'][t,env,k]*1000),
            delivered_backlog_mm=float(d['delivered_row_backlog'][t,env,k]*1000),
            projected_delta_mm=float(d['projected_margin_delta'][t,env,k]*1000)))
    return frames


reports = []
for run, original in [('delayed', OLD/'pending_final_pilot/cell_002.npz'),
                       ('nominal_table', OLD/'pending_nominal_l1/cell_001.npz')]:
    p = json.loads((OUT/run/'protocol.json').read_text())
    assert p['status']=='complete' and p['completed_cells']==1
    d = load(OUT/run/'cell_001.npz'); a=np.load(original)
    equivalence = {k:dict(exact=bool(np.array_equal(a[k],d[k])),
                           max_abs_error=float(np.max(np.abs(a[k]-d[k]))))
                   for k in ['q_initial','q','cmd','exec','official_margins']}
    assert equivalence['q_initial']['exact']
    if run == 'nominal_table': assert all(x['exact'] for x in equivalence.values())
    else: assert equivalence['cmd']['exact']
    cfg=p['effective_backstop']
    bad=d['official_margins']<0
    events=[]
    for env,channel in np.argwhere(bad.any(0)):
        env=int(env);channel=int(channel); first=int(np.flatnonzero(bad[:,env,channel])[0])
        pid,margin=failed_pair(d,first,env,channel)
        assert abs(margin-d['official_margins'][first,env,channel]) < 2e-5
        robot=0 if channel==1 else 1 if channel==2 else (int(d['km']['sphere_arm_id'][d['km']['pair_sphere_idx'][pid][0]])//2)
        fs=[feasibility(d,cfg,t,env,robot) for t in range(max(0,first-30),min(first+3,len(d['q'])))]
        events.append(dict(env=env,channel=CLASS[channel],first_step=first,
                           first_time_s=(first+1)*d['meta']['dt'],pair_id=pid,
                           pair=describe(pid,d['km']),first_margin_mm=margin*1000,
                           min_channel_mm=float(d['official_margins'][:,env,channel].min()*1000),
                           history=pair_history(d,pid,env,first),feasibility=fs))
    active=d['pre_row_valid']
    def q99(key): return float(np.quantile(np.abs(d[key]-d['pre_row_body_rate'])[active],.99)*1000)
    reports.append(dict(run=run,equivalence=equivalence,
        replay_type='exact original trajectory' if all(x['exact'] for x in equivalence.values()) else 'same initial pose and commands; changed physical trajectory',
        violations=int(bad.any((0,2)).sum()),bad_envs=np.flatnonzero(bad.any((0,2))).tolist(),
        runtime_backlog_aware=cfg['backlog_aware'],events=events,
        rate_errors_p99_mm_s=dict(arm_body=q99('pre_row_rate'),full_com_body=q99('pre_row_com_rate'))))
result=dict(schema='safeduo.latency_forensics.v1',runs=reports,
    safety_policy_modified=False,count_as_new_independent_windows=False,
    oracle_scope='Exact self/table active-row subset with alpha and target/speed box; p not captured, cross constraints omitted. Positive minimum slack proves the full set infeasible; zero slack does not prove full-set feasibility or dynamics safety.')
(OUT/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
for r in reports:
    print(r['run'],r['replay_type'],'violations',r['violations'],'rate_error',r['rate_errors_p99_mm_s'])
    for e in r['events']:
        fs=e['feasibility']
        print('event',e['env'],e['channel'],e['first_step'],e['pair'],
              'LP_max_slack_mm',round(max(x['slack_mm'] for x in fs),6),
              'max_subset_residual_mm',round(max(x['residual_mm'] for x in fs),6))
