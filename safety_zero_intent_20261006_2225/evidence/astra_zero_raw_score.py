"""Independent zero-inclusive numeric auditor; NumPy only, fresh closed raw inputs.

Own previous-cycle scoring primitives are reused with explicit three-mode schema
changes. No parent analysis/model imports, GPU calls or old trajectory reads.
"""
import argparse,datetime,hashlib,io,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from astra_zero_score_core import (CLASSES,MODES,aggregate,evaluation_classes,
    fingerprint,native_flags,paired_comparison,reduce_geometry,require,score_windows,unpack_exempt)
H=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/H.name
STEPS,ENVS,JOINTS,ROWS=960,64,26,9021
TERMINAL=('complete','complete_with_failures','failed')
COMPARISONS=(('joint_reference','zero_inclusive'),('joint_reference','tight_reference'),('tight_reference','zero_inclusive'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_owned(name,value):
    require(Path(name).name==name and name.startswith(('astra_zero_','ASTRA_ZERO_')),'outside owned output')
    dest=H/name;temporary=H/('astra_zero_tmp_'+str(os.getpid())+'_'+name)
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(dest)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

class Ledger:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        digest = sha(path)
        self._remember(path, digest, expected)
        return digest

    def _remember(self, path, digest, expected):
        key = str(path)
        if key in self.hashes:
            require(self.hashes[key] == digest, f'input changed on reread: {path}')
        self.hashes[key] = digest
        if expected is not None:
            require(digest == expected, f'input hash mismatch: {path}')

    def bytes(self, path, expected=None):
        path = Path(path)
        payload = path.read_bytes()
        self._remember(path, hashlib.sha256(payload).hexdigest(), expected)
        return payload

    def json(self, path, expected=None):
        return json.loads(self.bytes(path, expected))

    def npz(self, path, keys=None, expected=None):
        payload = self.bytes(path, expected)
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            return {key: z[key] for key in (z.files if keys is None else keys)}

    def recheck(self):
        changed = []
        for name, expected in self.hashes.items():
            try:
                if sha(name) != expected:
                    changed.append(name)
            except OSError:
                changed.append(name)
        return changed

def float_array(a, shape, name, allow_positive_inf=False):
    require(a.shape == shape and a.dtype == np.float32, f'{name}: shape/dtype mismatch')
    valid = ~np.isnan(a) & ~np.isneginf(a) if allow_positive_inf else np.isfinite(a)
    require(valid.all(), f'{name}: nonfinite data')

def exact(a, b, label):
    require(np.array_equal(a, b), label)

def relative_file(root, name):
    path = root/name
    require(path.resolve().is_relative_to(root.resolve()), 'receipt escaped cell directory')
    return path

def ids_mask(ids, shape):
    require(ids.dtype == np.int32 and ids.shape == shape, 'selected IDs shape/dtype')
    require(((ids >= -1) & (ids < ROWS)).all(), 'selected ID range')
    flat = ids.reshape(-1, ROWS)
    valid = flat >= 0
    r, c = np.nonzero(valid)
    dense = np.zeros_like(flat, dtype=bool)
    dense[r, flat[r, c]] = True
    exact(dense.sum(axis=1), valid.sum(axis=1), 'duplicate selected IDs')
    return dense.reshape(shape)

def verify_fifo(data):
    pending = data['pre_pending_actuator_targets']
    initial_target = pending[0, -1]
    initial_six = np.broadcast_to(initial_target, (6, ENVS, JOINTS))
    exact(pending[0], initial_six, 'initial FIFO must contain six initial targets')
    stream = np.concatenate((initial_six, data['controller_target']))
    exact(data['actuator_target'], stream[:STEPS], 'actual applied FIFO sequence')
    for lag in range(6):
        exact(pending[:, lag], stream[lag:lag+STEPS], f'pending FIFO slot {lag}')
    exact(data['pre_pending_project_history'], pending, 'project history differs from actuator FIFO')
    before_target = np.concatenate((initial_target[None], data['controller_target'][:-1]))
    before_q = np.concatenate((data['q_initial'][None], data['q'][:-1]))
    exact(data['pre_target_debt'], before_target-before_q, 'target debt is not target minus q')
    exact(data['effective_target_delta'], data['controller_target']-before_target, 'effective increment binding')
    limits = data['joint_soft_limits']
    integrated = np.clip(before_target+data['exec'], limits[..., 0], limits[..., 1])
    exact(integrated, data['controller_target'], 'native float32 actual target integration')
    return initial_target


def audit_reference_trace(data, diagnostic, mode, dt):
    """Independent native all-frame bounds/prelimit binding; no solver replay."""
    require(mode in MODES, 'unregistered reference mode')
    names=('bounds_lower','bounds_upper','project_input_cmd','external_raw_cmd','outer_returned_cmd')
    shape=(STEPS,ENVS,JOINTS)
    for name in names: float_array(diagnostic[name],shape,name)
    q=np.concatenate((data['q_initial'][None],data['q'][:-1]))
    target=np.concatenate((data['pre_pending_actuator_targets'][0,-1][None],data['controller_target'][:-1]))
    limits=data['joint_soft_limits']
    box=np.float32(1.5*dt)
    original_lo=np.maximum(limits[None,...,0]-target,-box)
    original_hi=np.minimum(limits[None,...,1]-target,box)
    require((original_lo<=original_hi).all(),'empty original reference interval')
    gap=np.float32(.050 if mode=='joint_reference' else .010)
    lo=np.minimum(np.maximum(q-gap-target,original_lo),original_hi)
    hi=np.minimum(np.maximum(q+gap-target,original_lo),original_hi)
    if mode=='zero_inclusive':
        require(((original_lo<=0)&(original_hi>=0)).all(),'zero outside original bounds invalidates candidate window')
        lo=np.minimum(lo,np.float32(0));hi=np.maximum(hi,np.float32(0))
    exact(lo,diagnostic['bounds_lower'],'all-frame lower reference bounds')
    exact(hi,diagnostic['bounds_upper'],'all-frame upper reference bounds')
    require(((original_lo<=lo)&(lo<=hi)&(hi<=original_hi)).all(),'reference interval escaped original bounds')
    limited=np.minimum(np.maximum(data['cmd'],lo),hi)
    exact(limited,diagnostic['project_input_cmd'],'all-frame prelimit input to actual projection')
    exact(data['cmd'],diagnostic['external_raw_cmd'],'all-frame external command binding')
    exact(data['exec'],diagnostic['outer_returned_cmd'],'all-frame actual projection output binding')
    raw_zero=data['cmd']==np.float32(0)
    injected=raw_zero&(limited!=np.float32(0))
    if mode=='zero_inclusive': require(not injected.any(),'zero intent changed by candidate prelimit')
    # Numerical scoring uses zero epsilon. Projection residuals are reported,
    # not reclassified through its separate 1e-6 source guard tolerance.
    residual=np.maximum(np.maximum(lo-data['exec'],data['exec']-hi),np.float32(0))
    per_env=[]
    for e in range(ENVS):
        per_env.append(dict(
            zero_excluded_bound_components=int(((lo[:,e]>0)|(hi[:,e]<0)).sum()),
            prelimit_injected_nonzero_on_zero_components=int(injected[:,e].sum()),
            prelimit_injected_env_steps=int(injected[:,e].any(-1).sum()),
            prefix_prelimit_injected_env_steps=int(injected[:60,e].any(-1).sum()),
            prefix_nonzero_projector_output_env_steps=int((data['exec'][:60,e]!=0).any(-1).sum()),
            prefix_nonzero_effective_target_delta_env_steps=int((data['effective_target_delta'][:60,e]!=0).any(-1).sum()),
            positive_native_bound_residual_env_steps=int((residual[:,e]>0).any(-1).sum()),
            maximum_native_bound_residual_rad=float(residual[:,e].max())))
    return dict(frames=STEPS,components=STEPS*ENVS*JOINTS,mode=mode,
        native_dtype='float32',bound_and_prelimit_binding=True,
        zero_execution_required=False,all_J_or_LP_recomputed=False,
        scope='Actual saved pre-state and actual issued target; six FIFO entries remain irrevocable. No same-step causal attribution.',
        windows=per_env)

def selected_failures(ledger, root, windows, data):
    receipt = ledger.json(root/'first_failure_receipts.json')
    expected = {env: w['strict']['first_step'] for env, w in enumerate(windows) if w['strict']['failed']}
    observed, candidates = {}, []
    for entry in receipt['receipts']:
        step = entry['step']
        require(isinstance(step, int) and 0 <= step < STEPS, 'bad first-failure step')
        for index, env in enumerate(entry['env_ids']):
            require(env not in observed and 0 <= env < ENVS, 'duplicate/invalid first-failure env')
            observed[env] = step
            candidates.append((step, env, index, entry))
    require(observed == expected and receipt['envs_with_failure'] == len(expected), 'first-failure receipt differs from independent native score')
    selected = []
    for step, env, index, entry in sorted(candidates, key=lambda x: (x[0], x[1]))[:2]:
        file = relative_file(root, entry['path'])
        raw = ledger.npz(file, expected=entry['sha256'])
        require(int(raw['step']) == step and int(raw['env_ids'][index]) == env, 'first-failure snapshot identity')
        s = {k: v[index] for k, v in raw.items() if k not in ('step', 'env_ids')}
        for k, v in s.items():
            if v.dtype.kind in 'fc':
                require(np.isfinite(v).all(), f'nonfinite first-failure {k}')
        pre_q = data['q_initial'][env] if step == 0 else data['q'][step-1, env]
        pre_target = data['pre_pending_actuator_targets'][0, -1, env] if step == 0 else data['controller_target'][step-1, env]
        for label, a, b in (
            ('pre q', s['pre_q'], pre_q), ('post q', s['post_q'], data['q'][step, env]),
            ('pre qd', s['snapshot_qd'], data['pre_qd_compact'][step, env]),
            ('pre target', s['pre_issued_target'], pre_target),
            ('pending', s['pre_pending_actuator_targets'], data['pre_pending_actuator_targets'][step, :, env]),
            ('project output', s['actual_project_return'], data['exec'][step, env]),
            ('integrated delta', s['returned_cmd'], data['effective_target_delta'][step, env])):
            exact(a, b, f'first failure {label} binding')
        if step < STEPS-1:
            exact(s['post_qd'], data['pre_qd_compact'][step+1, env], 'first failure next pre qd binding')
        selected.append(dict(step=step, env=env, path=str(file), raw=s))
    return selected

def score_cell(ledger, plan, job, campaign_record):
    root = Path(plan['output_root'])/job['id']
    require(campaign_record['status'] == 'complete' and campaign_record['exit_code'] == 0, 'condition did not complete successfully')
    require(campaign_record['id'] == job['id'] and campaign_record['argv'] == job['argv']+['--out',str(root)], 'canonical actual argv differs from frozen command')
    proto = ledger.json(root/'protocol.json', campaign_record['protocol_sha256'])
    require(proto['status'] == 'complete' and proto['completed_cells'] == 1 and len(proto['design']) == 1, 'protocol incomplete')
    require(proto['steps'] == STEPS and proto['dt'] == .016666, 'timebase mismatch')
    require(proto['source_sha256'] == plan['source_sha256'] and proto['checkpoint_sha256'] == plan['checkpoint_sha256'], 'source/actor protocol mismatch')
    for k, v in job['expected'].items():
        require(proto['design'][0][k] == v, f'condition differs: {k}')
    for k, v in job['expected_args'].items():
        require(proto['args'][k] == v, f'protocol argument differs: {k}')
    require(proto['args']['actuator_delay_steps'] == [6], 'actuator delay changed')
    mode = job['env']['SAFEDUO_JOINT_MODE']
    metadata = ledger.json(root/'guard_metadata.json')
    require(metadata['mode'] == mode and metadata['capacity'] == ROWS and metadata['strict_fifo_steps'] == 6, 'guard registration differs')
    require(metadata['reference_gap_rad'] == {'joint_reference':.050,'tight_reference':.010,'zero_inclusive':.010}[mode] and metadata['original_actor_rows'] == 32 and not metadata['joint_repair'] and not metadata['queue_preemption'], 'guard control contract differs')
    require(metadata['reference_zero_inclusive'] is (mode=='zero_inclusive'),'guard zero-inclusive flag differs')
    for path, digest in metadata['sources'].items():
        require(plan['research_source_sha256'].get(path) == digest, f'guard source not registered: {path}')
    keys = ('official_margins', 'official_deep', 'q', 'q_initial', 'cmd', 'external_unscaled_cmd',
            'exec', 'controller_target', 'actuator_target', 'pre_qd_compact', 'pre_target_debt',
            'pre_pending_actuator_targets', 'pre_pending_project_history', 'effective_target_delta',
            'joint_soft_limits', 'initial_violation', 'critical_selected_count', 'meta_json')
    data = ledger.npz(root/'cell_001.npz', keys)
    cell_meta = json.loads(str(data['meta_json']))
    for k, v in proto['design'][0].items():
        require(cell_meta[k] == v, f'raw cell condition differs: {k}')
    require(cell_meta['dt'] == proto['dt'] and cell_meta['cell_id'] == 1
            and cell_meta['checkpoint_sha256'] == proto['checkpoint_sha256']
            and cell_meta['env_yaml'] == proto['args']['env_yaml'], 'raw cell metadata differs')
    for key in ('q', 'cmd', 'external_unscaled_cmd', 'exec', 'controller_target', 'actuator_target',
                'pre_qd_compact', 'pre_target_debt', 'effective_target_delta'):
        float_array(data[key], (STEPS, ENVS, JOINTS), key)
    float_array(data['q_initial'], (ENVS, JOINTS), 'q_initial')
    float_array(data['joint_soft_limits'], (ENVS, JOINTS, 2), 'limits')
    float_array(data['official_margins'], (STEPS, ENVS, 4), 'official_margins', True)
    for key in ('pre_pending_actuator_targets', 'pre_pending_project_history'):
        float_array(data[key], (STEPS, 6, ENVS, JOINTS), key)
    strict, deep = native_flags(data['official_margins'])
    require(data['official_deep'].dtype == np.bool_ and data['official_deep'].shape == deep.shape,
            'producer deep flag shape/dtype')
    require(data['initial_violation'].shape == (ENVS,) and data['initial_violation'].dtype == np.bool_,
            'initial violation shape/dtype')
    require(data['critical_selected_count'].shape == (STEPS, ENVS)
            and data['critical_selected_count'].dtype.kind in 'iu', 'selected count shape/dtype')
    require((data['joint_soft_limits'][..., 0] <= data['joint_soft_limits'][..., 1]).all(), 'inverted soft limits')
    exact(data['official_deep'], deep, 'producer deep flags vs independently recomputed native threshold')
    initial_target = verify_fifo(data)
    bank_path = Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ'])
    bank_hash = ledger.bind(bank_path, plan['research_source_sha256'][str(bank_path)])
    bank = ledger.npz(bank_path, ['accepted_q', 'risk_pair_index', 'joint_soft_limits'], bank_hash)
    assignment = ledger.npz(root/'bank_assignment.npz', ['risk_pair_index'])
    require(bank['risk_pair_index'].shape == (ENVS,) and bank['risk_pair_index'].dtype.kind in 'iu', 'bank label shape/dtype')
    exact(assignment['risk_pair_index'], bank['risk_pair_index'], 'actual bank label assignment')
    recipe = ledger.npz(root/'input_recipe.npz', ['q_initial', 'sampled_initial', 'tape', 'joint_soft_limits', 'initial_violation'])
    float_array(recipe['tape'], (STEPS+2, ENVS, JOINTS), 'registered run tape plus two unexecuted frames')
    float_array(recipe['q_initial'], (ENVS, JOINTS), 'recipe actual initial state')
    float_array(recipe['sampled_initial'], (ENVS, JOINTS), 'recipe sampled initial state')
    float_array(bank['accepted_q'], (ENVS, JOINTS), 'bank sampled initial state')
    exact(recipe['tape'][:60], np.zeros((60, ENVS, JOINTS), dtype=np.float32), 'registered zero prefix')
    exact(recipe['q_initial'], data['q_initial'], 'input recipe actual q0 binding')
    exact(recipe['sampled_initial'], bank['accepted_q'], 'sampled q0 vs registered bank')
    exact(recipe['joint_soft_limits'], data['joint_soft_limits'], 'recipe soft limits binding')
    exact(bank['joint_soft_limits'], data['joint_soft_limits'], 'bank soft limits binding')
    exact(recipe['initial_violation'], data['initial_violation'], 'initial violation binding')
    exact(data['cmd'], recipe['tape'][:STEPS], 'executed external raw command differs from tape')
    exact(data['external_unscaled_cmd'], recipe['tape'][:STEPS], 'unscaled command/tape binding')
    diagnostic=ledger.npz(root/'project_diagnostics.npz',
        ('bounds_lower','bounds_upper','project_input_cmd','external_raw_cmd','outer_returned_cmd'))
    reference_trace=audit_reference_trace(data,diagnostic,mode,proto['dt'])
    metrics = score_windows(data['official_margins'])
    prefix_metrics = score_windows(data['official_margins'][:60])
    selected = selected_failures(ledger, root, metrics, data)
    geometry = chunk_audit(ledger, root, data, selected, mode)
    qseq = np.concatenate((data['q_initial'][None], data['q']), axis=0).astype(np.float64)
    path = np.abs(np.diff(qseq, axis=0)).sum(axis=(0, 2))
    ranges = np.ptp(qseq, axis=0).sum(axis=1)
    windows = []
    for env in range(ENVS):
        windows.append(dict(status='VERIFIED', env=env, metrics=metrics[env], prefix_metrics=prefix_metrics[env],
            reference_contract=reference_trace['windows'][env],
            q0_sha256=fingerprint(data['q_initial'][env]), qd0_sha256=fingerprint(data['pre_qd_compact'][0, env]),
            initial_target_sha256=fingerprint(initial_target[env]), tape_sha256=fingerprint(recipe['tape'][:STEPS, env]),
            full_tape_sha256=fingerprint(recipe['tape'][:, env]),
            bank_sha256=bank_hash, risk_pair_index=int(bank['risk_pair_index'][env]),
            first6_post_q_sha256=fingerprint(data['q'][:6,env]),
            first6_pre_qd_sha256=fingerprint(data['pre_qd_compact'][:6,env]),
            first6_post_qd_sha256=fingerprint(data['pre_qd_compact'][1:7,env]),
            first6_applied_sha256=fingerprint(data['actuator_target'][:6,env]),
            initial_violation_recorded=bool(data['initial_violation'][env]),
            actual_joint_L1_path_rad=float(path[env]), actual_joint_range_sum_rad=float(ranges[env])))
    return dict(id=job['id'], mode=mode, root=str(root), status='VERIFIED',
                windows=windows, counts=aggregate(windows), geometry=geometry,
                reference_contract={k:v for k,v in reference_trace.items() if k!='windows'},
                raw_cell_sha256=ledger.hashes[str(root/'cell_001.npz')])


def chunk_audit(ledger, root, data, selected, mode):
    identity = ledger.json(root/'full_row_identity.json')
    classes = evaluation_classes(identity)
    require(len(classes) == ROWS and identity['rows'] == ROWS, 'full geometry row count')
    receipt = ledger.json(root/'forecast_receipts.json')
    require(receipt['steps'] == STEPS and receipt['rows'] == ROWS and receipt['envs'] == ENVS, 'incomplete forecast receipt')
    require(len(receipt['chunks']) == 30, 'expected 30 complete forecast chunks')
    total_pre = np.empty((STEPS, ENVS, 4), dtype=np.float32)
    end, motion_rows, motion_env_steps, row_instances = 0, 0, 0, 0
    keys = ('measured_d', 'exempt', 'dmin', 'forecast', 'target_forecast',
            'reserve_forecast', 'selected_ids', 'baseline_ids', 'reserve_margin', 'reserve_gap')
    for entry in receipt['chunks']:
        start, stop = entry['start'], entry['stop']
        require(start == end and stop-start == 32 and stop <= STEPS, 'chunk gap/overlap/length')
        path = relative_file(root, entry['path'])
        z = ledger.npz(path, keys, entry['sha256'])
        shape = (stop-start, ENVS, ROWS)
        for k in ('measured_d','dmin','forecast','target_forecast','reserve_forecast'):
            float_array(z[k], shape, k)
        exempt = unpack_exempt(z['exempt'], ROWS)
        require(exempt.shape == shape, 'exemption shape')
        minimum = reduce_geometry(z['measured_d'], exempt, classes)
        total_pre[start:stop] = minimum
        skip = 1 if start == 0 else 0
        exact(minimum[skip:], data['official_margins'][start+skip-1:stop-1], 'full pre geometry vs previous native post margins')
        exact(z['forecast'], z['target_forecast'], 'all modes retain identical target-based admission forecast')
        for k in ('forecast','target_forecast','reserve_forecast'):
            require((z[k] <= z['measured_d']).all(), 'forecast omitted current geometry')
        for k in ('reserve_margin','reserve_gap'):
            float_array(z[k], (stop-start,ENVS), k)
        margin = np.where(exempt,np.float32(np.inf),z['reserve_forecast']-z['dmin']).min(axis=-1)
        require(np.isfinite(margin).all(), 'no eligible reserve row or invalid margin')
        exact(margin,z['reserve_margin'],'stored reserve minimum/exemption binding')
        if mode=='joint_reference':gap=np.full_like(margin,np.float32(.050))
        else:
            require(mode in ('tight_reference','zero_inclusive'),'unknown mode')
            gap=np.full_like(margin,np.float32(.010))
        exact(gap,z['reserve_gap'],'native stored chosen gap/mode binding')
        chosen = ids_mask(z['selected_ids'], shape)
        measured_base = ids_mask(z['baseline_ids'], shape)
        boundary = z['dmin'] + np.float32(.010)
        require((measured_base | ~(z['measured_d'] <= boundary)).all(), 'measured baseline omitted critical row')
        exact(chosen, measured_base | (z['forecast'] <= boundary), 'actual row union dropped/swapped rows')
        target_mask = measured_base | (z['target_forecast'] <= boundary)
        require((chosen | ~target_mask).all(), 'old target-based row missing')
        additional = chosen & ~target_mask
        require(not additional.any(), 'zero experiment mode introduced non-target admission rows')
        motion_rows += int(additional.sum()); motion_env_steps += int(additional.any(axis=-1).sum())
        row_instances += int(chosen.sum())
        exact(chosen.sum(axis=-1), data['critical_selected_count'][start:stop], 'selected count dense binding')
        for case in selected:
            t, e, s = case['step'], case['env'], case['raw']
            if not start <= t < stop:
                continue
            offset = t-start
            valid = s['snapshot_valid'].astype(bool)
            ids = s['selected_ids'][:len(valid)]
            exact(s['selected_ids'], z['selected_ids'][offset, e], 'first-failure selected ID binding')
            exact(s['snapshot_d'][valid], z['measured_d'][offset, e, ids[valid]], 'first-failure pre geometry binding')
            post_min = reduce_geometry(s['post_full_d'], s['post_full_exempt'], classes)
            exact(post_min, data['official_margins'][t, e], 'first-failure full post geometry binding')
            bad = np.flatnonzero((s['post_full_d'] < np.float32(0)) & ~s['post_full_exempt'])
            require(len(bad) > 0, 'selected first failure has no native negative row')
            J = np.concatenate((s['snapshot_J_F'], s['snapshot_J_U']), axis=-1).astype(np.float64)
            dq = s['post_q'].astype(np.float64)-s['pre_q'].astype(np.float64)
            effects = []
            for pid in bad:
                positions = np.flatnonzero(valid & (ids == pid))
                item = dict(pair_id=int(pid), class_name=CLASSES[classes[pid]],
                    pre_distance_m=float(z['measured_d'][offset, e, pid]),
                    post_distance_m=float(s['post_full_d'][pid]), selected=bool(len(positions)))
                if len(positions):
                    j = J[int(positions[0])]
                    item.update(J_actual_q_motion_m=float(j@dq),
                        J_integrated_target_increment_m=float(j@s['returned_cmd'].astype(np.float64)),
                        J_applied_target_error_m=float(j@(s['pre_pending_actuator_targets'][0].astype(np.float64)-s['pre_q'].astype(np.float64))))
                effects.append(item)
            case['binding'] = dict(status='PASS_SELECTED_FIRST_FAILURE_BINDINGS', step=t, env=e,
                path=case['path'], rows=effects, post_qd_dense_check=t < STEPS-1,
                no_unique_physical_cause_claim=True)
        end = stop
        del z, chosen, measured_base, target_mask, additional
        if stop % 256 == 0 or stop == STEPS:
            print(json.dumps(dict(event='INDEPENDENT_CHUNKS_READ',condition=root.name,frames=stop)),flush=True)
    require(end == STEPS and all('binding' in c for c in selected), 'incomplete chunk/snapshot coverage')
    return dict(full_pre_frames=STEPS, full_rows=ROWS, post_frames_crosschecked=STEPS-1,
        final_post_full_row_archive_available=False, final_post_source='native cell official_margins[959]',
        full_geometry_pre_strict_env_steps=int((total_pre < np.float32(0)).any(axis=-1).sum()),
        full_geometry_pre_deep_env_steps=int((total_pre < np.float32(-.005)).any(axis=-1).sum()),
        stored_forecast_mode_minima_and_unions_verified=True,
        non_target_added_row_instances_at_own_state=motion_rows,
        non_target_added_env_steps_at_own_state=motion_env_steps, selected_row_instances=row_instances,
        full_J_archive_recomputed=False, reserve_archive_minimum_and_gap_verified=True,
        first_failure_selection='first two distinct failing envs by (step,env) per condition, if present',
        selected_first_failures=[c['binding'] for c in selected])


def registered_inputs(ledger):
    reg=ledger.json(H/'NUMERIC_REGISTRATION.json')
    require(reg['status']=='FROZEN_BEFORE_ALL_NEW_POLICY_OUTCOMES','numeric registration not frozen')
    require(reg['method_windows']==576 and reg['paired_cases']==192,'wrong denominators')
    design=ledger.json(H/'RANDOM_EXPERIMENT_DESIGN.json')
    require(tuple(design['modes'])==MODES and design['frames_per_window']==STEPS,'design modes/frames')
    require(tuple(map(tuple,design['comparisons']))==COMPARISONS,'comparison registration')
    review=ledger.json(H/'ASTRA_ZERO_SOURCE_REVIEW.json')
    require(review['status']=='PASS_SCOPED_INDEPENDENT_CPU_ZERO_SOURCE_REVIEW','source review incomplete')
    plans=[]
    for block in range(3):
        path=H/'plans'/f'holdout_{block}_plan.json'
        plan=ledger.json(path,reg['plans'][str(path)])
        require(plan['sources_frozen'] and not plan['production_promoted'],'unfrozen/promoted plan')
        require(Path(plan['output_root'])==RAW/f'holdout_{block}','wrong fresh raw root')
        require(len(plan['jobs'])==3 and len({j['id'] for j in plan['jobs']})==3,'job inventory')
        require([j['env']['SAFEDUO_JOINT_MODE'] for j in plan['jobs']]==list(MODES),'registered method order')
        row=design['rows'][block];require(row['block']==block,'block order')
        for job in plan['jobs']:
            require(job['expected']['seed']==row['command_seed'],'command seed mismatch')
            require(Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ'])==RAW/'banks'/str(row['initial_seed'])/'bank.npz','bank mismatch')
        for path,digest in review['source_sha256'].items():
            require(plan['research_source_sha256'].get(path)==digest,'reviewed control source not frozen')
        plans.append(plan)
    require(len(reg['plans'])==3,'extra/missing plans')
    require(len({r['initial_seed'] for r in design['rows']})==3 and len({r['command_seed'] for r in design['rows']})==3,'duplicate registered seed blocks')
    return design,plans


def canonical_closed(ledger,plan):
    path=Path(plan['output_root'])/'campaign.json'
    if not path.exists():return None
    # Running metadata is observed, never added to the immutable input ledger.
    try:campaign=json.loads(path.read_text())
    except json.JSONDecodeError:return None
    if campaign.get('status') not in TERMINAL:return None
    campaign=ledger.json(path)
    require(campaign['status'] in TERMINAL and campaign['plan']==plan,'closed campaign plan changed')
    records={r['id']:r for r in campaign['jobs']}
    require(len(records)==len(campaign['jobs']),'duplicate closed campaign records')
    require(set(records).issubset({j['id'] for j in plan['jobs']}),'unregistered canonical child')
    return records


def bind_software(ledger):
    specification=ledger.json(H/'astra_zero_raw_plan.json')
    require(specification['status']=='REGISTERED_INDEPENDENT_TWO_WORKER_SCORER_BEFORE_RAW_OUTCOME_READS','scorer not frozen')
    require(specification['workers']==2 and specification['expected_windows']==576,'scorer contract changed')
    for name,digest in specification['source_sha256'].items():ledger.bind(H/name,digest)
    for name,digest in specification['registration_sha256'].items():ledger.bind(H/name,digest)
    return specification


def cell_worker(plan,job,record,block,index):
    started=time.monotonic();ledger=Ledger()
    try:
        require(record is not None,'canonical campaign closed without this child')
        result=score_cell(ledger,plan,job,record)
        changed=ledger.recheck()
        require(not changed,'consumed cell input changed: '+repr(changed))
    except Exception as error:
        result=dict(id=job['id'],mode=job['env']['SAFEDUO_JOINT_MODE'],
            root=str(Path(plan['output_root'])/job['id']),status='UNAVAILABLE_OR_INVALID',
            error=f'{type(error).__name__}: {error}',
            windows=[dict(status='UNAVAILABLE_OR_INVALID',env=e,metrics=None) for e in range(ENVS)])
    result.update(block=block,registered_index=index)
    checkpoint=dict(schema='astra.zero.cell_checkpoint.v1',utc=now(),result=result,
        input_sha256=ledger.hashes,cell_raw_before_after_pass=result['status']=='VERIFIED',seconds=time.monotonic()-started,
        source_sha256={n:sha(H/n) for n in ['astra_zero_raw_score.py','astra_zero_score_core.py']})
    # Distinct output files; no checkpoint is read as an input in this run.
    write_owned(f'astra_zero_raw_cell_{index:02d}.json',checkpoint)
    return checkpoint


def merge_checkpoint(ledger,checkpoint):
    for path,digest in checkpoint['input_sha256'].items():ledger._remember(path,digest,None)
    return checkpoint['result']


def assemble_report(design,cells,ledger,changed):
    require(len(cells)==9 and {c['registered_index'] for c in cells}==set(range(9)),'nine condition inventory')
    cases=[]
    for block,row in enumerate(design['rows']):
        by_mode={c['mode']:c for c in cells if c['block']==block}
        require(set(by_mode)==set(MODES),'missing/duplicate mode')
        for env in range(ENVS):
            cases.append(dict(case_id=f'b{block}:i{row["initial_seed"]}:c{row["command_seed"]}:e{env:02d}',
                block=block,initial_seed=row['initial_seed'],command_seed=row['command_seed'],env=env,
                windows={m:by_mode[m]['windows'][env] for m in MODES}))
    require(len(cases)==192 and sum(len(c['windows']) for c in cases)==576,'lost denominator')
    paired={a+'__vs__'+b:paired_comparison(cases,b,a) for a,b in COMPARISONS}
    counts={m:aggregate([c['windows'][m] for c in cases]) for m in MODES}
    all_valid=all(w['status']=='VERIFIED' for c in cases for w in c['windows'].values())
    all_pairs=all(v['verified_pairs']==192 for v in paired.values())
    prefixes={}
    for a,b in COMPARISONS:
        bad=[]
        for c in cases:
            wa,wb=c['windows'][a],c['windows'][b]
            if wa['status']!='VERIFIED' or wb['status']!='VERIFIED':
                bad.append(dict(case_id=c['case_id'],status='UNAVAILABLE'));continue
            keys=['first6_post_q_sha256','first6_pre_qd_sha256','first6_post_qd_sha256','first6_applied_sha256']
            mismatch=[k for k in keys if wa.get(k) is None or wa.get(k)!=wb.get(k)]
            if mismatch:bad.append(dict(case_id=c['case_id'],status='PREFIX_DIFFERENCE',fields=mismatch))
        prefixes[a+'__vs__'+b]=dict(checked_pairs=192-len(bad),differences_or_unavailable=bad,
            scope='Native dtype/shape/byte fingerprints. Does not assert full unarchived rigid-state equality.')
    return dict(schema='astra.zero.independent_numeric.v1',utc=now(),
        status='PASS_COMPLETE_INDEPENDENT_ZERO_NUMERIC' if all_valid and all_pairs and not changed else 'INCOMPLETE_OR_INVALID_ZERO_NUMERIC',
        expected_cases=192,expected_windows=576,actual_case_identities=len(cases),actual_window_records=576,
        modes=MODES,class_order=CLASSES,steps_per_window=STEPS,zero_prefix_included=True,
        duration_scope='Environment-step counts within 960 correlated frames; not IID samples or elapsed wall-clock duration.',
        gap_arithmetic='Fixed float32 .050 reference and .010 tight/zero-inclusive; no adaptive arithmetic or epsilon.',
        comparison_dtype='native float32 before JSON conversion',strict_boundary='<np.float32(0)',deep_boundary='<-np.float32(.005)',
        zero_prefix_totals={m:aggregate([dict(status=w['status'],metrics=w.get('prefix_metrics')) for c in cases for w in [c['windows'][m]]]) for m in MODES},
        dt_seconds=.016666,
        totals=counts,counts=counts,paired=paired,comparisons=paired,primary_comparison='joint_reference__vs__zero_inclusive',
        cases=cases,conditions=[{k:v for k,v in c.items() if k!='windows'} for c in sorted(cells,key=lambda c:c['registered_index'])],
        numeric_prefix=prefixes,input_sha256=ledger.hashes,raw_hash_before_after_pass=not changed,changed_inputs=changed,
        hash_scope='All consumed raw/source/plan inputs; per-cell pre/post hashes plus final whole-ledger reread. Unselected snapshots/unrelated raw files excluded.',
        completion_contract='Canonical closed campaign and complete zero-exit child protocol; outer orchestration exit is not a physics score.',
        geometry_scope='Full9021 pre rows bind959 preceding post frames; final post959 uses native class minima. First two failing environments per condition receive selected-J binding; no all-row J archive claim.',
        reserve_scope='Archived all-row reserve minima/exemptions and native gap formula verified; producer full-J checks remain source evidence, not independent full-J recomputation at every frame.',
        not_verified=['Isaac/GPU counterfactual replay','camera own-state/pixels','camera exact forward','parent fullLP diagnostics','parent results/report reconciliation'],
        parent_counts_used=False,parent_analysis_imported=False,policy_outcomes_used_for_tuning=False,
        physical_safety_certified=False,hardware_approved=False,production_promoted=False)


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')


def run():
    ledger=Ledger();spec=bind_software(ledger);design,plans=registered_inputs(ledger)
    source_bindings={}
    for plan in plans:
        source_bindings.update({str(Path(plan['cwd'])/p):d for p,d in plan['source_sha256'].items()})
        source_bindings.update(plan['research_source_sha256'])
        source_bindings[plan['checkpoint_path']]=plan['checkpoint_sha256']
    for path,digest in source_bindings.items():ledger.bind(path,digest)
    records_by_block={block:canonical_closed(ledger,plan) for block,plan in enumerate(plans)}
    require(all(v is not None for v in records_by_block.values()),'all canonical campaigns must be terminal; no watcher is available')
    require(available_memory()>=spec['minimum_mem_available_bytes'],'insufficient available memory; no worker launched')
    require(not (H/'ASTRA_ZERO_FINAL_SCORE.json').exists(),'final score already exists; preserve previous attempt')
    require(not list(H.glob('astra_zero_raw_cell_*.json')),'prior cell checkpoints exist; explicit recovery registration required')
    lock=H/'astra_zero_raw_writer.lock'
    with lock.open('x') as file:json.dump(dict(pid=os.getpid(),utc=now(),source_sha256=sha(__file__)),file)
    jobs=[(b,plan,j,b*3+i) for b,plan in enumerate(plans) for i,j in enumerate(plan['jobs'])]
    submitted=set();cells=[];pending={}
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='astra_zero_raw') as pool:
        while len(cells)<9:
            for block,plan,job,index in jobs:
                if len(pending)>=2:break
                if index in submitted or block not in records_by_block:continue
                if available_memory()<spec['minimum_mem_available_bytes']:break
                pending[pool.submit(cell_worker,plan,job,records_by_block[block].get(job['id']),block,index)]=index
                submitted.add(index)
            write_owned('astra_zero_raw_progress.json',dict(utc=now(),status='READING_CLOSED_CAMPAIGNS' if pending else 'WAITING_CANONICAL_CLOSURE',
                canonical_closed_blocks=sorted(records_by_block),completed_conditions=len(cells),total_conditions=9,
                running_conditions=sorted(pending.values()),workers_limit=2,
                conditions=[{k:c.get(k) for k in ['registered_index','block','id','status','error']} for c in cells]))
            if pending:
                done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    index=pending.pop(future);checkpoint=future.result();cell=merge_checkpoint(ledger,checkpoint)
                    for name,digest in checkpoint['source_sha256'].items():require(sha(H/name)==digest,'scoring source changed during cell')
                    ledger.bind(H/f'astra_zero_raw_cell_{index:02d}.json')
                    cells.append(cell);print(json.dumps(dict(event='INDEPENDENT_CELL_CLOSED',index=index,id=cell['id'],status=cell['status'],error=cell.get('error'))),flush=True)
            else:
                raise RuntimeError('available memory below registered gate; no indefinite wait or automatic retry')
    changed=ledger.recheck()
    report=assemble_report(design,cells,ledger,changed)
    write_owned('ASTRA_ZERO_FINAL_SCORE.json',report)
    write_owned('astra_zero_raw_execution.json',dict(status=report['status'],utc=now(),pid=os.getpid(),
        conditions=9,windows=576,workers=2,registered_order=[j['id'] for _,_,j,_ in jobs],
        raw_before_after_pass=report['raw_hash_before_after_pass'],score_sha256=sha(H/'ASTRA_ZERO_FINAL_SCORE.json'),
        independent_registration_sha256=sha(H/'astra_zero_raw_plan.json'),no_previous_checkpoint_reused=True))
    print(report['status'],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-plan',action='store_true')
    args=parser.parse_args()
    if args.check_plan:
        ledger=Ledger();design,plans=registered_inputs(ledger)
        print('PASS_REGISTERED_192_CASES_576_WINDOWS_NO_OUTCOME_READS',flush=True);return
    try:result=run()
    except BaseException as error:
        write_owned('astra_zero_raw_error.json',dict(status='FAILED_OR_INTERRUPTED_INDEPENDENT_AUDIT',utc=now(),error=f'{type(error).__name__}: {error}',pid=os.getpid()))
        raise
    if result['status'].startswith('INCOMPLETE'):raise SystemExit(1)


if __name__=='__main__':main()
