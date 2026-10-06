"""Dense full-row receipt, union, endpoint and FIFO audit. No caches."""
import hashlib,json
import numpy as np
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def forecast_audit(path, data):
    receipt = json.loads((path / 'forecast_receipts.json').read_text())
    mechanism = load(path / 'mechanism.npz')
    identity = json.loads((path / 'full_row_identity.json').read_text())
    assert receipt['steps'] == 960 and receipt['rows'] == 9021 and len(receipt['chunks']) == 30
    assert (mechanism['full_rows_checked'] == 9021).all() and (mechanism['guard_calls_before_step'] >= 1).all()
    cls = np.array(identity['class_id'])
    mode=json.loads((path/'guard_metadata.json').read_text())['mode']
    capacity=9021
    assert len(cls)==9021 and len(identity['pair_id'])==9021
    pair=np.array(identity['pair_sphere_idx']);arm=np.array(identity['sphere_arm_id'])
    assert pair.shape==(9021,2) and pair.dtype.kind in 'iu'
    assert ((pair[:,0]>=0)&(pair[:,0]<len(arm))).all()
    evaluation_cls=cls.copy();evaluation_cls[(cls==1)&(arm[pair[:,0]]>=2)]=2;evaluation_cls[cls==2]=3
    for key,value in mechanism.items():assert value.shape[:2]==(960,64) and np.isfinite(value).all(),key
    end = 0
    shadow_nonexempt_env_steps = 0
    full_negative_pre_env_steps = 0
    below_braking_band_pre_env_steps = 0
    selected = 0
    motion_only_added = 0
    motion_only_env_steps = 0
    for chunk in receipt['chunks']:
        file = path / chunk['path']; assert sha(file) == chunk['sha256']
        start, stop = chunk['start'], chunk['stop']; assert start == end
        z = load(file)
        assert z['forecast'].shape == (stop-start, 64, 9021)
        for key in ['forecast', 'measured_d', 'dmin','target_forecast','cv_forecast','pd_forecast','cv_h6','pd_h6']:
            assert z[key].shape==(stop-start,64,9021) and z[key].dtype==np.float32
            assert np.isfinite(z[key]).all(), (file, key)
        assert np.array_equal(z['forecast'].min(-1), mechanism['full_forecast_min'][start:stop])
        assert (z['forecast'] <= z['measured_d']).all()
        expected_forecast=z['target_forecast']
        if mode in ('velocity_admission','motion_admission'):
            expected_forecast=np.minimum(expected_forecast,z['cv_forecast'])
        if mode in ('pd_admission','motion_admission'):
            expected_forecast=np.minimum(expected_forecast,z['pd_forecast'])
        assert np.array_equal(expected_forecast,z['forecast']), 'registered nominal factor membership differs'
        for key in ('target_forecast','cv_forecast','pd_forecast'):
            assert (z[key] <= z['measured_d']).all(), key
        for key in ('cv_q_horizons','pd_q_horizons'):
            assert z[key].shape==(stop-start,4,64,26) and np.isfinite(z[key]).all(),key

        exempt = np.unpackbits(z['exempt'], axis=-1)[..., :9021].astype(bool)
        missed = np.unpackbits(z['shadow_missed'], axis=-1)[..., :9021].astype(bool)
        for key in ['exempt','shadow_missed']:
            assert z[key].shape==(stop-start,64,1128) and z[key].dtype==np.uint8
            assert not np.unpackbits(z[key],axis=-1)[...,9021:].any()
        assert np.array_equal(missed.sum(-1), mechanism['shadow_reference_missed_count'][start:stop])
        shadow_nonexempt_env_steps += int((missed & ~exempt).any(-1).sum())
        full_negative_pre_env_steps += int(((z['measured_d'] < 0) & ~exempt).any(-1).sum())
        below_braking_band_pre_env_steps += int(((z['measured_d']-z['dmin'] < 0) & ~exempt).any(-1).sum())
        masks={}
        for key in ['selected_ids','baseline_ids']:
            ids=z[key];assert ids.shape==(stop-start,64,capacity) and ids.dtype==np.int32
            assert ((ids>=-1)&(ids<9021)).all()
            valid=ids>=0;flat=ids.reshape(-1,capacity);fv=valid.reshape(-1,capacity)
            dense=np.zeros((flat.shape[0],9021),bool)
            rr=np.broadcast_to(np.arange(flat.shape[0])[:,None],flat.shape)
            dense[rr[fv],flat[fv]]=True
            masks[key]=dense.reshape(stop-start,64,9021)
            assert np.array_equal(masks[key].sum(-1),valid.sum(-1)), 'duplicate selected IDs'
        base=masks['baseline_ids'];actual=masks['selected_ids']
        critical=z['measured_d']<=z['dmin']+.010
        assert (base|~critical).all(), 'baseline missing raw-critical row'
        if capacity==9021:
            expected=base|(z['forecast']<=z['dmin']+.010)
            assert np.array_equal(expected,actual), 'forecast union membership differs; no dropped/swapped rows'
            added=actual&~base
            assert np.array_equal(added.sum(-1),mechanism['forecast_added_count'][start:stop])
        else:
            assert np.array_equal(base,actual) and not mechanism['forecast_added_count'][start:stop].any()
        original_target_mask=base|(z['target_forecast']<=z['dmin']+.010)
        motion_only=actual & ~original_target_mask
        assert (actual|~original_target_mask).all(), 'old target-admitted row dropped'
        if mode=='joint_reference':assert not motion_only.any()
        motion_only_added+=int(motion_only.sum())
        motion_only_env_steps+=int(motion_only.any(-1).sum())
        valid=z['selected_ids']>=0
        assert np.array_equal(valid.sum(-1), data['critical_selected_count'][start:stop])
        assert np.array_equal(valid.sum(-1),mechanism['required_rows'][start:stop])
        # Full raw nonexempt pre geometry matches the previous dense post frame.
        margin=np.stack([np.where(exempt| (evaluation_cls!=c),np.inf,z['measured_d']).min(-1) for c in range(4)],-1)
        offset=1 if start==0 else 0
        assert np.array_equal(margin[offset:],data['official_margins'][start+offset-1:stop-1])
        selected += int(valid.sum()); end=stop
    assert end == 960
    diag = load(path / 'project_diagnostics.npz')
    for k, v in diag.items():
        assert v.shape[0] == 960 and np.isfinite(v).all(), k
    # The returned command includes R19. The target delta includes the later
    # original soft-limit target clamp; both are distinct from measured PD motion.
    assert np.array_equal(diag['outer_returned_cmd'], data['exec'])
    assert np.array_equal(diag['external_raw_cmd'], data['cmd'])
    assert np.array_equal(diag['governor_changed'], data['governor_changed'])
    before = np.concatenate([data['q_initial'][None], data['controller_target'][:-1]])
    assert np.array_equal(data['effective_target_delta'], data['controller_target']-before)
    debt = before-np.concatenate([data['q_initial'][None],data['q'][:-1]])
    assert np.array_equal(debt, data['pre_target_debt'])
    fifo_pre = np.stack([np.concatenate([
        np.repeat(data['q_initial'][None], max(0, 6-i), 0),
        data['controller_target'][:max(0,960-(6-i))]])[:960] for i in range(6)], 1)
    assert np.array_equal(fifo_pre, data['pre_pending_actuator_targets'])
    assert data['pre_pending_project_history'].shape==(960,6,64,26)
    assert np.isfinite(data['pre_pending_project_history']).all()
    assert np.array_equal(fifo_pre,data['pre_pending_project_history']), 'independent project target history differs from original FIFO order'
    summaries = {}
    for robot in ['F','U']:
        for kind in ['original_safety', 'returned_safety', 'returned_alpha', 'returned_bound',
                     'target_safety','target_alpha','target_bound']:
            key=kind+'_residual_'+robot
            summaries[key] = dict(maximum=float(diag[key].max()), env_steps_over_solver_tol=int((diag[key] > 1e-6).sum()))
        key='individual_infeasibility_lower_bound_'+robot
        summaries[key]=dict(maximum=float(diag[key].max()), positive_env_steps=int((diag[key]>0).sum()))
    return dict(all_960_full9021_forecast_finite=True, rows_per_step=9021, frames=960,
        values_checked=960*64*9021, selected_ids_count=selected,
        motion_only_added_rows_at_own_state=motion_only_added,
        motion_only_added_env_steps=motion_only_env_steps,
        forecast_added_count_scope='includes old target forecast additions beyond measured/actor union; motion-only reported separately at own state',
        full_J_scope='producer finite checks all9021 before each forecast; selected J at9 snapshots, no full J archive',
        actual_fifo_pending_exact=True, separate_project_history_saved=True,project_history_order_exact=True,
        complete_selected_union_membership_verified=True,full_pre_margin_bound_to_previous_dense_post=True,
        raw_proposal_not_conservative_shadow_nonexempt_env_steps=shadow_nonexempt_env_steps,
        negative_full_pre_geometric_margin_env_steps=full_negative_pre_env_steps,
        below_braking_dmin_pre_env_steps=below_braking_band_pre_env_steps,
        maximum_target_debt_rad=float(np.abs(debt).max()),
        unreachable_reference_env_steps=int(mechanism['reference_envelope_unreachable'].any(-1).sum()),
        governor_modified_env_steps=int(data['governor_changed'].any(-1).sum()),
        max_target_project_difference_rad=float(np.abs(data['effective_target_delta']-data['exec']).max()),
        residuals=summaries)


