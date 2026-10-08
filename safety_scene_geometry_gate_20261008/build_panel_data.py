"""Package measured outcomes; never turns refusals into successful operations."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008')
read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
data=dict(runs={},traces=[],images=[],videos=read(p/'VIDEO_RECEIPT.json')['videos'])
summary=dict(date='2026-10-08',full_system0_acceptance=False,new_manipulation_tasks=0,threshold_n=.1,runs={},limits=[
    '仅固定四臂姿态下的两U手空手闭合；没有新完整System0协同任务。',
    '随机的是轨迹帧；26自由度全联合限位的随机操作支持为0。',
    '12个手部目标经过旧开发筛选；动作范围没有扩展到任意48维手指组合。',
    '几何包围盒与余量可能过度拒绝；拒绝不计作完成。',
    '移动障碍、真实夹持、带载释放、图像感知及构造期接触仍未验收。',
    '重复循环、左右手及同轨迹姿态相关；不报告独立任务事故概率置信界。'])
for run in ['matched_scene','fresh_scene_batch']:
    res=read(p/(run.upper()+'_FULL_RESULTS.json'));reg=read(p/(f'REGISTRATION_{run.upper()}.json'));receipt=read(raw/run/'recording_receipt.json');meta=read(raw/run/'native_metadata.json')
    summary['runs'][run]={k:res[k] for k in ['status','u_hand_paths','methods','env_states','native_cycles','original_images','abort_count','geometry_refusals','maximum_fk_position_error_m','maximum_fk_quaternion_component_error','arm_actual_target_deviation_rad','arm_target_max_error_rad','registration_sha256','recording_receipt_sha256','geometry_schema_sha256']}
    summary['runs'][run]['references_s']=[c['reference_time_s'] for c in reg['cases']]
    data['runs'][run]=dict(rows=[{k:row[k] for k in ['cycle','env','arm','profile_index','reference_time_s','method','no_motion_target','admitted','qualified','refusal_reason','neutral_safe','path_max_scalar_all_partner_normal_n','closed_error_rad','return_error_rad','peak_speed_rad_s']} for row in res['rows'] if row['arm'].startswith('U')],recoveries=res['recoveries'],events=read(raw/run/'admission_records.json'))
    for im in receipt['images']:
        if im['step']%720 in [359,659]:data['images'].append(dict(run=run,**im))
    if run=='matched_scene':
        for env in range(8):
            for arm in ['U_L','U_R']:
                series=[]
                for ch in receipt['chunks'][:6]:
                    with np.load(raw/run/ch['file']) as z:
                        for ix,step in enumerate(z['step']):
                            # Curves include every native physics frame, not only render frames.
                            series.append([int(step),float(z[f'e{env}:{arm}:hand_partner_scalar_normal'][ix].max()),float(abs(z[arm+':qd'][ix,env,meta['arms'][arm]['hand_ids']]).max())])
                data['traces'].append(dict(env=env,arm=arm,reference_time_s=reg['cases'][env]['reference_time_s'],method=reg['cases'][env]['method'],dt=reg['physics_dt_s'],series=series))
    # Explicit marginal support: these are trajectory-derived poses, not a reachable volume.
    spans=[]
    for arm,m in meta['arms'].items():
        q=np.array(m['arm_reference_rad']);ids=m['arm_ids'];limits=np.array(m['soft_limits_rad'])[0,ids,:]
        spans.extend(((q.max(0)-q.min(0))/(limits[:,1]-limits[:,0])).tolist())
    summary['runs'][run]['26_joint_marginal_span_fraction_minmax']=[min(spans),max(spans)]
summary['interrupted_attempts']=[read(p/'INTERRUPTED_FRESH_ATTEMPT.json'),read(p/'INTERRUPTED_FRESH_RETRY_ATTEMPT.json')]
summary['recorder_validation']=read(p/'BATCH_PILOT_READY_MAPPING_GATE.json')
summary['failed_initializations']=[read(p/'BATCH_PILOT_FAILURE_GATE.json'),read(p/'BATCH_PILOT_RETRY_FAILURE_GATE.json')]
summary['total_native_env_states']=sum(x['env_states'] for x in summary['runs'].values())
summary['original_native_images']=sum(x['original_images'] for x in summary['runs'].values())
fresh=summary['runs']['fresh_scene_batch']['methods']['scene_v4']
summary['limited_empty_hand_candidate_pass']=all(m['aborts']==0 and m['admitted']==m['executed_qualified'] and m['refused']==m['refused_neutral_safe'] and m['moving_admitted']>0 for m in [fresh,summary['runs']['matched_scene']['methods']['scene_v4']])
workspace=read(p/'WORKSPACE_CHARACTERIZATION.json')
summary['observed_workspace']={k:workspace[k] for k in ['joint_dimensions','scope']}
summary['observed_workspace']['six_arm_pairs']=[{k:row[k] for k in ['arms','minimum_aabb_distance_lower_bound_m','zero_aabb_distance_samples']} for row in workspace['six_arm_pairs']]
summary['observed_workspace']['wrist_xyz_span_m']={arm:row['xyz_span_m'] for arm,row in workspace['wrist_positions'].items()}
data['workspace']=workspace
summary['paired_admission_tradeoff']=read(p/'PAIRED_ADMISSION_TRADEOFF.json')['summary']
summary['admission_fraction']=fresh['admitted']/fresh['requests']
summary['mechanism']='USD实际碰撞尺寸与关节FK → 完整闭合路径/对侧手扫掠预测 → 动作前准入 → 原生逐点法向接触监测 → 锁存空手退回'
summary['diagnostic_sha256']=sha(p/'ARCHIVED_FAILURE_DIAGNOSTIC.json')
for fn,obj in [('SUMMARY.json',summary),('PANEL_DATA.json',data)]:
    (p/fn).write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
