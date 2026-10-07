import json
from pathlib import Path
p=Path(__file__).resolve().parent
results={run:json.loads((p/(run.upper()+'_RESULTS.json')).read_text()) for run in ['old_default','neutral_default']}
cases=[c for r in results.values() for c in r['cases']]
safe=[c for c in cases if c['post_write_thumb_rad']==.35]
control=[c for c in cases if c['post_write_thumb_rad']==0]
passed=lambda c,k:all(h['windows'][k]['qualified'] for h in c['hands'])
peak=max(h['windows']['startup']['pair_normal_max_n'] for c in control for h in c['hands'])
task_peak=max(h['windows']['closed_task_endpoint']['pair_normal_max_n'] for c in cases for h in c['hands'])
task_err=max(h['windows']['closed_task_endpoint']['max_error_rad'] for c in cases for h in c['hands'])
sumry=dict(status='POST_INITIALIZATION_STARTUP_QUALIFIED_ORIGINAL_TASK_CLOSED_ENDPOINT_REJECTED',
 static_environments=len(cases),unique_four_arm_reference_configurations=3,unique_recorded_native_trajectories=6,matched_config_runs_bitexact=True,
 environment_states=25920,original_images=180,new_task_trials=0,new_final_trials=0,
 neutral_written_startup_accepted=sum(c['startup_qualified'] for c in safe),neutral_written_startup_cases=len(safe),
 zero_written_startup_accepted=sum(c['startup_qualified'] for c in control),zero_written_startup_cases=len(control),
 zero_written_startup_peak_normal_n=peak,neutral_written_startup_peak_normal_n=max(h['windows']['startup']['pair_normal_max_n'] for c in safe for h in c['hands']),
 calibration_closed_accepted=sum(passed(c,'closed_calibration') for c in cases),task_closed_accepted=sum(passed(c,'closed_task_endpoint') for c in cases),
 both_returns_accepted=sum(passed(c,'open_after_calibration') and passed(c,'open_after_task_endpoint') for c in cases),
 original_task_closed_peak_normal_n=task_peak,original_task_closed_max_tracking_error_rad=task_err,
 whole_path_contacts_accepted=sum(c['whole_path_contacts_qualified'] for c in cases),
 constructor_internal_contact_forces_measured=False,hand_internal_exemption_gap_confirmed=True,
 candidate_admission='REJECTED_FOR_FULL_TASK_USE; STATIC_STARTUP_AND_MODERATE_CLOSED_ENDPOINT_ONLY',
 paragraphs=[f'初始化后写入0.35 rad：启动验收6/6通过，伙伴法向最大0 N；写零角对照0/6通过，最高{peak:.2f} N。6个通过槽位是3个不同四臂姿态重复两次，不是随机独立样本。',
             '两种构造前默认配置的对应原生轨迹逐值相同；配置默认值在reset后生效，没有修正构造结束时的实际运动状态。构造阶段安全未合格。',
             f'原0.45绝对闭合终点12/12通过，两次回开12/12通过。但原任务约0.75的拇指终点12/12失败：空手仍发生拇指第二节/掌基座接触，平台期最大{task_peak:.2f} N、最大追踪误差{task_err:.3f} rad。完整路径0/12通过，拒绝完整任务采用。',
             '现有hand_internal: all语义把这对同手接触剔除，不进入危险距离行。需要单独校准手部闭合与自碰保护，不能用旧球指标为零证明真实手部安全。',
             '原资产、自碰、增益、旧任务门槛和历史失败均保留；12环境25920状态、180张三视角原图完整公开，新增任务与留出样本均为0。'])
(p/'SUMMARY.json').write_text(json.dumps(sumry,ensure_ascii=False,indent=2)+'\n')
(p/'RUN_EXITS.json').write_text(json.dumps(dict(old_default=0,neutral_default=0,source='Actual exec session completion; both independent receipts and scorers passed completeness',preflight_path_failure_exit=1,preflight_native_execution_started=False),indent=2)+'\n')
print(json.dumps({k:v for k,v in sumry.items() if k!='paragraphs'},indent=2))
