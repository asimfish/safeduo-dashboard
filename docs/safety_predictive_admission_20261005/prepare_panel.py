import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_predictive import load

HERE=Path(__file__).resolve().parent
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')
LABELS={'raw':'无安全保护','baseline':'原System 0动态512','envelope_050':'固定积压限制','predictive':'完整速度提前入选','predictive_envelope':'提前入选＋积压限制'}
COLORS={'raw':'#dc2626','baseline':'#6b7280','envelope_050':'#d97706','predictive':'#2563eb','predictive_envelope':'#7c3aed'}



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview-json',type=Path);args=parser.parse_args()
    r=json.loads((HERE/'results.json').read_text())
    zero_path=HERE.parent/'safety_zero_command_20261005/results.json'
    new_zero_path=HERE.parent/'safety_zero_predictive_20261005/results.json'
    if new_zero_path.exists():zero_path=new_zero_path
    if zero_path.exists():
        r['zero_command_diagnostic']=json.loads(zero_path.read_text())
        for x in r['zero_command_diagnostic']['aggregate_rows']:x['method_label']=LABELS[x['mode']]
    speed_path=HERE.parent/'safety_speed_box_control_20261005/same_device_results.json'
    if speed_path.exists():
        r['speed_box_control']=json.loads(speed_path.read_text())
        for x in r['speed_box_control']['aggregate_rows']:x['method_label']=LABELS.get(x['mode'],'仅速度限制（无安全投影）')
        if r['speed_box_control']['completed_windows']==192 and r['speed_box_control']['comparison_completed_windows']==384:
            r['speed_comparison_rows']=deepcopy(r['speed_box_control']['aggregate_rows'])

    r['summary']=f'完整速度入选因子与全新初态/指令留出：{r["completed_windows"]}/960完整16秒窗口，无效{r["invalid_windows"]}，待完成{r["pending_windows"]}；192组新初态与外部随机指令、五方法配对；完整批次提供全部192组轨迹与保护器放大视图。'
    r['protocol_note']='新种子331047829/389116237/451902773，每种子六臂对各8风险初态＋16广域稳定初态；新的初态库，不复用上一批留出。五方法同初态、同全部外部随机输入，各自新进程；严格六步FIFO约100ms。全16秒计入安全分母，60步零输入＋900步压力；四臂独立方向和异步保持1/4/15/30/90/180步，幅度0.005/0.015/0.025/0.05rad。全部失败保留，参数在开发前冻结，不依留出结果调参。'
    r['protocol_note']+=f' 记录dt={r["dt"]:.6f}s，960步名义16秒；FIFO实际6×dt。raw沿用原无保护旁路，不受投影器约0.025rad/步速度盒限制；四个保护条件共享该限制，因此raw差异不能全部归因于安全约束。'
    r['protocol_note']+=f' 已完整记录窗口的60步零前缀违规{sum(x["zero_prefix_violations"] for x in r["aggregate_rows"])}，首随机目标实际到达前违规{sum(x["before_first_random_delivery"] for x in r["aggregate_rows"])}；保护器可改变早期运动，实际66步q前缀是否一致另列。'
    r['mechanism_note']='新因子只增加安全行入选：使用在线完整身体速度，预测distance−horizon×正接近速度≤当前有效dmin＋10mm；cross/self/table时间0.16/0.40/0.30s，精确保留旧入选集合和瞬时关键行，动态预算2048，超限中止。投影、actor32、豁免及六步FIFO不改。旧固定±0.050rad积压限制单独复测并列组合；包络不可达时按原增量逐步回收，实际可暂超±0.050rad，由解析oracle核验，不瞬间重置目标。'
    r['mechanism_note']+=' 违规按非豁免安全球距裕量<0判定，超过5mm指相对该阈值的不足，不等同真实网格或接触侵入深度。'
    r['mechanism_note']+=' 软限位oracle核验发出的目标；实际关节状态不据此假定始终在软限位内。'
    traces=r['diagnostic']['traces']
    r['diagnostic_note']=f'旧开发离线估计：baseline58个首次分类失败事件中{traces[0]["events_with_missed_predicted_frames"]}个有预测关键帧未入选；旧积压限制9个事件中{traces[1]["events_with_missed_predicted_frames"]}个有此现象。缺失行使用后续有限差分速度（hindsight）和名义dmin，条件豁免可能使估计偏多，不能冒充当帧完整有效dmin下的线上验证。仅两个新预测方法使用当前完整身体速度，各帧验证旧行无丢失/预测关键行无遗漏；旧方法该指标未执行，其0数组只是占位。更早入选不等于动态安全保证。'
    sampling=r['bank_audit']['sampling']
    r['bank_note']=f'初态库先于机制测试登记：{sampling["geometry_candidates"]:,}个几何候选、{sampling["settling_candidates"]:,}条原地资格记录均保存；192个选择引用核验。初始保留球距≥1mm，1秒raw零输入无违规、漂移≤0.05rad、末速≤0.1rad/s。风险臂对起距20–60mm，general从95%软限位跨度LHS筛选；候选数不计入闭环窗口。'
    for row in r['rows']:
        row['admission_metrics_evaluated']=row['admission']['predictive']
        if not row['admission_metrics_evaluated']:
            row['predicted_required_missing']=None;row['legacy_rows_missing']=None
    completed={x['mode']:x for x in r['aggregate_rows']}
    verdict=[]
    for left,right in [('baseline','predictive'),('envelope_050','predictive_envelope')]:
        if left not in completed or right not in completed:continue
        a,b=completed[left],completed[right]
        matches=[x for x in r['paired_inputs'] if x['left_mode']==left and x['right_mode']==right and 'both_failure' in x]
        repaired=sum(x['left_only_failure'] for x in matches);new=sum(x['right_only_failure'] for x in matches)
        verdict.append(f'{LABELS[left]}→{LABELS[right]}：违规{a["violations"]}/{a["windows"]}→{b["violations"]}/{b["windows"]}，同输入救回{repaired}条、新增{new}条失败；平均实际关节跨度{100*a["mean_within_window_joint_range"]:.2f}%→{100*b["mean_within_window_joint_range"]:.2f}%。')
    r['verdict']=' '.join(verdict)+' 全部仍按原球距负值判失败，安全准入与资料交付分开。保留新增失败和操作范围代价，候选未推广；边际关节/末端格不证明26维联合空间或物体协同任务，控制周期球距不证明连续接触或实机安全。'
    if r['completed_windows']!=960:
        r['verdict']='本批尚未完成，各方法已有结果可能来自不同种子集合，不比较未配齐的汇总分母。只按完整配对的种子展示轨迹，所有失败/无效与待完成数分别保留，不能先作总体安全结论。'
    for collection in ('rows','aggregate_rows','strata','paired_inputs'):
        for row in r[collection]:
            if 'mode' in row:row['method_label']=LABELS[row['mode']]
    for row in r.get('speed_comparison_rows',[]):
        if row['mode'] in LABELS:row['method_label']=LABELS[row['mode']]
    r['operation_joint_rows']=[dict(mode=m['mode'],method_label=LABELS[m['mode']],**g) for m in r['operation_pairwise'] for g in m['groups']]
    r['pair_rows']=[x for x in r['strata'] if x['stratum']>=0]
    r['coverage_cases']=[]
    for g in r['group_coverage']:
        c=g['coverage']
        r['coverage_cases'].append(dict(label=g['label']+' · '+LABELS[g['mode']],mode=g['mode'],stratum=g['stratum'],
            initial=c['initial_joint'],visited=c['visited_joint'],ee_initial=c['initial_ee'],ee_visited=c['visited_ee'],pairs=g['pairs'],
            note=f'{c["episodes"]}窗口全部960帧（含失败），平均单窗口关节跨度{100*c["mean_within_window_joint_range"]:.2f}%，总关节路径{c["mean_joint_path_rad"]:.2f}rad。橙框为初态、蓝色为实际访问；只表示边际访问，不表示26维联合或末端可达体积覆盖率。'))
    r['default_coverage']=next((i for i,x in enumerate(r['coverage_cases']) if x['mode']=='predictive' and x['stratum']==-2),0)
    r['traces']=[]
    hold=[x for x in r['rows'] if x['phase']=='holdout' and x['status']=='complete']
    for seed in sorted(set(x['seed'] for x in hold)):
        same={x['mode']:x for x in hold if x['seed']==seed}
        if len(same)!=5:continue
        d={k:load(Path(x['path'])/'cell_001.npz') for k,x in same.items()}
        labels=np.array(same['baseline']['risk_pair_index'])
        bad={k:(v['official_margins']<0).any((0,2)) for k,v in d.items()}
        blocks={k:v['official_margins'].reshape(320,3,64,4).min((1,3))*1000 for k,v in d.items()}
        group_names={x['stratum']:x['label'] for x in r['strata'] if x['mode']=='predictive_envelope'}
        for e in range(64):
            outcome='组合新增失败' if not bad['envelope_050'][e] and bad['predictive_envelope'][e] else '提前入选新增失败' if not bad['baseline'][e] and bad['predictive'][e] else '组合仍失败' if bad['predictive_envelope'][e] else '组合全窗0违规'
            series=[]
            for mode in LABELS:
                values=blocks[mode][:,e];shown=np.round(values,6)
                # Preserve strict-negative signs even for sub-rounding penetrations.
                shown=np.where((values<0)&(shown==0),values,shown)
                series.append(dict(label=LABELS[mode],color=COLORS[mode],values=shown.tolist()))
            low=min(min(x['values']) for x in series);high=max(max(x['values']) for x in series);pad=max(1.,(high-low)*.06)
            first={}
            for mode,v in d.items():
                frames=np.flatnonzero((v['official_margins'][:,e]<0).any(-1))
                first[mode]=float((int(frames[0])+1)*r['dt']) if len(frames) else None
            r['traces'].append(dict(label=f'{outcome} · {group_names[int(labels[e])]} · seed{seed} · env{e}',outcome=outcome,
                seed=seed,env=e,risk_pair_index=int(labels[e]),window_violations={k:bool(v[e]) for k,v in bad.items()},
                first_failure_times_s=first,class_violations={k:(v['official_margins'][:,e]<0).any(0).tolist() for k,v in d.items()},
                unit='mm',time_s=(np.arange(3,961,3)*r['dt']).tolist(),x_min=0,x_max=16,y_min=low-pad,y_max=high+pad,
                events=[dict(time_s=60*r['dt'],label='随机输入开始',color='#6b7280'),dict(time_s=66*r['dt'],label='首随机目标到达',color='#2563eb')],series=series,
                caption='全部192组按种子/环境固定顺序展示，含所有失败；五方法q0及全程外部输入逐位一致，保护会改变零输入阶段实际运动。每点为连续3帧（约50ms）区间最小官方非豁免裕量，保留所有负值区间，不能据区间点读精确触发时刻。违规分母与首次时间使用全960帧，显示数值保留到微米以下，极小负号保留。'))
    details=[]
    for trace in r['traces']:
        trace['caption']+=' 按全部960帧计算的首次违规：'+'；'.join(f'{LABELS[k]} '+('全窗未违规' if t is None else f'{t:.6f}s') for k,t in trace['first_failure_times_s'].items())+'。'
        detail=deepcopy(trace)
        detail['label']='保护器细节 · '+trace['label']
        detail['series']=detail['series'][1:]
        lo=min(min(s['values']) for s in detail['series']);hi=max(max(s['values']) for s in detail['series']);pad=max(1.,(hi-lo)*.06)
        detail['y_min']=lo-pad;detail['y_max']=hi+pad
        detail['caption']='这里放大四个保护版本的曲线；同seed/env的全方法曲线在选择列表中保留raw。显示范围不会改变全部960帧违规统计。 '+detail['caption']
        details.append(detail)
    r['traces']+=details
    r['default_trace']=next((i for i,t in enumerate(r['traces']) if t['label'].startswith('保护器细节') and t['outcome']=='组合新增失败'),
        next((i for i,t in enumerate(r['traces']) if t['label'].startswith('保护器细节') and t['outcome']=='提前入选新增失败'),
            next((i for i,t in enumerate(r['traces']) if t['label'].startswith('保护器细节') and t['outcome']=='组合仍失败'),0)))
    (HERE/'panel_data.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    report=['# 完整身体速度提前入选与全新随机留出\n',r['summary'],r['verdict'],r['mechanism_note'],r['protocol_note'],r['bank_note'],
            '\n|方法|完整/计划|违规|>5mm|cross / self_F / self_U / table|四臂同时动|输出/命令L2|目标积压P95|平均关节跨度/路径|\n|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for x in r['aggregate_rows']:
        report.append(f'|{x["method_label"]}|{x["windows"]}/192|{x["violations"]}|{x["deep"]}|{x["class_violations"]}|{100*x["four_arms_moving_fraction"]:.2f}%|{x["exec_command_l2_ratio"]:.4f}|{x["pre_target_debt_abs_p95_rad"]:.4f}rad|{100*x["mean_within_window_joint_range"]:.2f}% / {x["mean_joint_path_rad"]:.2f}rad|')
    report+=['\n四类失败可重叠。四臂活动量为随机输入发出后（第60步起）每控制步四臂关节位移L2均>1mrad；L2和P95为逐种子统计后平均，不是任务成功率或总体置信界。臂对暴露另列实际首随机目标到达后的第66步起统计。',
             'raw沿用原无保护旁路；四个保护方法受原速度盒约0.025rad/步限制，raw不受这一盒限制。不能把raw对照的全部改善归因于安全行；本轮两因子的消融在相同原System 0限制下比较。速度盒单独作用另以独立登记的同GPU1补充对照检验，主五方法条件保持冻结，补充结果不混入主960分母。',
             '\n## 配对失败变化\n','|种子|左 / 右|左独有失败|右独有失败|双方失败|双方通过|66步实际q前缀相同|最大q差rad|\n|---|---|---:|---:|---:|---:|---|---:|']
    for x in r['paired_inputs']:
        if 'both_failure' not in x:continue
        report.append(f'|{x["seed"]}|{LABELS[x["left_mode"]]} / {LABELS[x["right_mode"]]}|{x["left_only_failure"]}|{x["right_only_failure"]}|{x["both_failure"]}|{x["neither_failure"]}|{x["actual_q_prefix66_exact"]}|{x["actual_q_prefix66_max_diff_rad"]:.6f}|')
    report+=['\n## 臂对配额与真实压力\n','|臂对 / 方法|完整/24|违规|>5mm|≥3帧80mm：全窗/随机到达后|\n|---|---:|---:|---:|---:|']
    for x in r['pair_rows']:
        report.append(f'|{x["label"]} / {x["method_label"]}|{x["windows"]}/24|{x["violations"]}|{x["deep"]}|{x["target_pair_exposure"]["exposed_80mm"]} / {x["target_pair_pressure_exposure"]["exposed_80mm"]}|')
    report+=['\n## 首次违规前的实际操作\n','|方法|平均安全前缀秒|前缀平均关节跨度|前缀关节路径rad|\n|---|---:|---:|---:|']
    for x in r['aggregate_rows']:
        report.append(f'|{x["method_label"]}|{x["mean_safe_duration_s"]:.3f}|{100*x["mean_safe_prefix_joint_range"]:.2f}%|{x["mean_safe_prefix_joint_path_rad"]:.2f}|')
    report+=['首次负裕量的危险转移及其后所有运动从前缀指标排除；不同前缀长度同时报告，完整16秒运动指标仍保留。',
             '\n## 二维关节边际访问\n','|方法 / 臂组|关节对数|平均 / 最少 / 最多已访格（每对100格）|\n|---|---:|---:|']
    for x in r['operation_joint_rows']:
        report.append(f'|{x["method_label"]} / {x["arm_i"]}–{x["arm_j"]}|{x["joint_pairs"]}|{x["mean_cells"]:.2f} / {x["minimum_cells"]} / {x["maximum_cells"]}|')
    report+=['该二维格使用所有完整物理窗口含失败，超软限位样本不增加对应格。325个二维边际并不证明26维联合覆盖，UR周期同姿态未去重；补充指标在新压力留出前登记。',
             '实际状态越软限位的关节样本比例（全窗口含失败）：'+'；'.join(f'{LABELS[x["mode"]]} {100*x["outside_soft_joint_sample_fraction"]:.4f}%' for x in r['operation_pairwise'])+'。目标软限位通过不等于实际关节始终在软限位内；此处球距主要终点也不是实机安全认证。']
    if 'zero_command_diagnostic' in r:
        z=r['zero_command_diagnostic']
        report+=['\n## 16秒连续零指令对照（补充）\n',f'{z["completed_windows"]}/{z["registered_windows"]}完整窗口、无效{z["invalid_windows"]}、待完成{z["pending_windows"]}；复用本轮192个初态、0新增随机指令，不与主压力分母混合。原三方法576在主压力部分结果可见后、自身结果前登记；补充两个新因子384在已有零/速度补充结果可见后、自身结果前登记。参数/初态不改，不依此重筛初态或排除主压力失败。',
                 '|方法|完整/192|违规|四类cross/F/U/table|保护非零输出窗口|最大实际关节漂移|\n|---|---:|---:|---|---:|---:|']
        for x in z['aggregate_rows']:
            report.append(f'|{x["method_label"]}|{x["windows"]}/192|{x["violations"]}|{x["class_violations"]}|{x["nonzero_execution_windows"]}|{x["max_actual_joint_drift_rad"]:.6f}rad|')
        report+=['完整960步严格零输入逐帧核验；raw issued目标每帧等于q0。1秒资格不等于16秒保证；保护器可主动退让，是否失败以原官方负裕量判定。所有失败和原始16秒数组保留，零指令结果不能外推为随机压力安全。']
    if 'speed_box_control' in r:
        s=r['speed_box_control']
        report+=['\n## 速度限制单独作用（同GPU补充对照）\n',f'仅限速{s["completed_windows"]}/192完整窗口、无效{s["invalid_windows"]}、待完成{s["pending_windows"]}；配套raw/System0重复{s["comparison_completed_windows"]}/384，共{s["supplementary_total_windows"]}/576相关窗口。三方法均cuda:1；复用主实验192组初态和全部外部指令，0新增独立随机轨迹。主960条件未改变。仅限速按原vmax×dt裁剪增量，忽略actor门控与全部几何约束，原软目标限位/积分目标/FIFO保持。',
                 '补充登记时主压力部分结果已可见；仅限速在自身结果前登记，配套GPU1对照在仅限速运行完成但结果分析前登记。全三种子原样配套，没有依结果筛选。旧analyze_speed跨GPU0/GPU1比较仅作为审计保留，面板补充结论使用同GPU1三方法；不能假定跨GPU实际轨迹逐位相同。',
                 '|方法|完整/192|违规|>5mm|四类cross/F/U/table|平均关节跨度|平均安全前缀|前缀跨度|\n|---|---:|---:|---:|---|---:|---:|---:|']
        for x in r.get('speed_comparison_rows',[]):
            report.append(f'|{x["method_label"]}|{x["windows"]}/192|{x["violations"]}|{x["deep"]}|{x["class_violations"]}|{100*x["mean_within_window_joint_range"]:.2f}%|{x["mean_safe_duration_s"]:.3f}s|{100*x["mean_safe_prefix_joint_range"]:.2f}%|')
        report+=['|种子 / 左→仅速度限制|左独有失败|仅限速独有失败|双方失败|双方通过|\n|---|---:|---:|---:|---:|']
        for x in s['paired_inputs']:
            if 'both_failure' in x:report.append(f'|{x["seed"]} / {LABELS[x["left_mode"]]}|{x["left_only_failure"]}|{x["right_only_failure"]}|{x["both_failure"]}|{x["neither_failure"]}|')
        report+=['独立NumPy裁剪/目标积分/软限位及实际FIFO oracle逐960帧核验；安全行残差未计算，零占位不是几何可行性证据。比较为相关配对窗口，不给IID安全认证；完整安全与真实操作仍须同时审视。']
    report+=['\n## 预登记、开发与证据边界\n',r['diagnostic_note'],
        '新候选与三个种子均在候选开发结果之前冻结。bank登记不依随机策略结果；其全部候选和静置资格结果与192个选择引用可追溯。在线完整9021几何的d/closing/dmin非有限会中止，不以min统计隐藏非有限；原模型/配置/279项生产源码不变。',
        '登记继承字段的历史说明与冻结说明稿残留见REGISTRATION_NOTE.md；当前真实设备以每条件argv/protocol为准，开发cuda:1，新留出cuda:0。原登记文件不做事后重写。',
        'CPU独立集合oracle覆盖250组随机边界、异构批padding、接近/远离、有效dmin/豁免以及超容量中止，6项通过；旧静态union在相同提前入选契约上失败。旧积压限制wrapper使用原解析oracle逐帧核验。',
        '旧候选9个首次事件中5个arm/full距离速度差≤1mm/s，不能把全部失败归因于未控身体速度；另外4个有差异但本比较不识别其来源，见RATE_DIAGNOSTIC.json。',
        '开发60317411仅两个新因子128重复窗口，不计新增独立输入。旧因果重播仅离线解释，不加入本轮960分母。',
        '已完成的队列目标预测实验另见[原独立面板](../safety_mechanism_20261005_causal_obs/)，其120/192→40/192来自不同机制、不同新指令且旧初态库，不与本轮合并。原有限性v2完整16秒评测、真实物体搬运/夹持与四臂任务仍为独立未完成策略关卡。',
        '四臂均有独立随机外部命令；观察运动量和关节/末端边际访问，不能称为26维空间充分覆盖或四臂有目标任务成功。纯随机压力不替代物体尺寸/状态感知、真实接触力、抓持保持或实机安全。',
        '全部旧scientific字段、队列证据独立面板与视频保留；本轮仅增加新字段与报告。所有大型原始数组在NAS，公开报告提供小型协议、参数、代码和hash。']
    for x in [x for x in r['rows'] if x['phase']=='development']:
        report.append(f'- 旧开发 {x["method_label"]}：{x.get("violations","无效")}/64；状态{x["status"]}，不删失败。')
    if len(r['aggregate_rows'])==5:
        fig,axes=plt.subplots(1,3,figsize=(15,4));names=['Raw','System 0','Envelope','Predictive','Combined']
        axes[0].bar(names,[x['violations']/x['windows']*100 for x in r['aggregate_rows']],color=list(COLORS.values()))
        axes[0].set(ylabel='Windows with any violation (%)',ylim=(0,100),title='Independent holdout: full16s')
        axes[1].bar(names,[100*x['mean_within_window_joint_range'] for x in r['aggregate_rows']],color=list(COLORS.values()))
        axes[1].set(ylabel='Mean within-window joint soft-limit range (%)',ylim=(0,100),title='Actual operating range')
        axes[2].bar(names,[100*x['four_arms_moving_fraction'] for x in r['aggregate_rows']],color=list(COLORS.values()))
        axes[2].set(ylabel='Four arms moving per pressure step (%)',ylim=(0,100),title='Movement >1mrad is not task success')
        fig.tight_layout();fig.savefig(HERE/'predictive_effect.png',dpi=160);plt.close(fig)
        report+=['\n![安全表现与实际运动量](predictive_effect.png)']
    (HERE/'REPORT.md').write_text('\n'.join(report)+'\n')
    (HERE/'panel_data.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    previous=json.loads(DATA.read_text());current=deepcopy(previous)
    current.update(safety_predictive_admission=r,summary=r['summary'],updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    link=dict(label='最新：完整速度入选五方法与全新初态留出',url='https://asimfish.github.io/safeduo-dashboard/docs/'+HERE.name+'/REPORT.md')
    current['links']=[link]+[x for x in current['links'] if x['url']!=link['url']]
    for key,val in previous.items():
        if key not in ('safety_predictive_admission','summary','updated','links'):assert current[key]==val,key
    (args.preview_json or DATA).write_text(json.dumps(current,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    public=DASH/'docs'/HERE.name;public.mkdir(parents=True,exist_ok=True)
    keep={'PLAN.md','VERIFICATION.md','bank_registration.json','candidate_registration.json','development_plan.json','holdout_plan.json','admission_audit.json','AUDIT_LIMITATIONS.md','RATE_DIAGNOSTIC.json','REGISTRATION_NOTE.md','METRICS_ADDENDUM.md','METRICS_REGISTRATION.json','metric_tests.log','red_contract.log','contract_tests.log','results.json','REPORT.md','panel_data.json'}
    for file in HERE.iterdir():
        if file.is_file() and (file.name in keep or file.suffix=='.py' or file.name=='predictive_effect.png'):
            shutil.copy2(file,public/file.name)
    nas=Path('/mnt/nas/data/lyf/double_hand/safety_predictive_admission_20261005')
    for phase in ('development','holdout'):
        root=nas/(phase+'_cells')
        shutil.copy2(root/'campaign.json',public/(phase+'_campaign.json'))
        for job in json.loads((root/'campaign.json').read_text())['jobs']:
            dest=public/'protocols'/job['id'];dest.mkdir(parents=True,exist_ok=True)
            for name in ('protocol.json','admission_manifest.json','mechanism_manifest.json','random_manifest.json','capacity_failure.json','predictive_capacity_failure.json'):
                file=root/job['id']/name
                if file.exists():shutil.copy2(file,dest/name)
    for seed in (331047829,389116237,451902773):
        dest=public/'banks'/str(seed);dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(nas/'holdout_banks'/str(seed)/'metadata.json',dest/'metadata.json')
    zero=HERE.parent/'safety_zero_command_20261005'
    if (zero/'results.json').exists():
        dest=public/'zero_command_diagnostic';dest.mkdir(exist_ok=True)
        for file in zero.iterdir():
            if file.is_file() and file.suffix in ('.json','.py','.md'):shutil.copy2(file,dest/file.name)
        root=Path('/mnt/nas/data/lyf/double_hand/safety_zero_command_20261005/cells')
        shutil.copy2(root/'campaign.json',dest/'campaign.json')
        for j in json.loads((root/'campaign.json').read_text())['jobs']:
            sub=dest/'protocols'/j['id'];sub.mkdir(parents=True,exist_ok=True)
            for n in ('protocol.json','mechanism_manifest.json','random_manifest.json'):
                f=root/j['id']/n
                if f.exists():shutil.copy2(f,sub/n)
    speed=HERE.parent/'safety_speed_box_control_20261005'
    if (speed/'results.json').exists():
        dest=public/'speed_box_control';dest.mkdir(exist_ok=True)
        for file in speed.iterdir():
            if file.is_file() and file.suffix in ('.json','.py','.md'):shutil.copy2(file,dest/file.name)
        root=Path('/mnt/nas/data/lyf/double_hand/safety_speed_box_control_20261005/cells')
        shutil.copy2(root/'campaign.json',dest/'campaign.json')
        for j in json.loads((root/'campaign.json').read_text())['jobs']:
            sub=dest/'protocols'/j['id'];sub.mkdir(parents=True,exist_ok=True)
            for n in ('protocol.json','speed_box_manifest.json','random_manifest.json'):
                f=root/j['id']/n
                if f.exists():shutil.copy2(f,sub/n)
        comparison=root.parent/'comparison_cells'
        if (comparison/'campaign.json').exists():
            shutil.copy2(comparison/'campaign.json',dest/'comparison_campaign.json')
            for j in json.loads((comparison/'campaign.json').read_text())['jobs']:
                sub=dest/'protocols'/j['id'];sub.mkdir(parents=True,exist_ok=True)
                for n in ('protocol.json','mechanism_manifest.json','random_manifest.json'):
                    f=comparison/j['id']/n
                    if f.exists():shutil.copy2(f,sub/n)
    neutral_new=HERE.parent/'safety_zero_predictive_20261005'
    if (neutral_new/'results.json').exists():
        dest=public/'zero_predictive_diagnostic';dest.mkdir(exist_ok=True)
        for file in neutral_new.iterdir():
            if file.is_file() and file.suffix in ('.json','.py','.md'):shutil.copy2(file,dest/file.name)
        root=Path('/mnt/nas/data/lyf/double_hand/safety_zero_predictive_20261005/cells')
        shutil.copy2(root/'campaign.json',dest/'campaign.json')
        for j in json.loads((root/'campaign.json').read_text())['jobs']:
            sub=dest/'protocols'/j['id'];sub.mkdir(parents=True,exist_ok=True)
            for n in ('protocol.json','admission_manifest.json','mechanism_manifest.json','random_manifest.json'):
                f=root/j['id']/n
                if f.exists():shutil.copy2(f,sub/n)
    print(json.dumps(dict(coverage=len(r['coverage_cases']),traces=len(r['traces']),legacy_preserved=True)))

if __name__=='__main__':main()
