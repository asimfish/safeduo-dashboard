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
from analyze_mechanism import load

HERE=Path(__file__).resolve().parent
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')
LABELS={'raw':'无安全保护','baseline':'原System 0动态512',
        'box_only':'System 0＋增量限制','envelope_050':'System 0＋增量与积压限制'}
COLORS={'raw':'#dc2626','baseline':'#6b7280','box_only':'#d97706','envelope_050':'#059669'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview-json',type=Path);args=parser.parse_args()
    r=json.loads((HERE/'results.json').read_text())
    r['summary']=f'机制因子对照与独立留出：已完成{r["completed_windows"]}/768个16秒窗口，无效{r["invalid_windows"]}，待完成{r["pending_windows"]}；192组全新外部随机轨迹。'
    r['protocol_note']='三个新种子152684921/198470327/237901613，每种子六臂对各8风险初态＋16广域稳定初态。四方法同初态、同全程外部输入，分别独立进程；严格6步FIFO（100ms）。全16秒计入安全分母，包括60步零输入；后900步四臂独立方向、异步保持，幅度0.005/0.015/0.025/0.05rad，保持1/4/15/30/90/180步。不删除失败，不根据留出结果调参。'
    r['mechanism_note']='新机制：把持久目标限制在实测关节位置±0.050rad内，同时守住原0.025rad每步增量和软限位。包络本步不可达时只能逐步移向最近可达端点；旧目标仍按原FIFO交付，无队列抢占或目标瞬间重置。增量限制另列单因子，alpha作用于限幅后的输入；actor与安全预测参数冻结。'
    diagnostic=r['diagnostic']
    r['diagnostic_note']=f'旧60317411取证重播64窗口，39违规；初态、命令、实际q、执行、裕量、控制/设备目标与旧测试逐帧一致，目标历史与真实FIFO逐帧一致。首次违规帧的危险行已被选中；32个桌面首次事件均无positive-cap结构行跳过。234个近失败线性可行性快照中9个不可行。已看到退让目标发出后，旧排队目标与惯性仍在闭合；这支持积压缓解实验，不能断言唯一根因。'
    sampling=r['bank_audit']['sampling']
    r['bank_note']=f'初态库先于机制测试登记：{sampling["geometry_candidates"]:,}个几何候选、{sampling["settling_candidates"]:,}条原地资格记录均保存；192个选择引用核验。初始保留球距≥1mm，1秒raw零输入无违规、漂移≤0.05rad、末速≤0.1rad/s。风险臂对起距20–60mm，general从95%软限位跨度LHS筛选；候选数不计入闭环窗口。'
    completed={x['mode']:x for x in r['aggregate_rows']}
    if 'baseline' in completed and 'envelope_050' in completed:
        a,b=completed['baseline'],completed['envelope_050']
        matched=[x for x in r['paired_inputs'] if x['left_mode']=='baseline' and x['right_mode']=='envelope_050' and 'both_failure' in x]
        repaired=sum(x['left_only_failure'] for x in matched);regressions=sum(x['right_only_failure'] for x in matched)
        r['verdict']=f'独立留出原System 0违规{a["violations"]}/{a["windows"]}，积压限制候选违规{b["violations"]}/{b["windows"]}。同输入配对消除{repaired}条基线失败，候选新增{regressions}条失败。候选压力阶段四臂每步均移动>1mrad为{100*b["four_arms_moving_fraction"]:.2f}%，输出/外部命令L2总量比{b["exec_command_l2_ratio"]:.3f}；平均单窗口关节跨度{100*b["mean_within_window_joint_range"]:.2f}%、路径{b["mean_joint_path_rad"]:.2f}rad。'+('仍有违规，候选未通过安全准入，保持评测用途。' if b['violations'] else '该批次未发现球层违规，仍不能据此推广为完整安全保证。')
    else:
        r['verdict']='实验尚未完成，不将未完成窗口视为安全。'
    r['verdict']+=' 边际关节格不证明26维联合空间覆盖；周期末球距不证明连续接触、物体操作或实机安全。运动范围和全部失败并列展示，旧结果完整保留。'
    for collection in ('rows','aggregate_rows','strata','paired_inputs'):
        for row in r[collection]:
            if 'mode' in row:row['method_label']=LABELS[row['mode']]
    r['pair_rows']=[x for x in r['strata'] if x['stratum']>=0]
    r['coverage_cases']=[]
    for g in r['group_coverage']:
        c=g['coverage']
        r['coverage_cases'].append(dict(label=g['label']+' · '+LABELS[g['mode']],mode=g['mode'],stratum=g['stratum'],
            initial=c['initial_joint'],visited=c['visited_joint'],ee_initial=c['initial_ee'],ee_visited=c['visited_ee'],pairs=g['pairs'],
            note=f'{c["episodes"]}窗口全部960帧（含失败），平均单窗口关节跨度{100*c["mean_within_window_joint_range"]:.2f}%，总关节路径{c["mean_joint_path_rad"]:.2f}rad。橙框为初态、蓝色为实际访问；只表示边际访问，不表示26维联合或末端可达体积覆盖率。'))
    r['default_coverage']=next((i for i,x in enumerate(r['coverage_cases']) if x['mode']=='envelope_050' and x['stratum']==-2),0)
    r['traces']=[]
    hold=[x for x in r['rows'] if x['phase']=='holdout' and x['status']=='complete']
    for seed in sorted(set(x['seed'] for x in hold)):
        same={x['mode']:x for x in hold if x['seed']==seed}
        if len(same)!=4:continue
        d={k:load(Path(x['path'])/'cell_001.npz') for k,x in same.items()}
        labels=np.array(same['baseline']['risk_pair_index'])
        bad={k:(v['official_margins']<0).any((0,2)) for k,v in d.items()}
        for group in [x for x in r['strata'] if x['mode']=='envelope_050' and x['stratum']>=-1]:
            available=np.flatnonzero(labels==group['stratum'])
            examples=[]
            for title,mask in [('原保护失败、候选全窗0违规',bad['baseline']&~bad['envelope_050']),
                               ('候选仍失败',bad['envelope_050']),
                               ('候选新增失败',~bad['baseline']&bad['envelope_050'])]:
                choices=available[mask[available]]
                if len(choices):examples.append((title,int(choices[0])))
            if not examples:examples=[('两保护版本全窗0违规',int(available[0]))]
            for title,e in examples:
                idx=np.arange(0,960,3);series=[]
                for mode in LABELS:
                    series.append(dict(label=LABELS[mode],color=COLORS[mode],values=(d[mode]['official_margins'][idx,e].min(-1)*1000).tolist()))
                low=min(min(s['values']) for s in series);high=max(max(s['values']) for s in series);pad=max(1.,(high-low)*.06)
                r['traces'].append(dict(label=f'{title} · {group["label"]} · seed{seed} · env{e}',outcome=title,
                    seed=seed,env=e,window_violations={mode:bool(value[e]) for mode,value in bad.items()},
                    unit='mm',time_s=((idx+1)*r['dt']).tolist(),x_min=0,x_max=16,y_min=low-pad,y_max=high+pad,
                    events=[dict(time_s=1.,label='随机输入开始',color='#6b7280'),dict(time_s=1.1,label='首随机目标到达',color='#2563eb')],series=series,
                    caption='每种子/初态分层按最低编号展示基线失败而候选全窗0违规、候选仍失败和候选新增失败，缺少类别就不造样例。四方法外部输入与q0逐位一致；保护会改变零输入阶段实际运动。曲线每3帧取点，违规分母使用全部960帧，纵轴是官方非豁免球层距最小值。'))
    details=[]
    for trace in r['traces']:
        detail=deepcopy(trace)
        detail['label']='保护器细节 · '+trace['label']
        detail['series']=detail['series'][1:]
        lo=min(min(s['values']) for s in detail['series']);hi=max(max(s['values']) for s in detail['series']);pad=max(1.,(hi-lo)*.06)
        detail['y_min']=lo-pad;detail['y_max']=hi+pad
        detail['caption']='这里放大三个保护版本的曲线；同seed/env的全方法曲线在选择列表中保留raw。显示范围不会改变全部960帧违规统计。 '+detail['caption']
        details.append(detail)
    r['traces']+=details
    r['default_trace']=next((i for i,t in enumerate(r['traces']) if t['label'].startswith('保护器细节') and t['outcome']=='原保护失败、候选全窗0违规'),0)
    (HERE/'panel_data.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    report=['# 目标积压机制与独立随机留出\n',r['summary'],r['verdict'],r['mechanism_note'],r['protocol_note'],r['bank_note'],
            '\n|方法|完整/计划|违规|>5mm|cross / self_F / self_U / table|四臂同时动|输出/命令L2|目标积压P95|平均关节跨度/路径|\n|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for x in r['aggregate_rows']:
        report.append(f'|{x["method_label"]}|{x["windows"]}/192|{x["violations"]}|{x["deep"]}|{x["class_violations"]}|{100*x["four_arms_moving_fraction"]:.2f}%|{x["exec_command_l2_ratio"]:.4f}|{x["pre_target_debt_abs_p95_rad"]:.4f}rad|{100*x["mean_within_window_joint_range"]:.2f}% / {x["mean_joint_path_rad"]:.2f}rad|')
    report+=['\n四类失败可重叠。四臂活动量为随机输入发出后（第60步起）每控制步四臂关节位移L2均>1mrad；L2和P95为逐种子统计后平均，不是任务成功率或总体置信界。臂对暴露另列实际首随机目标到达后的第66步起统计。',
             '\n## 配对失败变化\n','|种子|左 / 右|左独有失败|右独有失败|双方失败|双方通过|66步实际q前缀相同|最大q差rad|\n|---|---|---:|---:|---:|---:|---|---:|']
    for x in r['paired_inputs']:
        if 'both_failure' not in x:continue
        report.append(f'|{x["seed"]}|{LABELS[x["left_mode"]]} / {LABELS[x["right_mode"]]}|{x["left_only_failure"]}|{x["right_only_failure"]}|{x["both_failure"]}|{x["neither_failure"]}|{x["actual_q_prefix66_exact"]}|{x["actual_q_prefix66_max_diff_rad"]:.6f}|')
    report+=['\n## 臂对配额与真实压力\n','|臂对 / 方法|完整/24|违规|>5mm|≥3帧80mm：全窗/随机到达后|\n|---|---:|---:|---:|---:|']
    for x in r['pair_rows']:
        report.append(f'|{x["label"]} / {x["method_label"]}|{x["windows"]}/24|{x["violations"]}|{x["deep"]}|{x["target_pair_exposure"]["exposed_80mm"]} / {x["target_pair_pressure_exposure"]["exposed_80mm"]}|')
    report+=['\n## 原始开发回归与诊断\n',r['diagnostic_note'],
             '32个桌面事件中30个在前12步曾有未入选帧；不能将碰撞时入选解释为整个预测区间都足够提前。17个涉及条件接触；9个线性不可行快照对应4个环境。全body速度和目标积压已保存；不能凭线性投影可行性推断物理安全。']
    for x in [x for x in r['rows'] if x['phase']=='development']:
        report.append(f'- 旧60317411 {x["method_label"]}：{x["violations"]}/64违规；重复旧外部随机轨迹，不增加独立指令数。')
    candidate=r['candidate_diagnostic']
    r['candidate_diagnostic_note']=f'冻结候选另作64个重复观测窗口，q/输入/执行/裕量/目标均与开发候选逐帧一致，仍9违规（U自碰2、桌面7）。{candidate["opening_delivered_target_with_closing_measured_rate_first_events"]}/9个首次事件在已交付目标线性方向为打开时，实际距离速度仍闭合；54个近失败LP快照中{candidate["near_failure_lp_infeasible"]}个不可行。积压限制仍未解决所有惯性与共同可行性问题。'
    report += [r['candidate_diagnostic_note'],
               '该64窗口另计重复观测，不追加独立外部输入。54个LP快照中alpha/包络硬约束冲突为0，但15个安全行共同可行性失败；这与原baseline的234快照来自不同失败集合，不能直接按比例断言哪个求解器更差。',
               '\n## 登记文件审计\n',
               '附加取证第一次启动在仿真前发现旧PLAN.md说明文件hash与00:13登记不符（文件00:15修订），执行0窗口。原登记、失败manifest和双方hash保留于REGISTRATION_AUDIT.json；v2显式登记当前说明文件。所有取证执行代码、生产源码与actor hash均匹配，原重播物理结果逐帧相同；开发/留出冻结说明与参数未变。不能声称旧诊断说明文件从登记后一直未修改。']
    report+=['\n## 契约与限制\n',
             '- 生产源码279项、actor、原配置保持冻结。候选仅在独立评测wrapper生效，未推广。',
             '- actual actuator_target逐帧等于controller_target延后6步；目标软限位、真实增量、包络可达性用独立解析oracle核验。包络不可达比例和最大目标积压保留在逐条件results.json。',
             '- 4项CPU契约覆盖积压累积、速率回收、500组独立标量边界oracle和非法状态；旧积分行为下积压测试RED，新机制GREEN。不是机械臂安全证明。',
             '- 64窗口因果重播和128开发窗口单列；768留出配对窗口来自192组外部随机轨迹，不能相加声称896条独立轨迹。',
             '- 没有查看留出结果后调参、重筛初态、排除失败或改变终点。范围与运动量按全部帧报告；UR周期姿态、26维联合操作空间、手指随机、连续网格/力学接触及真实物体任务未被此压力评测充分证明。',
             '- 本批常规留出trace保存物理结果与目标契约；solver残差的精确因果审计仅在旧60317411重播保存，不冒称新留出每行都有该诊断。',
             '- 所有旧safety_*字段逐位保留；不会替换旧安全和不安全曲线或视频。']
    if len(r['aggregate_rows'])==4:
        fig,axes=plt.subplots(1,3,figsize=(15,4));names=['Raw','System 0','+box','+envelope']
        axes[0].bar(names,[x['violations']/x['windows']*100 for x in r['aggregate_rows']],color=list(COLORS.values()))
        axes[0].set(ylabel='Windows with any violation (%)',ylim=(0,100),title='Independent holdout: full16s')
        axes[1].bar(names,[100*x['mean_within_window_joint_range'] for x in r['aggregate_rows']],color=list(COLORS.values()))
        axes[1].set(ylabel='Mean within-window joint soft-limit range (%)',ylim=(0,100),title='Actual operating range')
        axes[2].bar(names,[100*x['four_arms_moving_fraction'] for x in r['aggregate_rows']],color=list(COLORS.values()))
        axes[2].set(ylabel='Four arms moving per pressure step (%)',ylim=(0,100),title='Movement >1mrad is not task success')
        fig.tight_layout();fig.savefig(HERE/'mechanism_effect.png',dpi=160);plt.close(fig)
        report+=['\n![安全表现与实际运动量](mechanism_effect.png)']
    (HERE/'REPORT.md').write_text('\n'.join(report)+'\n')
    (HERE/'panel_data.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    previous=json.loads(DATA.read_text());current=deepcopy(previous)
    current.update(safety_mechanism=r,summary=r['summary'],updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    link=dict(label='最新：积压机制因子对照与独立留出',url='https://asimfish.github.io/safeduo-dashboard/docs/'+HERE.name+'/REPORT.md')
    current['links']=[link]+[x for x in current['links'] if x['url']!=link['url']]
    for key,val in previous.items():
        if key not in ('safety_mechanism','summary','updated','links'):assert current[key]==val,key
    (args.preview_json or DATA).write_text(json.dumps(current,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    public=DASH/'docs'/HERE.name;public.mkdir(parents=True,exist_ok=True)
    allowed={'PLAN.md','HOLDOUT_PLAN.md','MECHANISM_PLAN.md','REPORT.md','results.json','panel_data.json','causal_results.json',
             'trace_plan.json','development_plan.json','holdout_plan.json','holdout_registration.json','mechanism_effect.png',
             'CANDIDATE_DIAGNOSIS_PLAN.md','VERIFICATION.md','candidate_trace_plan.json','candidate_trace_plan_v2.json',
             'candidate_causal_results.json','REGISTRATION_AUDIT.json'}
    for f in HERE.iterdir():
        if f.is_file() and (f.name in allowed or f.suffix=='.py'):
            shutil.copy2(f,public/f.name)
    for row in r['rows']:
        dest=public/row['phase']/row['id'];dest.mkdir(parents=True,exist_ok=True)
        for f in Path(row['path']).glob('*.json'):shutil.copy2(f,dest/f.name)
    for phase in ('development','holdout'):
        plan=json.loads((HERE/(phase+'_plan.json')).read_text())
        shutil.copy2(Path(plan['output_root'])/'campaign.json',public/(phase+'_campaign.json'))
    nas=Path('/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005')
    for name in ('trace_cells','candidate_trace_cells','candidate_trace_cells_v2'):
        dest=public/name;dest.mkdir(exist_ok=True)
        shutil.copy2(nas/name/'campaign.json',dest/'campaign.json')
        for cell in (nas/name).iterdir():
            if not cell.is_dir():continue
            child=dest/cell.name;child.mkdir(exist_ok=True)
            for f in cell.glob('*.json'):shutil.copy2(f,child/f.name)
    for seed in (152684921,198470327,237901613):
        dest=public/'banks'/str(seed);dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path('/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005/holdout_banks')/str(seed)/'metadata.json',dest/'metadata.json')
    print(json.dumps(dict(complete=r['completed_windows'],invalid=r['invalid_windows'],pending=r['pending_windows'],legacy_preserved=True)))


if __name__=='__main__':main()
