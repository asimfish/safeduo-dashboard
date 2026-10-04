"""New risk results coexist with every previous scientific experiment."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='safeduo-risk-strata-20261004'
import matplotlib.pyplot as plt
import numpy as np
from analyze_risk import load

HERE=Path(__file__).resolve().parent
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')
LABELS={'raw':'无安全保护','system0':'System 0动态完整行候选'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview-json',type=Path);args=parser.parse_args()
    r=json.loads((HERE/'results.json').read_text())
    r['summary']=f'六组臂对定额压力测试：已完成{r["completed_windows"]}/384个16秒配对窗口，无效{r["invalid_windows"]}，待完成{r["pending_windows"]}。'
    r['protocol_note']='三个新种子60317411/80692357/109441003；每种子六组臂对各8个风险初态＋16个广域稳定初态。64环境，前60步零输入＋900步四臂独立方向、异步保持；幅度0.005/0.015/0.025/0.05rad，保持1/4/15/30/90/180步。每条件独立进程，严格6步FIFO（100ms），actor32行与权重冻结，保护器动态完整临界行硬预算512。全部16秒作为安全分母，不删除初态准备后重现的失败。'
    audit=r['bank_audit']
    r['bank_note']=f'初态准入：所有保留桌面球距和非豁免机器人球距≥1mm，再经1秒零输入检查，违规为0、漂移≤0.05rad、末步速度≤0.1rad/s。{audit["sampling"]["geometry_candidates"]:,}个几何候选、{audit["sampling"]["settling_candidates"]:,}个原地检查记录均保存，192个初态的选择引用全部核验。候选数不计入闭环样本；只筛初态和零输入稳定性，未读取后续随机策略结果。'
    r['verdict']='风险初态按20–60mm臂对球面余量定向搜索，是条件压力分布；广域组从95%关节跨度LHS条件筛选。它们与旧纯随机组分别报告，不把初态准备改善当作安全策略提升。六臂对实际80mm暴露仍须逐组核验；风险探针包含豁免球对，官方安全结论按四类非豁免余量。保留所有失败，仍有违规就不能宣布System 0安全通过。仅证明已测控制周期末球距表现，不证明26维联合空间、连续碰撞、物体接触或实机安全。'
    r['aggregate_rows']=[{**x,'method_label':LABELS[x['method']]} for x in r['strata'] if x['stratum']<0]
    r['pair_rows']=[{**x,'method_label':LABELS[x['method']]} for x in r['strata'] if x['stratum']>=0]
    r['coverage_cases']=[]
    for g in r['group_coverage']:
        c=g['coverage'];same=[x for x in r['rows'] if x['method']==g['method'] and x['status']=='complete']
        simultaneous=100*np.mean([x['four_arms_moving_fraction'] for x in same])
        execution=np.mean([x['exec_command_l2_ratio'] for x in same])
        r['coverage_cases'].append(dict(
            label=g['label']+' · '+LABELS[g['method']],method=g['method'],stratum=g['stratum'],
            initial=c['initial_joint'],visited=c['visited_joint'],ee_initial=c['initial_ee'],ee_visited=c['visited_ee'],pairs=g['pairs'],
            note=f'{c["episodes"]}个窗口的全部16秒实测，包含全部失败；平均单窗口关节跨度{100*c["mean_within_window_joint_range"]:.2f}%，总关节路径{c["mean_joint_path_rad"]:.2f}rad。同方法全组压力阶段四臂每步同时移动>1mrad比例{simultaneous:.1f}%，输出/外部命令L2总量比{execution:.3f}。橙框为初态，蓝色为实测边际访问；10cm末端格按跨种子并集，不表示可达体积覆盖率。'))
    r['default_coverage']=next((i for i,x in enumerate(r['coverage_cases']) if x['method']=='system0' and x['stratum']==-2),0)
    r['traces']=[]
    for s in [x for x in r['rows'] if x['method']=='system0' and x['status']=='complete']:
        raw=next(x for x in r['rows'] if x['method']=='raw' and x['seed']==s['seed'])
        if raw['status']!='complete':continue
        da,db=load(Path(raw['path'])/'cell_001.npz'),load(Path(s['path'])/'cell_001.npz')
        labels=np.array(s['risk_pair_index']);bad_a=(da['official_margins']<0).any((0,2));bad_b=(db['official_margins']<0).any((0,2))
        for group in [x for x in r['strata'] if x['method']=='system0' and x['stratum']>=-1]:
            candidates=np.flatnonzero(labels==group['stratum'])
            failures=candidates[bad_b[candidates]];protected_pass=candidates[bad_a[candidates]&~bad_b[candidates]]
            examples=[]
            if len(protected_pass):examples.append(('保护全窗0违规',int(protected_pass[0])))
            if len(failures):examples.append(('保护仍失败',int(failures[0])))
            if not examples:examples.append(('双方全窗0违规',int(candidates[0])))
            for outcome,e in examples:
                idx=np.arange(0,960,3);v=np.stack([da['official_margins'][idx,e].min(-1),db['official_margins'][idx,e].min(-1)])*1000
                lo,hi=float(v.min()),float(v.max());pad=max(1.,(hi-lo)*.08)
                r['traces'].append(dict(label=f'{outcome} · {group["label"]} · seed{s["seed"]} · env{e}',outcome=outcome,
                    raw_window_violated=bool(bad_a[e]),protected_window_violated=bool(bad_b[e]),unit='mm',time_s=((idx+1)*r['dt']).tolist(),
                    x_min=0,x_max=16,y_min=lo-pad,y_max=hi+pad,
                    events=[dict(time_s=1.,label='随机输入开始',color='#6b7280'),dict(time_s=1.1,label='首条随机目标到达',color='#2563eb')],
                    series=[dict(label='无安全保护',color='#dc2626',values=v[0].tolist()),dict(label='System 0候选',color='#7c3aed',values=v[1].tolist())],
                    caption='每个种子/初态分层分别展示最低编号的“raw失败、保护全窗0违规”和“保护仍失败”；缺少某类则不造样例。曲线只说明这些窗口，总体结果包含全部失败。双方初态和960步外部输入逐位一致；保护器可能在零输入阶段产生运动，实际q前缀一致性另列。图每3帧取点，违规统计用全部960帧。纵轴为四类官方非豁免球距最小值，负值是球层违规，不是网格接触力。'))
    r['default_trace']=0
    (HERE/'panel_data.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    reports=['# 六组臂对风险与初态稳定性实验\n',r['summary']+'\n',r['protocol_note']+'\n',r['bank_note']+'\n',r['verdict']+'\n',
             '\n|初态分层|方法|完整窗口|违规|>5mm|零输入前缀违规|随机目标到达前违规|\n|---|---|---:|---:|---:|---:|---:|']
    for x in r['aggregate_rows']:reports.append(f'|{x["label"]}|{x["method_label"]}|{x["windows"]}/{x["planned_windows"]}|{x["violations"]}|{x["deep"]}|{x["zero_prefix_violations"]}|{x["before_first_random_delivery"]}|')
    reports+=['\n|风险初态臂对|方法|完整窗口|3帧80mm：全窗/随机目标到达后|违规|>5mm|\n|---|---|---:|---:|---:|---:|']
    for x in r['pair_rows']:reports.append(f'|{x["label"]}|{x["method_label"]}|{x["windows"]}/24|{x["target_pair_exposure"]["exposed_80mm"]}/{x["target_pair_pressure_exposure"]["exposed_80mm"]}|{x["violations"]}|{x["deep"]}|')
    reports+=['\n全窗80mm暴露是预登记口径，另补充首次随机目标到达后（第66步后）的同阈值暴露，避免只靠初态/零输入阶段声称随机压力风险覆盖。该补充统计不影响初态选择、主安全分母或已登记暴露判定。']
    reports+=['\n## 零输入诊断（旧分布，独立于新增384窗口）\n',
              '三个旧初态库共192窗口，26关节零输入且目标恒定，20窗口仍违规；19初态含条件豁免桌面球层重叠，14窗口负桌面余量的豁免撤回。零时钟刷新q、球心、flag逐位一致，未证实简单缓存刷新可以解决。原amp=0 CLI在启动仿真前被拒绝，三个失败日志保留，计0诊断窗口；v2 nominal amp=.05但真实tape全零，完整192窗口另计。',
              '\n## 核验与边界\n',
              '- 原地准入属于实验初始化；生产安全核心、actor、配置保持冻结，动态512是未推广的评测候选。',
              '- 每个风险重启最多取一个姿态；全部候选、拒绝、稳定性结果和选择引用保留。不同种子共享同一模拟场景与模型，不报IID置信界。',
              '- 命令、初态、FIFO、源hash和解析配置核验；实际越普通求解器增量箱的次数保留在逐条件results，不假称所有指令被限幅。',
              '- 两项新增CPU契约验证零前缀保留全部随机后缀、配额不足/策略结果筛选拒绝；不把它们当安全证明。上一轮19项契约不重复计数。',
              '- 后续失败的窗口全部留在主16秒分母，包含零输入和延迟前缀；官方违规与包含豁免球对的风险探针分开。',
              '- 旧1536完整／576容量无效窗口及全部旧面板字段保持原样。新增192不同指令窗口对应384相关配对窗口，零输入诊断和bank候选分开。',
              '- 原始NPZ与完整候选在NAS，网页公开协议/源hash/完整episode JSON/CSV/标准图件。尚未做连续时间、物体感知/抓持和实机验收。',
              '- 距离曲线在每种子/分层中分别取最低环境编号的raw失败保护通过、保护仍失败，公开选择规则；它是结果分类示例，不是预登记的额外安全检验，不影响完整分母。',
              '\n[完整结果](results.json) · [逐条件CSV](windows.csv) · [预登记](RISK_PLAN.md) · [零输入结果](zero_results.json)\n']
    if r['coverage_cases']:
        c=r['coverage_cases'][r['default_coverage']];counts=np.array(c['visited']['counts']);initial=np.array(c['initial']['counts'])
        fig,ax=plt.subplots(figsize=(10,7));im=ax.imshow(np.log1p(counts),aspect='auto',cmap='Blues')
        for j,i in zip(*np.where(initial>0)):ax.add_patch(plt.Rectangle((i-.5,j-.5),1,1,fill=False,edgecolor='#d97706',linewidth=.6))
        labels=[f'{a} j{j+1}' for a,n in zip(['F_L','F_R','U_L','U_R'],[7,7,6,6]) for j in range(n)]
        ax.set_yticks(range(26),labels);ax.set_xticks(range(10),[f'{i*10}-{(i+1)*10}%' for i in range(10)])
        ax.set(xlabel='Normalized soft-limit marginal bins',title='Stable risk bank: all measured frames, including failures')
        fig.colorbar(im,ax=ax,label='log(1 + measured samples)');fig.tight_layout();fig.savefig(HERE/'joint_coverage.png',dpi=160);fig.savefig(HERE/'joint_coverage.svg',metadata={'Date':None});plt.close(fig)
        reports+=['![实测关节边际覆盖](joint_coverage.png)']
    reports+=['\n## 配对外部输入与实际观测前缀\n',
              '初态和外部命令一致不保证实际q前缀一致：保护器可能对零输入也产生避险增量。这里保存随机首目标到达前66步的真实差值，不冒称双方在随机压力开始时状态相同。\n',
              '|种子|初态/命令逐位一致|66步q前缀逐位一致|最大q差(rad)|raw单独失败|保护单独失败|双方失败|\n|---|---|---|---:|---:|---:|---:|']
    for p in r['paired_inputs']:
        if 'pre_first_random_delivery_q_exact' not in p:continue
        reports.append(f'|{p["raw"]}|是|{p["pre_first_random_delivery_q_exact"]}|{p["pre_first_random_delivery_q_max_diff_rad"]:.6f}|{p["raw_only_failure"]}|{p["protected_only_failure"]}|{p["both_failure"]}|')
    reports+=['\n## 实际执行量与强随机输入\n',
              '|条件|压力阶段四臂同时动(>1mrad)|输出/命令L2总量比|零输入仍有保护输出的窗口|零前缀最大q漂移(rad)|四类违规 cross/F/U/table|\n|---|---:|---:|---:|---:|---|']
    for x in r['rows']:
        if x['status']!='complete':continue
        reports.append(f'|{x["id"]}|{100*x["four_arms_moving_fraction"]:.2f}%|{x["exec_command_l2_ratio"]:.4f}|{x["zero_input_output_nonzero_windows"]}|{x["zero_prefix_max_joint_drift_rad"]:.6f}|{x["class_violations"]}|')
    reports+=['\n按全部900步实际外部cmd报告26关节逐关节最小/最大、逐窗口双符号比例、保持时长与幅度档更新次数，见results.json；这些是边际输入统计，不冒称覆盖26维联合操作空间。风险初态搜索改变初态分布，后续指令保持状态独立随机。各四类违规可重叠，不能相加当唯一失败数。']
    (HERE/'REPORT.md').write_text('\n'.join(reports)+'\n')
    previous=json.loads(DATA.read_text());current=deepcopy(previous)
    current.update(safety_risk_strata=r,summary=r['summary'],updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    link=dict(label='最新：六臂对风险与稳定初态实验',url='https://asimfish.github.io/safeduo-dashboard/docs/'+HERE.name+'/REPORT.md')
    current['links']=[link]+[x for x in current['links'] if x['url']!=link['url']]
    for key,value in previous.items():
        if key not in ['safety_risk_strata','summary','updated','links']:assert current[key]==value,key
    (args.preview_json or DATA).write_text(json.dumps(current,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    public=DASH/'docs'/HERE.name;public.mkdir(parents=True,exist_ok=True)
    for f in [*HERE.glob('*.py'),*HERE.glob('*.md'),*HERE.glob('*.json'),*HERE.glob('*.csv'),*HERE.glob('*.png'),*HERE.glob('*.svg')]:
        if f.name in ['preview_scientific.json','previous_scientific.json','DELIVERY.json','NEXT.md','release_state.json']:continue
        if f.suffix=='.svg':(public/f.name).write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n')
        else:shutil.copy2(f,public/f.name)
    root=Path('/mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004')
    for row in r['rows']:
        dest=public/'cells'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        for f in Path(row['path']).glob('*.json'):shutil.copy2(f,dest/f.name)
    for seed in [60317411,80692357,109441003]:
        dest=public/'banks'/str(seed);dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'risk_banks'/str(seed)/'metadata.json',dest/'metadata.json')
    # Campaign and zero-diagnostic protocols retain failures; never upload bulk NPZ.
    for sub in ['cells','zero_cells_v2']:
        dest=public/sub;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/sub/'campaign.json',dest/'campaign.json')
    for cell in (root/'zero_cells_v2').iterdir():
        if not cell.is_dir():continue
        dest=public/'zero_cells_v2'/cell.name;dest.mkdir(exist_ok=True)
        for f in cell.glob('*.json'):shutil.copy2(f,dest/f.name)
    print(json.dumps(dict(complete=r['completed_windows'],invalid=r['invalid_windows'],pending=r['pending_windows'],legacy_preserved=True)))

if __name__=='__main__':main()
