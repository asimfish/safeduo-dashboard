"""Build complete factor evidence before activating a new dashboard release."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np


OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=OUT.parent/'safety_perturbation_20261003'
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')
parser=argparse.ArgumentParser();parser.add_argument('--preview-json',type=Path);args=parser.parse_args()
source=json.loads((OUT/'observer_manifest.json').read_text())
for path,sha in source['core_sources'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
diagnosis=json.loads((OUT/'analysis.json').read_text())
audit=json.loads((OUT/'factor_solver_audit.json').read_text())
datasets={};manifests={};records=[]


def load_run(name):
    path=OUT/name
    p=json.loads((path/'protocol.json').read_text())
    assert p['status']=='complete' and p['completed_cells']==len(p['design'])
    assert p['checkpoint_sha256']==source['checkpoint_sha256']
    assert all(p['source_sha256'][k]==v for k,v in source['core_sources'].items())
    manifests[name]=p
    if name not in ['delayed','nominal_table']:
        f=json.loads((path/'factor_manifest.json').read_text())
        assert f['original_actor_observation_rows']==32
        assert f['backlog_aware']==p['effective_backstop']['backlog_aware']
        for filename,sha in f['source_sha256'].items():
            assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest()==sha
    ep=json.loads((path/'episodes.json').read_text())
    records.extend(dict(run=name,**e) for e in ep)
    datasets[name]=[np.load(path/f'cell_{i+1:03d}.npz') for i in range(p['completed_cells'])]
    for z in datasets[name]:
        meta=json.loads(str(z['meta_json']));k=meta['actuator_delay_steps']
        if k:
            np.testing.assert_array_equal(z['actuator_target'][:k],
                np.repeat(z['q_initial'][None],k,axis=0))
            np.testing.assert_array_equal(z['actuator_target'][k:],z['controller_target'][:-k])
    return ep


def summarize(ep):
    safe=[e for e in ep if not e['initial_violation']]
    return dict(episodes=len(ep),initial_violations=len(ep)-len(safe),initially_safe=len(safe),
        violations=sum(e['violation'] for e in safe),damaging=sum(e['damaging'] for e in safe),
        by_class={c:sum(e['violation_by_class'][c] for e in safe) for c in ['cross','self_F','self_U','table']},
        mean_joint_range=float(np.mean([e['joint_range_fraction_mean'] for e in safe])),
        mean_joint_path_rad=float(np.mean([e['measured_joint_path_rad'] for e in safe])),
        min_nonexempt_mm={c:min(e['min_nonexempt_margin_m'][c] for e in safe)*1000
                         for c in ['cross','self_F','self_U','table']},
        bad_envs=[e['env_id'] for e in safe if e['violation']])


baseline={key:load_run(name) for key,name in [('nominal','nominal_table'),('delayed','delayed')]}
labels={'baseline':'原历史距离预测','debit':'启用闭合扣减','critical':'保留临界行','both':'二者联合'}
rows=[dict(factor='baseline',label=labels['baseline'],nominal=summarize(baseline['nominal']),
           delayed=summarize(baseline['delayed']),rule='原预测32行')]
for factor in ['debit','critical','both']:
    row=dict(factor=factor,label=labels[factor],rule='预测行 + 全部临界行，容量128' if factor!='debit' else '原预测32行')
    for scenario in ['nominal','delayed']:
        name=f'{scenario}_{factor}';ep=load_run(name);row[scenario]=summarize(ep)
        base=datasets['nominal_table' if scenario=='nominal' else 'delayed'][0]
        z=datasets[name][0]
        np.testing.assert_array_equal(base['q_initial'],z['q_initial'])
        if scenario=='delayed': np.testing.assert_array_equal(base['cmd'],z['cmd'])
        assert manifests[name]['effective_backstop']['backlog_aware']==(factor in ['debit','both'])
        a=deepcopy(manifests[name]['resolved_config']);b=deepcopy(manifests['delayed']['resolved_config'])
        a['safety']['backstop'].pop('backlog_aware',None);b['safety']['backstop'].pop('backlog_aware',None)
        assert a==b
    rows.append(row)
holdout=load_run('critical_holdout')
holdout_rows=[]
for i in range(4):
    ep=[e for e in holdout if e['cell_id']==i+1];m=ep[0]
    holdout_rows.append(dict(seed=m['seed'],jitter_rad=m['initial_jitter_rad'],
        delay_ms=m['actuator_delay_steps']*m['dt']*1000,**summarize(ep)))
assert len(holdout_rows)==4 and all(x['seed']==14 for x in holdout_rows)
critical=next(x for x in rows if x['factor']=='critical')
hold=summarize(holdout)
candidate_ep=[e for e in records if e['run'] in ['nominal_critical','delayed_critical','critical_holdout']]
candidate=summarize(candidate_ep)
old_bad=set(rows[0]['delayed']['bad_envs']);new_bad=set(critical['delayed']['bad_envs'])
changes=dict(new_failures=sorted(new_bad-old_bad),corrected_failures=sorted(old_bad-new_bad))
all_factor=[e for e in records if e['run'] not in ['nominal_table','delayed']]
assert len(all_factor)==320
result=dict(schema='safeduo.safety_forensics.v1',rows=rows,holdout_rows=holdout_rows,
    holdout=hold,candidate_total=candidate,candidate_status='experimental_not_promoted',
    factor_windows=320,count_as_independent_windows=False,pilot_failure_comparison=changes,
    diagnosis=diagnosis,solver_audit=audit,source=source,
    factors={k:json.loads((OUT/k/'factor_manifest.json').read_text())
             for k in manifests if k not in ['nominal_table','delayed']})
result['summary']=(f"临界行保留：原初态违规 {rows[0]['nominal']['violations']}/32 → {critical['nominal']['violations']}/32；"
                   f"100ms长时重播 {rows[0]['delayed']['violations']}/32 → {critical['delayed']['violations']}/32。"
                   f"新种子矩阵 {hold['violations']}/{hold['initially_safe']}。")
result['verdict']=(f"仅保留临界行候选共 {candidate['violations']}/{candidate['initially_safe']} 个初态无违规窗口发生违规，"
    '仍未通过安全评测、未推广为默认策略。只启用闭合扣减使原初态1/32恶化为15/32，联合版本同样15/32，不能直接启用。'
    '本轮改善来自解析约束保留，actor参数未变，不归因于学习头。严格FIFO的延迟失效仍存在，后续需要动力学与队列预测；执行器侧保护或停机抢占仍需真实链路支持。')
result['coverage_note']=(f"seed14预登记，±0.1/±0.3rad × 0/100ms，保持0.5秒，5秒窗口，共{hold['episodes']}。"
    '初态碰撞不筛选；本轮没有seed14原候选或无保护控制，不能称配对改善率。局部初态和相关窗口，不是全空间或实机安全证明。')
result['protocol_note']='actor固定a31b；输入观测原32行。临界行并集只进入解析层，d ≤ 当前d_min+10mm，容量128；溢出中止。全部输出仍经过相同目标FIFO；控制周期16.67ms末采样。漫游是状态反馈源，命令可随方法变化。完整协议与源代码见报告。'
result['runtime_correction']='上一轮运行协议backlog_aware=false：历史距离预测已启用，单步历史闭合量扣减未启用。旧CPU反事实测试使用true，仅验证该分支，不能证明旧仿真已执行；原计数和球距保持不变。'
result['traces']=[]


def add_trace(label,series,unit,caption):
    values=np.stack([v for _,_,v in series]);assert np.isfinite(values).all()
    time=(np.arange(values.shape[-1])+1)*.016666
    low=float(values.min());high=float(values.max());pad=max((high-low)*.08,.1)
    result['traces'].append(dict(label=label,unit=unit,time_s=time.tolist(),x_min=0,
        x_max=float(time[-1]),y_min=low-pad,y_max=high+pad,
        series=[dict(label=n,color=c,values=v.tolist()) for n,c,v in series],caption=caption))


base=datasets['nominal_table'][0];new=datasets['nominal_critical'][0]
add_trace('已知桌面回归：原初态env9 · 非豁免桌面距离',
    [('原历史距离预测','#dc2626',base['official_margins'][:,9,3]*1000),
     ('保留临界行','#15803d',new['official_margins'][:,9,3]*1000)],'mm',
    '原轨迹精确重播，env9桌面最深重叠4.325mm；本候选原初态分组0/32。曲线为每帧最小非豁免桌面球距，最危险球可能改变；条件接触仍按原语义处理，不是接触力测量。')
pid=diagnosis['runs'][1]['events'][0]['pair_id']
old_selected=(base['pre_row_id'][:,9]==pid).any(-1).astype(float)
new_selected=(new['pre_row_id'][:,9]==pid).any(-1).astype(float)
add_trace('临界桌面球的约束是否保留 · 原初态env9',
    [('原32行集','#dc2626',old_selected),('临界行并集','#15803d',new_selected)],'0/1',
    '固定U_R/hand/right_little_2与table[1]，pair8996。原step253–257的32行全为cross，该球缺席；step257开始前间距仅2.318mm且豁免已取消，step258重入时已重叠。新候选保留全部达到临界阈值的行；状态随策略改变，选择曲线不表示相同物理轨迹。')
base=datasets['delayed'][0];new=datasets['delayed_critical'][0]
margin=new['official_margins'].min(-1);env=int(np.unravel_index(np.argmin(margin),margin.shape)[1])
add_trace(f'仍未解决：100ms延迟 · 候选最深违规env{env}',
    [('原历史距离预测','#dc2626',base['official_margins'][:,env].min(-1)*1000),
     ('保留临界行','#15803d',margin[:,env]*1000)],'mm',
    f"相同26关节初态和完整纯随机命令；原候选11/32，新候选{critical['delayed']['violations']}/32。固定展示新候选最深违规环境，每帧最小官方非豁免通道球距，球对/通道可改变。原延迟取证重播的物理轨迹与旧发布结果不同，不冒充逐位精确重播。")
result['default_trace']=2
(OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
with (OUT/'episodes.csv').open('w') as f:
    cols=['run','cell_id','env_id','seed','flow','initial_jitter_rad','actuator_delay_steps',
          'initial_q_sha256','command_sha256','initial_violation','violation','damaging',
          'measured_joint_path_rad','joint_range_fraction_mean']
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(records)
lines=['# 临界约束保留与单因素复测','',result['summary'],'',result['verdict'],'',
    '## 运行配置勘误','',result['runtime_correction'],'',
    '|因素|原初态5s违规|100ms/±0.3rad 10s违规|', '|---|---:|---:|']
for row in rows: lines.append(f"|{row['label']}|{row['nominal']['violations']}/32|{row['delayed']['violations']}/32|")
lines += ['', '## 未参与诊断的seed14', '', result['coverage_note'],'',
          '|初态±rad|延迟ms|初态违规|无违规初态后的违规|','|---:|---:|---:|---:|']
for row in holdout_rows:
    lines.append(f"|{row['jitter_rad']}|{row['delay_ms']:.0f}|{row['initial_violations']}/{row['episodes']}|{row['violations']}/{row['initially_safe']}|")
lines += ['', '## 证据边界','',
    '- 98项相关CPU回归通过；旧配置与不保留临界行的反事实均产生RED。不能用CPU通过替代仿真违规结果。',
    '- 原初态取证q/exec/命令/官方球距逐位一致；延迟取证仅初态与命令逐位一致，物理轨迹改变。旧计数11/32、新取证同样11/32且违规环境相同；不偷换成精确轨迹。',
    '- 原取证LP仅判断自碰/桌面子集，未记录p，不声称全约束可行。因素探针记录p后按真实cross预算、alpha、历史扣减、authority和限位重建完整约束，独立LP近失效抽查见factor_solver_audit.json。全部残差重算与记录差<2µm；线性可行不能证明动力学安全。',
    '- 纯随机保持组共享初态与命令；漫游是状态反馈源，策略改变后命令不同。actor32行观测/权重、几何和动力学不变；新增行只进入解析层，保护也等待同一FIFO。',
    '- 六个因素探针192窗口加新种子128窗口是分组相关实验，不能作为320个独立安全样本；取证64窗口不增加样本。仅critical候选192窗口单列，不与旧512、384窗口混算。',
    '- 本轮无新视频、物体任务或接触力证据；控制周期末球距采样未排除周期内短暂碰撞。',
    '', '原延迟重播候选新增/修复环境：`'+json.dumps(changes)+'`。',
    '', '源码core和actor hash冻结；评测适配器变更单列factor_manifest；全部协议、逐窗口CSV和源文件随报告发布。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
data=json.loads(DATA.read_text());data['safety_forensics']=result
data['safety_perturbation']['runtime_correction']=result['runtime_correction']
data['safety_perturbation']['summary']=data['safety_perturbation']['summary'].split(' 配置勘误：')[0]+' 配置勘误：该轮启用历史距离预测，未启用闭合量扣减；数值结果不变。'
historical=data.setdefault('historical_headline',[])
for card in data['headline']:
    if not any(x['label']==card['label'] for x in historical): historical.append(card)
data['headline']=[dict(label='延迟重播 · '+x['label'],episodes=32,violations=x['delayed']['violations'],
    rate=x['delayed']['violations']/32,cp95_upper=None,status='bad' if x['delayed']['violations'] else 'warn',
    note='System 0；seed12；±0.3rad / 100ms；10秒，相同命令tape') for x in rows]
data['summary']=result['summary'];data['updated']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
data['coverage_assessment']=dict(title='覆盖判定：增加临界行保留与未参与诊断的新种子',
    summary=result['verdict'],items=[result['coverage_note'],'未验证全空间、随机手指、质量/摩擦、丢包、感知延迟和实机。'])
prefix='https://github.com/asimfish/safeduo-dashboard/blob/main/docs/safety_latency_forensics_20261003/'
data['links']=[dict(label='临界行与单因素完整报告',url=prefix+'REPORT.md'),
               dict(label='预登记与证据边界',url=prefix+'PLAN.md'),
               dict(label='完整约束独立LP核查',url=prefix+'factor_solver_audit.json'),
               dict(label='逐窗口复现CSV',url=prefix+'episodes.csv')]+[x for x in data['links'] if '/safety_latency_forensics_20261003/' not in x['url']]
(args.preview_json or DATA).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
dest=DASH/'docs/safety_latency_forensics_20261003';dest.mkdir(parents=True,exist_ok=True)
for name in ['PLAN.md','REPORT.md','results.json','analysis.json','analysis.log','factor_solver_audit.json',
             'factor_solver_audit.log','episodes.csv','observer_manifest.json','critical_rows.py',
             'trace_runner.py','counterfactual_runner.py','test_candidate_contracts.py','analyze.py',
             'audit_factor_solver.py','run_probes.py','run_factors.py','run_holdout.py','prepare_panel.py',
             'contract_suite.log','candidate_contracts_red.log','candidate_contracts_green.log',
             'profile_debit_counterfactual_red.log','row_union_counterfactual_red.log']:
    shutil.copy2(OUT/name,dest/name)
shutil.copy2(ROOT/'src/safeduo/configs/duo_env_a31_pending_debit_guard.yaml',dest/'duo_env_a31_pending_debit_guard.yaml')
pd=dest/'protocols';pd.mkdir(exist_ok=True)
for name in manifests:
    shutil.copy2(OUT/name/'protocol.json',pd/f'{name}.json')
    if (OUT/name/'factor_manifest.json').exists():
        shutil.copy2(OUT/name/'factor_manifest.json',pd/f'{name}_factor.json')
for name in ['check_browser.py','browser_check.log','live_browser_check.log','VALIDATION.md','panel.png','DELIVERY.json']:
    if (OUT/name).exists():shutil.copy2(OUT/name,dest/name)
for p in dest.rglob('*.log'):
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
print('Prepared complete factor evidence and',args.preview_json or DATA)
