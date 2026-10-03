"""Prepare evidence and preview before activating new empirical dashboard data."""
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')
parser=argparse.ArgumentParser()
parser.add_argument('--preview-json',type=Path)
args=parser.parse_args()
r=json.loads((OUT/'results.json').read_text())
d=json.loads(DATA.read_text())
labels={'duo_env_a31_bounded_guard.yaml':'原限位策略',
        'duo_env_a31_latency_guard.yaml':'仅延长预测时间',
        'duo_env_a31_stored_guard.yaml':'完整积压预测',
        'duo_env_a31_pending_guard.yaml':'待执行目标预测'}
for row in r['rows']: row['display_candidate']=labels[row['candidate']]
r['rows'].sort(key=lambda row: list(reversed(labels)).index(row['candidate']))
new=[x for x in r['rows'] if x['campaign']=='pending_final_pilot']
old=[x for x in r['rows'] if x['campaign']=='pilot_initial_delay']
assert len(new)==len(old)==2
pairs=[]
for delay in [0,6]:
    a=next(x for x in old if x['delay_steps']==delay)
    b=next(x for x in new if x['delay_steps']==delay)
    pairs.append(f"{a['delay_ms']:.0f}ms延迟 {a['safe_initial_violations']}/{a['initially_safe']} → {b['safe_initial_violations']}/{b['initially_safe']}")
active=[x for x in r['rows'] if x['candidate']=='duo_env_a31_pending_guard.yaml' and x['method']!='raw']
nv=sum(x['safe_initial_violations'] for x in active);n=sum(x['initially_safe'] for x in active)
r['summary']='实验候选尚未通过安全回归。新增随机初态±0.1/±0.3rad及执行端0/100ms延迟；原失败条件匹配重播：'+'；'.join(pairs)+'。'
r['overview']='随机初态与执行延迟：'+'；'.join(pairs)
r['verdict']=(f"待执行目标预测共 {nv}/{n} 个初态无违规保护窗口发生违规；"
              '初始碰撞另列，包含解析兜底和System 0，分布及窗口相关，不给独立样本置信证明。'
              '仅延长预测时间未解决原失败；只看当前目标也未解决100ms延迟。预测保留最近6个已发目标，是局部线性风险模型，仍未覆盖全操作空间、动力学扰动及物体任务。')
nominal=next(x for x in r['rows'] if x['campaign']=='pending_nominal_l1')
r['verdict']+=(f"原初态漫游回归新增{nominal['safe_initial_violations']}/{nominal['initially_safe']}违规。"
               '候选未推广；原bounded_guard作为冻结参考保留。待确认安全指令是否能抢占队列或保护是否在执行器侧运行，再继续架构修改。')
r['candidate_status']='experimental_not_promoted'
r['protocol_note']='最新候选 duo_env_a31_pending_guard.yaml；actor、几何和动力学固定。seed12用于失败条件诊断与匹配重播，seed13用于预登记扰动矩阵。延迟为安全层之后的执行目标FIFO，控制周期约16.67ms；启动前核对真实求解器配置。无保护4个控制cell复用前一候选同命令/初态，不增加独立样本数。历史数据使用各自协议。'
r['traces']=[]
for cell,delay in [(1,0),(2,6)]:
    base=np.load(OUT/'pilot_initial_delay'/f'cell_{cell:03d}.npz')
    candidate=np.load(OUT/'pending_final_pilot'/f'cell_{cell:03d}.npz')
    np.testing.assert_array_equal(base['q_initial'],candidate['q_initial'])
    np.testing.assert_array_equal(base['cmd'],candidate['cmd'])
    x=base['official_margins'][:,31,2]*1000;y=candidate['official_margins'][:,31,2]*1000
    time=(np.arange(len(x))+1)*.016666
    group=next(row for row in new if row['delay_steps']==delay)
    r['traces'].append(dict(label=f"±0.3rad初态 · {delay*.016666*1000:.0f}ms执行延迟 · seed12/env31",
        unit='mm',time_s=time.tolist(),x_min=0,x_max=10,y_min=float(min(x.min(),y.min())-5),y_max=float(max(x.max(),y.max())+5),
        series=[dict(label='原限位策略',color='#dc2626',values=x.tolist()),
                dict(label='待执行目标预测',color='#15803d',values=y.tolist())],
        caption=f"相同初态和逐步指令；测量最小self_U球距，含本臂手部与连杆及U_L–U_R。原窗口最小{x.min():.2f}mm，新候选{y.min():.2f}mm；零线以下为球层重叠。本候选整个分组仍有{group['safe_initial_violations']}/{group['initially_safe']}违规，曲线不包括其他通道。示例固定为原零延迟失败env31，不增加独立样本数量。"))
base=np.load(OUT/'pilot_initial_delay/cell_002.npz')
candidate=np.load(OUT/'pending_final_pilot/cell_002.npz')
old_table=base['official_margins'][:,22,3]*1000
new_table=candidate['official_margins'][:,22,3]*1000
assert np.isfinite(old_table).all() and np.isfinite(new_table).all()
r['traces'].append(dict(label='尚未解决：100ms执行延迟 · 非豁免桌面距离 · env22',
    unit='mm',time_s=time.tolist(),x_min=0,x_max=10,
    y_min=float(min(old_table.min(),new_table.min())-5),y_max=float(max(old_table.max(),new_table.max())+5),
    series=[dict(label='原限位策略',color='#dc2626',values=old_table.tolist()),
            dict(label='待执行目标预测',color='#15803d',values=new_table.tolist())],
    caption=f"固定展示原延迟组持续失败env22：每帧最小非豁免桌面球距，允许最危险球发生变化。原策略{old_table.min():.2f}mm，新候选{new_table.min():.2f}mm，两者均发生球层违规；100ms分组仍11/32。该图不是实物接触力或单一球对轨迹。"))
r['default_trace']=2
fig,axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
for ax,trace,delay in zip(axes,r['traces'][:2],[0,100]):
    for s,label in zip(trace['series'],['Bounded guard','Pending-target prediction']):
        ax.plot(trace['time_s'],s['values'],color=s['color'],label=label,lw=1.6)
    ax.axhline(0,color='.3',ls='--');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set(xlabel='Simulation time (s)',ylabel='Measured self_U sphere distance (mm)',title=f'Initial jitter ±0.3 rad; actuator delay {delay} ms')
fig.savefig(OUT/'initial_latency_comparison.png',dpi=180)
fig.savefig(OUT/'initial_latency_comparison.svg');plt.close(fig)
d['safety_perturbation']=r
historical=d.get('historical_headline',[])
for card in d['headline']:
    if not card['label'].startswith('初态/延迟矩阵') and not any(x['label']==card['label'] for x in historical):
        historical.append({**card,'note':'原初态、零额外延迟；'+card['note']})
d['historical_headline']=historical
d['headline']=[dict(label='初态/延迟矩阵 · '+x['label'],episodes=x['initially_safe'],violations=x['safe_initial_violations'],
                  rate=x['safe_initial_violations']/x['initially_safe'],cp95_upper=None,
                  status='bad' if x['safe_initial_violations'] else 'warn',
                  note=f"seed13；±0.1/0.3rad×0/100ms；初态违规另列{x['initial_violations']}/{x['episodes']}")
               for x in r['matrix_headline']]
d['summary']=r['summary'];d['updated']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
d['safety_robustness']['summary']=d['safety_robustness']['summary'].replace('本轮','上一轮')
d['coverage_assessment']=dict(title='覆盖判定：新增局部初态与目标延迟，仍未穷尽空间',summary=r['verdict'],
    items=['初态采用独立随机流，与命令tape分离，逐窗口核对SHA；没有按碰撞筛选初态。',
           '随机动作控制26个臂关节，未随机手指；初态只在既定姿态附近±0.1/±0.3rad采样。',
           '执行目标延迟为0/6控制步，测量保持当前，不外推为传感延迟或网络丢包安全。',
           '原512窗口仍保留为原初态/零额外延迟证据，不与扰动窗口混算。'])
d['limitations']=r['limitations']+[x for x in d['limitations'] if x not in r['limitations']]
prefix='https://github.com/asimfish/safeduo-dashboard/blob/main/docs/safety_perturbation_20261003/'
d['links']=[{'label':'初态与延迟完整报告','url':prefix+'REPORT.md'},
            {'label':'预测失败与积压诊断','url':prefix+'DIAGNOSIS.md'},
            {'label':'初态与命令逐窗口配对CSV','url':prefix+'paired.csv'}]+[x for x in d['links'] if '/docs/safety_perturbation_20261003/' not in x['url']]
(args.preview_json or DATA).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
dest=DASH/'docs/safety_perturbation_20261003';dest.mkdir(parents=True,exist_ok=True)
for name in ['PLAN.md','DIAGNOSIS.md','REPORT.md','results.json','paired.csv','build_report.py','prepare_dashboard.py',
             'time_only_results.json','time_only_REPORT.md','stored_only_results.json','stored_only_REPORT.md',
             'backlog_prediction_red.log','perturbation_test_red.log','pending_prediction_red.log',
             'pending_credit_counterfactual_red.log','run_nominal_regression.py',
             'delivery_tests.log','initial_latency_comparison.png','initial_latency_comparison.svg']:
    shutil.copy2(OUT/name,dest/name)
for manifest in r['manifests']:
    name=Path(manifest['args']['out']).name
    target=dest/'protocols';target.mkdir(exist_ok=True)
    shutil.copy2(OUT/name/'protocol.json',target/f'{name}.json')
for name in ['stored_candidate_pilot','stored_candidate_matrix','pending_verified_pilot','pending_verified_matrix']:
    target=dest/'invalid_runs'/name;target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUT/name/'protocol.json',target/'protocol.json')
    shutil.copy2(OUT/name/'INVALIDATION.json',target/'INVALIDATION.json')
target=dest/'protocols'
shutil.copy2(OUT/'internal_motion_diagnostic/protocol.json',target/'internal_motion_diagnostic.json')
for src in ['src/safeduo/safety/backstop.py','src/safeduo/envs/duo_env.py','src/safeduo/eval/research_battery.py',
            'src/safeduo/eval/perturbations.py','src/safeduo/safety/target_history.py',
            'src/safeduo/configs/duo_env_a31_latency_guard.yaml','src/safeduo/configs/duo_env_a31_pending_guard.yaml',
            'src/safeduo/configs/duo_env_a31_stored_guard.yaml','tests/test_backlog_prediction.py','tests/test_pending_target_prediction.py',
            'tests/test_eval_perturbations.py','tests/test_research_battery.py']:
    shutil.copy2(ROOT/src,dest/Path(src).name)
shutil.copytree(OUT/'source_before',dest/'source_before',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(OUT/'source_stored',dest/'source_stored',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(OUT/'source_pending_credit',dest/'source_pending_credit',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for optional in ['check_browser.py','browser_check.log','live_browser_check.log','VALIDATION.md','consistency_check.json','panel_results.png']:
    if (OUT/optional).exists(): shutil.copy2(OUT/optional,dest/optional)
for file in dest.rglob('*'):
    if file.suffix in {'.svg','.log'}:
        file.write_text('\n'.join(line.rstrip() for line in file.read_text().splitlines())+'\n')
print('Prepared perturbation evidence and',args.preview_json or DATA)
