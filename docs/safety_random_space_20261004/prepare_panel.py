"""Publish measured random coverage with invalid and unfinished cells separate."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='safeduo-random-space-20261004'
import matplotlib.pyplot as plt
import numpy as np

from analyze import load

OUT=Path(__file__).resolve().parent
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')
FLOW_LABELS={'wide_iid':'±1.2rad初态 + 逐步独立噪声','wide_burst':'±1.2rad初态 + 四臂异步保持',
             'global_burst':'95%关节跨度初态（未筛选）+ 异步保持','feasible_burst':'95%跨度初态条件分布 + 异步保持'}
METHOD_LABELS={'raw':'无安全保护','original128':'原固定128行 System 0','dynamic512':'动态行 System 0（实验候选）'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview-json',type=Path);args=parser.parse_args()
    r=json.loads((OUT/'results.json').read_text())
    r['summary']=(f'扩大随机范围：已完成{r["completed_windows"]}个15秒窗口，容量失败{r["invalid_windows"]}个，'
                  f'待完成{r["pending_windows"]}个。完成、无效和待完成分别统计，动态行候选尚未推广。')
    r['protocol_note']=(
        '64环境×15秒/条件；四臂26关节独立方向；原三种子731923/2048171/9987031，新条件分布种子13447771/27180353/48921161。'
        '逐步IID指令或各臂独立保持1/4/15/30/90/180步；幅度档0.005/0.015/0.025/0.05rad。'
        '原初态扰动从±0.3扩大至±1.2rad，另采样归一化软限位[2.5%,97.5%]的随机LHS。'
        '每条件独立新进程，严格6步（100ms）FIFO；已完成配对的初态、初态标志和完整命令逐位核验。'
        'actor原32行和权重冻结；解析层原128行容量失败后新增动态完整临界行、硬预算512的候选。'
        '33登记条件共2112相关配对窗口，768条不同命令窗口；容量候选复用原输入，不增加独立样本。'
        '全域LHS是边际分层；条件分布按初始几何筛选；不报告IID安全置信区间。')
    banks=r['pose_banks'];n=sum(b['candidate_count'] for b in banks)
    r['bank_note']=(f'条件初态：{n}个候选，按采样次序取192个初态无违规且所有非豁免球距≥0.1mm的姿态；'
                    f'候选初态违规{sum(b["initial_violation_count"] for b in banks)}个。所有候选与选择标志均保留。'
                    '未使用后续策略成败筛选；候选数量不加到15秒闭环测试样本。UR关节周期角会对应重复空间姿态。')
    r['overview']='随机更广已暴露原约束容量失败；安全性能按初态无违规的完整窗口报告，尚不证明全操作空间安全。'
    r['verdict']=(
        '原128行组9/9条件容量溢出，576个窗口无效，不能记为0违规。动态行预算512是评测候选，全部保留临界约束，actor不变。'
        '全域未筛选初态192个只有6个无违规，须与新增192个条件初态单列；条件初态合法不等于运动后安全。'
        '灰色关节分箱和未暴露臂对都是覆盖缺口。末端格数是实测访问量，范围框不等于可达空间体积；26关节边际覆盖不等于联合覆盖。'
        '本轮没有事后触发或按失败时刻停止。命令上限0.05rad；远处旁路可能超过普通求解器0.024999rad增量箱，实际越箱次数另存，不冒称全部限幅通过。'
        '测量仅控制周期末非豁免机器人/桌面安全球，不证明物体接触、抓持、连续碰撞或实机安全。尚有运行后违规时不能宣称System 0安全通过。')
    for a in r['aggregates']:
        a.update(label=FLOW_LABELS[a['flow']],method_label=METHOD_LABELS[a['method']],
                 planned_windows=192,pending_windows=192-a['attempted_windows'])
    r['coverage_cases']=[]
    for group in r['group_coverage']:
        c=group['coverage']
        r['coverage_cases'].append(dict(
            label=f'跨种子并集：{FLOW_LABELS[group["flow"]]} · {METHOD_LABELS[group["method"]]}',
            flow=group['flow'],method=group['method'],seed=' / '.join(map(str,group['seeds'])),
            initial=c['initial_joint'],visited=c['visited_joint'],ee_initial=c['initial_ee'],
            ee_visited=c['visited_ee'],pairs=group['pairs'],
            note=(f'已完成{group["completed_cells"]}/3种子，共{c["episodes"]}个初态无违规窗口，包含后续失败。'
                  f'平均单窗口关节跨度{c["mean_within_window_joint_range"]*100:.2f}%、总路径{c["mean_joint_path_rad"]:.2f}rad。'
                  '体素按跨种子并集计数，不相加重复格。初态橙框与实测蓝色分开；边际分箱不证明26维联合覆盖。'
                  '臂对80mm按安全球余量≥3帧统计，运动方向标签只按相对末端速度。')))
    for row in r['rows']:
        if row['status']!='complete' or not row['initially_safe']:continue
        c=row['safe_initial_coverage']
        r['coverage_cases'].append(dict(
            label=f'{FLOW_LABELS[row["flow"]]} · {METHOD_LABELS[row["method"]]} · seed{row["seed"]}',
            flow=row['flow'],method=row['method'],seed=row['seed'],initial=c['initial_joint'],visited=c['visited_joint'],
            ee_initial=c['initial_ee'],ee_visited=c['visited_ee'],pairs=row['pairs_safe_initial'],
            note=(f'仅该条件{c["episodes"]}个初态无违规窗口，包含后续失败。平均单窗口关节跨度{c["mean_within_window_joint_range"]*100:.2f}%、'
                  f'关节总路径{c["mean_joint_path_rad"]:.2f}rad；四臂同时每步移动>1mrad比例（全组）{row["four_arms_moving_fraction"]*100:.1f}%。'
                  '热图包含初态和全部控制末实测q，超软限位样本不裁入分箱。末端已扣环境原点，格数不表示可达体积。'
                  '臂对80mm按安全球余量≥3帧统计；接近/远离/切向只按相对末端速度标注。')))
    r['default_coverage']=next((i for i,c in enumerate(r['coverage_cases']) if c['flow']=='feasible_burst' and c['method']=='dynamic512'),
                               next((i for i,c in enumerate(r['coverage_cases']) if c['flow']=='feasible_burst'),0))
    r['traces']=[]
    for protected in r['rows']:
        if protected['method']!='dynamic512' or protected['status']!='complete':continue
        raw=next(x for x in r['rows'] if x['flow']==protected['flow'] and x['seed']==protected['seed'] and x['method']=='raw')
        assert raw['status']=='complete'
        a,b=load(Path(raw['path'])/'cell_001.npz'),load(Path(protected['path'])/'cell_001.npz')
        choices=protected['failed_safe_envs'] or raw['failed_safe_envs']
        if not choices:choices=np.flatnonzero(~a['initial_violation']).tolist()
        if not choices:continue
        e=choices[0];idx=np.arange(0,900,3);values=np.stack([a['official_margins'][idx,e].min(-1),b['official_margins'][idx,e].min(-1)])*1000
        low,high=float(values.min()),float(values.max());pad=max(1.,(high-low)*.08)
        r['traces'].append(dict(
            label=f'{FLOW_LABELS[protected["flow"]]} · seed{protected["seed"]} · env{e}',unit='mm',
            flow=protected['flow'],time_s=((idx+1)*r['dt']).tolist(),x_min=0,x_max=15,
            y_min=low-pad,y_max=high+pad,events=[],
            series=[dict(label='无保护',color='#dc2626',values=values[0].tolist()),
                    dict(label='System 0动态行候选',color='#7c3aed',values=values[1].tolist())],
            caption=('按候选失败环境编号选首个；候选无失败时按raw失败编号选首个，双方无失败则选首个合法初态。'
                     '这是代表性轨迹，不是成功率统计。完整初态和900步命令一致；每3帧显示一次，违规统计用完整900帧。'
                     '曲线是四个非豁免球距通道的最小值，负值为球层违规；完整失败和容量失败均计入表。')))
    r['default_trace']=next((i for i,t in enumerate(r['traces']) if t['flow']=='feasible_burst'),0)
    (OUT/'panel_data.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

    c=r['coverage_cases'][r['default_coverage']]
    counts=np.array(c['visited']['counts']);initial=np.array(c['initial']['counts'])
    fig,ax=plt.subplots(figsize=(10,7));image=ax.imshow(np.log1p(counts),aspect='auto',cmap='Blues')
    for j,i in zip(*np.where(initial>0)):ax.add_patch(plt.Rectangle((i-.5,j-.5),1,1,fill=False,edgecolor='#d97706',linewidth=.6))
    labels=[f'{arm} j{k+1}' for arm,njoint in zip(['F_L','F_R','U_L','U_R'],[7,7,6,6]) for k in range(njoint)]
    ax.set_yticks(range(26),labels);ax.set_xticks(range(10),[f'{i*10}-{(i+1)*10}%' for i in range(10)])
    ax.set(xlabel='normalized soft joint limit (marginal bins)',title=f'Initial-safe measured coverage: {c["flow"]}, {c["method"]}, seeds {c["seed"]}')
    fig.colorbar(image,ax=ax,label='log(1 + observed samples)');fig.tight_layout()
    fig.savefig(OUT/'joint_coverage.png',dpi=160);fig.savefig(OUT/'joint_coverage.svg',metadata={'Date':None});plt.close(fig)

    report=[f'# 四臂强随机与操作空间覆盖\n\n{r["summary"]}\n\n{r["protocol_note"]}\n\n{r["bank_note"]}\n\n{r["verdict"]}\n',
            '|随机来源|方法|完整窗口|初态违规|合法初态后违规|深度>5mm|容量失败|待完成|\n|---|---|---:|---:|---:|---:|---:|---:|']
    for a in r['aggregates']:
        outcome=f'{a["safe_initial_violations"]}/{a["initially_safe_evaluated"]}' if a['initially_safe_evaluated'] else '无完整合法初态窗口'
        report.append(f'|{a["label"]}|{a["method_label"]}|{a["completed_windows"]}/192|{a["initial_violations"]}/{a["attempted_windows"]}|{outcome}|{a["safe_initial_damaging"]}|{a["invalid_windows"]}|{a["pending_windows"]}|')
    report+=['\n## 科研口径与证据\n',
             '- 预登记PLAN、CAPACITY_PLAN、FEASIBLE_PLAN与各campaign_plan保留。原128行失败不被扩容候选替换；actor32输入不变，没有修改生产安全代码或学习权重。',
             '- 三条CPU随机流分别产生方向、保持时间、幅度档；初态另用派生种子。逐步IID源输入无状态反馈；异步保持源有时间相关性，不能叫逐帧IID。',
             '- 同种子wide_iid/wide_burst复用初态；System 0扩容复用原命令；新条件分布的192条命令使用新种子。768条不同命令窗口也不是IID可靠性样本。',
             '- 19项CPU契约通过，覆盖分层/裁剪、源复现、四臂异步、真实FIFO、真实子进程隔离、exit0但协议失败、源漂移、128行溢出与动态行无丢弃、覆盖统计。不是19项系统安全证明。',
             '- 真实模拟全部完整窗口逐位核验命令、初态、6步FIFO；所有配对初态flag一致；所有完结条件的checkpoint、源码、解析配置、协调器、backstop相同。',
             '- 图形设备枚举日志曾报错，但几何bank完整产物核验通过。GPU1闭环显式cuda:1、两设备可见；GPU0原条件另列。配对在同GPU，不声称跨GPU动态前缀逐位相同。',
             '- 软限位[2.5%,97.5%]跨度是关节边际范围。UR周期角可能对应同一空间位置，未计算26维可达/无碰撞空间覆盖率。',
             '- 面板默认只显示初態无违规窗口的实际覆盖，仍包含后续失败；初始分箱与实测分箱分别统计，超软限位实测q不裁入范围。完整all_coverage另存。',
             '- 末端10cm实测体素、包围框和六臂对风险暴露用于发现覆盖洞，不把包围框当体积覆盖分母；接近/远离/切向标签依据末端，不代表最近球对的速度。臂对探针为球面几何余量，包含豁免球对；官方违规以上方四通道非豁免判定为准。',
             '- 新条件初态三组在第一条随机指令到达前（前6步）raw/System 0分别8/5/5个窗口越界；这些失败保留在192分母内。静态球距合法不保证物理初态稳定，不能只归因于安全策略响应。',
             '- bank全部5888候选、192选取、其余拒绝和多余合格记录保留在NAS；候选不计闭环窗口。选取只用初始几何，不用后续安全结果。',
             '- 原始NPZ、输入tape、全部候选、日志保留在NAS实验目录；网页公开全部协议、源hash、JSON episode、CSV汇总和可下载图件，未上传NPZ。',
             '- 强输入0.05rad/步高于普通求解器箱0.024999rad/步。记录实际目标增量和越箱次数；不把远处旁路默认算成限幅通过。',
             '- 未完成窗口保留“待完成”，无效窗口保留“容量失败”，不在安全分母内。没有物体、接触力、连续碰撞或实机验收。',
             '\n![合法初态实测关节覆盖](joint_coverage.png)\n',
             f'全结果见[results.json](results.json)，逐条件见[windows.csv](windows.csv)。实际状态：{r["campaign_status"]}。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    previous=json.loads(DATA.read_text());current=deepcopy(previous)
    current.update(safety_random_space=r,summary=r['summary'],updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    link=dict(label='最新：四臂强随机与操作空间覆盖',url='https://asimfish.github.io/safeduo-dashboard/docs/safety_random_space_20261004/REPORT.md')
    current['links']=[link]+[x for x in current['links'] if x['url']!=link['url']]
    for key,value in previous.items():
        if key not in ['updated','summary','links','safety_random_space']:assert current[key]==value,key
    (args.preview_json or DATA).write_text(json.dumps(current,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    public=DASH/'docs'/OUT.name;public.mkdir(parents=True,exist_ok=True)
    files=[*OUT.glob('*.py'),*OUT.glob('*.md'),*OUT.glob('*.json'),OUT/'windows.csv',OUT/'all_contract_tests.log',
           OUT/'red_contract.log',OUT/'zero_exit_red.log',OUT/'joint_coverage.png',OUT/'joint_coverage.svg']
    for f in files:
        if f.name in ['preview_scientific.json','DELIVERY.json','finalization_state.json']:continue
        if f.suffix in ['.log','.svg']:
            (public/f.name).write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n')
        else:shutil.copy2(f,public/f.name)
    shutil.copytree(OUT/'dependencies',public/'dependencies',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for row in r['rows']:
        dest=public/'cells'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        for f in Path(row['path']).glob('*.json'):shutil.copy2(f,dest/f.name)
    for bank in banks:
        dest=public/'pose_banks'/str(bank['seed']);dest.mkdir(parents=True,exist_ok=True)
        src=Path('/mnt/nas/data/lyf/double_hand/safety_random_space_20261004/pose_banks')/str(bank['seed'])/'metadata.json'
        shutil.copy2(src,dest/'metadata.json')
    print(json.dumps({'completed':r['completed_windows'],'invalid':r['invalid_windows'],'pending':r['pending_windows'],
                      'coverage_cases':len(r['coverage_cases']),'traces':len(r['traces']),'legacy_preserved':True}))


if __name__=='__main__':main()
