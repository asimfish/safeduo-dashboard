"""Publish bounded stop diagnostics with all failures and legacy evidence intact."""
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

from analyze import load

OUT=Path(__file__).resolve().parent
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview-json',type=Path)
    args=parser.parse_args()
    r=json.loads((OUT/'results.json').read_text())
    rows=r['rows']; control=rows[0]
    counts='；'.join(f'{x["lead_steps"]}步 {x["violations"]}/{x["episodes"]}' for x in rows[1:])
    r['summary']=f'遵守原输出限幅、仍等待100ms FIFO：未干预{control["violations"]}/32重叠；事后停止提前{counts}。每条件独立启动，属于制动诊断。'
    r['protocol_note']=(
        'seed12 · 32环境×10秒 · ±0.3rad初态 · 纯随机保持90步（1.5秒）· 幅度0.015rad/步。'
        'actor权重和critical-only规则冻结；每个条件独立新进程且仅一个cell，初态和完整命令逐位匹配。'
        f'实际目标每步限幅{r["ordinary_increment_box_rad"]:.6f}rad（浮点容差5e-7rad），全部FIFO和软限位核验通过。'
        '触发表来自历史失败，六个环境干预、env31未干预；没有前瞻检测器。'
        '探针停止目标未经过原距离行投影，仅验证增量箱和软限位；不推广生产控制。'
        '32对照+96事后诊断窗口相关，未增加独立安全样本；测量仅控制周期末非豁免球距。')
    exact=sum(x['observed_prefix_exact'] for row in rows for x in row['trajectories'])
    r['verdict']=(
        f'四组实际目标在浮点容差内遵守原增量箱，严格6步FIFO逐位核验通过；{exact}/18条干预前观测前缀与控制逐位一致。'
        f'12/30/60步探针全环境分别{rows[1]["violations"]}/{rows[1]["episodes"]}、{rows[2]["violations"]}/{rows[2]["episodes"]}、{rows[3]["violations"]}/{rows[3]["episodes"]}重叠。'
        '该结果用于判断在这些已知失败轨迹中，限幅停止是否可行。'
        '利用事后失败时刻的改善不构成System 0前瞻安全性能；未干预失败与新增失败完整保留。'
        '尚需研发并验证能及时触发的检测器、停止动作与距离约束的兼容性，再扩大种子、操作空间和物体任务。'
        '无接触力、连续碰撞或实机证据。')
    data={x['id']:load(OUT/'cells'/x['id']/'cell_001.npz') for x in rows}
    r['traces']=[]
    colors=['#dc2626','#d97706','#2563eb','#15803d']
    labels=['未干预控制','提前12步（事后）','提前30步（事后）','提前60步（事后）']

    def trace(label,series,unit,caption,events=None):
        values=np.stack([v for _,_,v in series])
        assert np.isfinite(values).all()
        low,high=float(values.min()),float(values.max())
        pad=max((high-low)*.08,.1 if unit=='mm' else 1e-4)
        r['traces'].append(dict(label=label,unit=unit,time_s=((np.arange(values.shape[-1])+1)*r['dt']).tolist(),
                                x_min=0,x_max=float(values.shape[-1]*r['dt']),y_min=low-pad,y_max=high+pad,
                                events=events or [],series=[dict(label=n,color=c,values=v.tolist()) for n,c,v in series],caption=caption))

    envs=sorted(set(rows[1]['triggered_envs'])|set(e for row in rows for e in row['failed_envs']))
    for e in envs:
        note=('env31无触发，失败原样保留。' if e==31 else '触发取自历史失败时刻，是事后干预。')
        prefix='；'.join(f'{row["lead_steps"]}步前缀'+('一致' if next(x for x in row['trajectories'] if x['env']==e)['observed_prefix_exact'] else '不同') for row in rows[1:] if e in row['triggered_envs'])
        trace(f'非豁免球距：env{e}',[(label,c,data[row['id']]['official_margins'][:,e].min(-1)*1000)
                                   for row,label,c in zip(rows,labels,colors)],'mm',
              f'{note}{prefix}。每帧取四个官方非豁免通道最小值，通道和球对可变；负值为球层重叠。曲线不表示物体任务完成或全空间安全。')
    e=15
    trace('实际目标增量：env15（26关节最大值）',
          [(label,c,np.abs(data[row['id']]['controller_target'][:,e]-data[row['id']]['pre_target'][:,e]).max(-1))
           for row,label,c in zip(rows,labels,colors)]+[('普通增量上限','#6b7280',np.full(600,r['ordinary_increment_box_rad']))],
          'rad','展示实际发送目标增量，包含停止探针覆盖后的目标。全32环境、全部步数均做限幅和软限位核验；目标仍等待6步FIFO。符合箱约束不证明符合原距离投影或可部署安全。')
    r['default_trace']=envs.index(15)
    (OUT/'panel_data.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    fig,ax=plt.subplots(figsize=(9,4))
    for row,c,label in zip(rows,colors,['control','bounded -12','bounded -30','bounded -60']):
        ax.plot((np.arange(600)+1)*r['dt'],data[row['id']]['official_margins'][:,15].min(-1)*1000,label=label,color=c)
    ax.axhline(0,color='black',linestyle='--',linewidth=.8)
    ax.set(xlabel='time (s)',ylabel='minimum nonexempt sphere margin (mm)',title='env15: hindsight bounded stop diagnostics; FIFO 100 ms')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'bounded_stop_env15.png',dpi=160);plt.close(fig)
    report=[f'# 限幅停止与独立进程评测\n\n{r["summary"]}\n\n{r["verdict"]}\n\n{r["protocol_note"]}\n',
            '|条件 / 提前量|全环境重叠|干预环境重叠|未干预失败|新增失败|最小非豁免球距|最大实际目标增量|\n|---|---:|---:|---|---|---:|---:|']
    for row in rows:
        label='未干预' if row['lead_steps'] is None else f'{row["lead_steps"]}步（{row["lead_steps"]*r["dt"]*1000:.0f}ms）'
        intervened='—' if row['lead_steps'] is None else f'{row["intervened_violations"]}/{len(row["triggered_envs"])}'
        report.append(f'|{label}|{row["violations"]}/{row["episodes"]}|{intervened}|{row["non_intervened_failed_envs"]}|{row["new_failed_envs"]}|{row["min_nonexempt_mm"]:.3f}mm|{row["max_actual_increment_rad"]:.6f}rad|')
    report+=['\n## 各条停止轨迹\n', '|提前量|env|观测前缀一致|触发/首消息到达/固定目标全部到达(step)|到达前/后重叠|旧目标偏差|\n|---|---:|---|---|---|---:|']
    for row in rows[1:]:
        for x in row['trajectories']:
            report.append(f'|{row["lead_steps"]}|{x["env"]}|{x["observed_prefix_exact"]}|{x["trigger_step"]}/{x["first_stop_message_delivery_step"]}/{x["goal_reached_delivery_step"]}|{x["violation_before_first_delivery"]}/{x["violation_after_first_delivery"]}|{x["old_target_offset_rad"]:.6f}rad|')
    report+=[
        '\n## 可复核证据与边界\n',
        '- 预登记三种提前量和同六个环境；全32环境逐条结果见 all_environments.csv，协议和聚合结果随报告发布。未给env31增补事后触发，没有剔除未干预失败。',
        '- 原始NPZ保留在本地实验目录；网页发布协议、全部episode JSON、汇总、CSV和逐帧曲线，没有上传完整原始NPZ。',
        '- 17项CPU契约通过。RED为限幅功能缺失的导入失败，随后固定采样、限幅/软限位、真实FIFO和真实子进程隔离检查GREEN；未声称所有契约均单独对旧实现完成RED。',
        f'- 输出箱上限{r["ordinary_increment_box_rad"]:.6f}rad/步，浮点容差5e-7rad；核验实际 controller_target，exec是覆盖前求解器输出，不能用它冒充实际停止动作。',
        '- 独立启动工具拒绝覆盖旧输出，拒绝未完成或多cell子进程，失败则停止后续条件并保留日志/协议。隐藏重置状态原因尚未定位，独立进程只隔离前序条件。',
        f'- 当前控制对旧冷启动控制六项全轨迹逐位一致={all(r["old_control_replay_exact_fields"].values())}。源码差异为先前已记录的可视化文件；当前四组源快照、解析配置、actor和协调器逐项相同。',
        '- 没有修改生产安全核心或学习权重。探针遵守速度箱、软限位和FIFO，但未重新投影原距离约束，仅为机制验证；不能按此直接发布默认控制策略。',
        '- 32控制+96诊断窗口相关，actor/输入重复。不得加到历史512或其他批次构造独立安全置信界；新增独立安全窗口计为0。',
        '- 限幅停止是缓慢向一次采样的实测q移动，并非真实关节立即静止；非受控手部关节不保证停机。无物体/接触力/连续碰撞/实机结果。',
        '- 下一步：建立可前瞻触发的队列与制动裕量检测，验证距离约束兼容性和新种子配对对照；完善全四臂操作范围覆盖。',
        '\n![env15诊断](bounded_stop_env15.png)\n']
    (OUT/'REPORT.md').write_text('\n'.join(report).rstrip()+'\n')
    current=json.loads(DATA.read_text());previous=deepcopy(current)
    current.update(safety_bounded_stop=r,summary=r['summary'],updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    link=dict(label='最新：限幅停止与独立进程报告',url='https://asimfish.github.io/safeduo-dashboard/docs/safety_bounded_stop_20261004/REPORT.md')
    current['links']=[link]+[x for x in current['links'] if x['url']!=link['url']]
    for key,value in previous.items():
        if key not in ['summary','updated','links','safety_bounded_stop']:
            assert current[key]==value,key
    target=args.preview_json or DATA
    target.write_text(json.dumps(current,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    public=DASH/'docs'/OUT.name;public.mkdir(parents=True,exist_ok=True)
    files=[*OUT.glob('*.py'),*OUT.glob('*.md'),*OUT.glob('*.json'),OUT/'contract_tests.log',OUT/'red_contract.log',OUT/'bounded_stop_env15.png']
    files += [x for x in [OUT/'panel.png',OUT/'browser_check.log',OUT/'live_browser_check.log'] if x.exists()]
    for f in files:
        if f.name!='preview_scientific.json':shutil.copy2(f,public/f.name)
    shutil.copytree(OUT/'dependencies',public/'dependencies',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(OUT/'all_environments.csv',public/'all_environments.csv')
    shutil.copy2(OUT/'cells/campaign.json',public/'campaign.json')
    for row in rows:
        dest=public/'cells'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        for f in (OUT/'cells'/row['id']).glob('*.json'):shutil.copy2(f,dest/f.name)
    print(json.dumps(dict(preview=bool(args.preview_json),rows=len(rows),trajectories=18,traces=len(r['traces']),legacy_preserved=True)))


if __name__=='__main__':main()
