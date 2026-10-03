"""Publish observations without turning hindsight holds into safety successes."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

from analyze import load

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'safety_latency_forensics_20261003'
DASH = Path('/home/liyufeng/safeduo-dashboard')
DATA = Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')
parser = argparse.ArgumentParser()
parser.add_argument('--preview-json', type=Path)
args = parser.parse_args()
r = json.loads((OUT / 'results.json').read_text())
r['dt'] = .016666
r['candidate_status'] = 'experimental_not_promoted'
counts = '；'.join(f"{x['delay_ms']:.0f}ms {x['violations']}/{x['initially_safe']}" for x in r['controls'])
reverse_counts = '；'.join(f"{x['delay_ms']:.0f}ms {x['violations']}/{x['initially_safe']}" for x in r['reverse_controls'])
cold_counts = '；'.join(f"{x['delay_ms']:.0f}ms {x['violations']}/{x['initially_safe']}" for x in r['cold_start_controls'])
repeat = r['repeated_control']
r['summary'] = (f'冻结临界行方案，独立进程首窗口：{cold_counts}。'
                f'连续窗口正向：{counts}；反向：{reverse_counts}。'
                '初态与完整随机命令逐位一致，运行历史影响已单列。')
r['diagnostic_note'] = (
    '在本次100ms控制的首次失败之前12步，分别固定触发前旧持久目标、或一次采样的实测q。'
    '只作用于该控制的失败环境，停止消息仍等待6步FIFO，手部关节行为不变。'
    '这是利用事后失败时刻的诊断；下表观察不计入预防成功率。'
    '对原第三cell控制的q前缀均不同，仅作同输入重播；对冷启动100ms控制的观测前缀见表。')
pair_exact = [x['env'] for x in r['stop_pairwise_prefix'] if x['observed_prefix_exact']]
r['diagnostic_note'] += (f' 两停止方式相互比较，env{pair_exact}的q、qd、球心、命令、目标与待执行输入观测前缀逐位一致；'
                         '隐藏物理求解器状态未记录。逆目标超调指新固定目标在运动反方向时，关节仍沿原速度方向额外移动。')
r['diagnostic_note'] += (' 全32环境也保留：' + '；'.join(
    f'{"固定旧目标" if s["mode"] == "stored" else "固定实测q"}仍有{s["all_environment_violations"]}/{s["all_environment_windows"]}违规，未干预失败env{s["non_intervened_failed_envs"]}'
    for s in r['stop_diagnostics']) + '。未因触发表未包含env31而隐藏该失败。')
not_exact = sum(not x['intervention_prefix_q_bit_exact'] for s in r['stop_diagnostics'] for x in s['trajectories'])
cold_exact = sum(x['cold_control_observed_prefix_exact'] for s in r['stop_diagnostics'] for x in s['trajectories'])
post = {s['mode']: sum(x['violation_after_stop_delivery'] for x in s['trajectories']) for s in r['stop_diagnostics']}
n = len(r['stop_diagnostics'][0]['trajectories'])
measured = next(s for s in r['stop_diagnostics'] if s['mode'] == 'measured')['trajectories']
overshoot = [x['max_further_travel_away_from_target_200ms_rad'] for x in measured]
jumps = [x['actual_target_jump_at_trigger_rad'] for x in measured]
r['diagnostic_note'] += (f' 实测q固定目标探针一次跳变{min(jumps):.3f}–{max(jumps):.3f}rad，'
                         f'超过普通输出{measured[0]["ordinary_increment_box_rad"]:.3f}rad/步的箱约束；'
                         '这是显式绕过普通增量约束的机制诊断，不是兼容现有输出约束的修复。')
r['verdict'] = (
    f"100ms首窗口与上一轮q/qd/exec/球距/目标/球心逐位一致={all(r['cold_100_replay_exact_fields'].values())}，仍7/32。"
    f"同参数第三cell为6/32，q与旧首窗口最大差{repeat['max_q_difference_rad']:.6f}rad。"
    '存在运行顺序/重置历史影响，隐藏状态原因尚未定位；不记作策略改进。'
    f'固定目标诊断中，停止消息到达后仍有重叠的轨迹：旧目标{post["stored"]}/{n}，实测q目标{post["measured"]}/{n}；'
    f'对原第三cell的{not_exact}条q前缀不同，对冷启动控制有{cold_exact}/{2*n}条观测前缀一致。'
    f'两种停止方式相互比较有{len(pair_exact)}/{n}条观测前缀一致，可在该范围判别停止目标方式；'
    'env15旧目标仍重叠约62mm，实测q固定目标未重叠，但触发时刻是事后选定。'
    f'实测q目标到达后仍有{min(overshoot):.4f}–{max(overshoot):.4f}rad可观测逆目标超调。'
    '全批次固定旧目标仍3/32违规，固定实测q仍1/32（未干预env31）；任何一组都不能报告全环境安全。'
    '实测q探针一次目标跳变超出普通增量约束，只能判别原因，不能直接推广。'
    '即使固定目标到达，也必须核验后续运动，不能用零输出宣称已停。'
    '本轮未修改或推广生产策略；现有100ms延迟候选仍不安全。')
r['protocol_note'] = (
    'seed12；32环境×10秒；±0.3rad初态；纯随机保持1.5秒、幅度0.015rad/步。'
    'critical-only固定10mm/128行，actor32行/权重不变，backlog_aware=false；全部目标FIFO精确核对。'
    '主比较各用新进程首窗口；连续cell顺序组单列。冷启动0/100ms复用已完成首cell、只补一个50ms首cell。停止探针不增加独立安全窗口。'
    '既有连续cell结果保留为当时顺序协议下的实测值，不自动等价于独立冷启动条件；后续需按新条件复核。'
    '控制周期末球距测量不证明连续碰撞、物体任务、接触力或实机安全。')
datasets = [load(OUT / f'delay_controls/cell_{i:03d}.npz') for i in range(1, 4)]
stops = {mode: load(OUT / f'stop_{mode}/cell_001.npz') for mode in ['stored', 'measured']}
reference = datasets[2]
cold_reference = load(OUT / 'reverse_controls/cell_001.npz')
worst = int(np.unravel_index(np.argmin(reference['official_margins']), reference['official_margins'].shape)[1])
stop_env = 15  # Registered observed-prefix comparison; no selection of a green-only example.
assert stop_env in pair_exact
r['traces'] = []


def add_trace(label, series, unit, caption, events=None, bounds=None):
    values = np.stack([v for _, _, v in series])
    assert np.isfinite(values).all()
    time = (np.arange(values.shape[-1]) + 1) * r['dt']
    visible = values[:, (time >= bounds[0]) & (time <= bounds[1])] if bounds else values
    low, high = float(visible.min()), float(visible.max())
    pad = max((high - low) * .08, 1e-4 if unit == 'rad' else .1)
    r['traces'].append(dict(label=label, unit=unit, time_s=time.tolist(),
                           x_min=bounds[0] if bounds else 0,
                           x_max=bounds[1] if bounds else float(time[-1]),
                           y_min=low - pad, y_max=high + pad, events=events or [],
                           series=[dict(label=n, color=c, values=v.tolist()) for n, c, v in series],
                           caption=caption))


add_trace(f'延迟对照：100ms组最深重叠环境env{worst}',
          [(f'{d["meta"]["actuator_delay_steps"] * r["dt"] * 1000:.0f}ms', color,
            d['official_margins'][:, worst].min(-1) * 1000)
           for d, color in zip(datasets, ['#15803d', '#d97706', '#dc2626'])], 'mm',
          '同初态、同随机命令，各延迟下物理状态随反馈改变。展示100ms组最深重叠环境；每帧最小官方非豁免通道球距，球对/通道可变。负值表示球层重叠。不能从一条曲线推断全环境或全操作空间安全。')
add_trace(f'独立进程首窗口的延迟对照：env{worst}',
          [(label, color, d['official_margins'][:, worst].min(-1) * 1000)
           for label, color, d in [('0ms 首窗口', '#15803d', datasets[0]),
                                   ('50ms 首窗口', '#d97706', load(OUT / 'cold_50/cell_001.npz')),
                                   ('100ms 首窗口', '#dc2626', cold_reference)]], 'mm',
          '每种延迟都取独立新进程的第一个cell，初态和命令逐位一致。避免前序cell的重置历史混入条件比较；100ms与上一轮完整观察轨迹逐位一致、仍7/32。仅限此种子/分布和控制周期末球距，不是全空间安全证明。')
item = next(x for x in r['stop_diagnostics'][0]['trajectories'] if x['env'] == stop_env)
trigger, arrival = item['trigger_step'], item['first_stop_message_delivery_step']
events = [dict(time_s=trigger * r['dt'], label='离线诊断触发', color='#6b7280'),
          dict(time_s=arrival * r['dt'], label='首条固定目标到达', color='#2563eb')]
add_trace(f'固定目标制动诊断：env{stop_env}（事后触发）',
          [('未干预控制（冷启动）', '#dc2626', cold_reference['official_margins'][:, stop_env].min(-1) * 1000),
           ('固定旧目标', '#d97706', stops['stored']['official_margins'][:, stop_env].min(-1) * 1000),
           ('固定实测q', '#15803d', stops['measured']['official_margins'][:, stop_env].min(-1) * 1000)], 'mm',
          'env15的两停止方式与冷启动未干预控制在q、qd、球心、目标、命令和停止前待执行输入观测前缀一致。旧目标仍重叠，固定实测q未重叠。停止消息都等待100ms；触发表保持原登记不变，是事后选定时刻，绿色不表示策略验收。隐藏物理求解器状态未记录。',
          events, ((trigger - 15) * r['dt'], (arrival + 25) * r['dt']))
add_trace(f'停止消息到达后的实际关节运动：env{stop_env}',
          [(label, color, np.abs(d['q'][:, stop_env] - d['pre_q'][arrival, stop_env]).max(-1))
           for label, color, d in [('固定旧目标', '#d97706', stops['stored']),
                                   ('固定实测q', '#15803d', stops['measured'])]], 'rad',
          '每种探针各自以首条停止消息到达前的实测q为基准，画26个受控关节最大绝对位移。停止目标保持不变，物理关节仍可能运动；这不是接触力，也不测未受控手关节。',
          [events[1]], (arrival * r['dt'], (arrival + 20) * r['dt']))
add_trace('100ms重复控制的边界波动：env31',
          [('正向顺序100ms', '#dc2626', reference['official_margins'][:, 31].min(-1) * 1000),
           ('反向顺序100ms', '#7c3aed', cold_reference['official_margins'][:, 31].min(-1) * 1000),
           ('实测q诊断（env31未干预）', '#15803d', stops['measured']['official_margins'][:, 31].min(-1) * 1000)], 'mm',
          '同规则/输入但不同运行历史，物理轨迹不逐位一致。触发表冻结自原第三cell的6个失败环境，未包含env31；固定实测q探针全批次仍1/32违规，env31失败完整保留。不能把六条干预轨迹0重叠称为全环境安全。')
r['default_trace'] = 2
(OUT / 'panel_data.json').write_text(json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

report = [f'# 严格目标FIFO与固定目标制动\n\n{r["summary"]}\n\n{r["verdict"]}\n',
          '## 匹配输入的延迟对照\n\n|顺序|延迟|违规/初态无违规窗口|最小非豁免球距|平均实测关节范围|\n|---|---|---:|---:|---:|']
for order, rows in [('独立进程首窗口', r['cold_start_controls']),('正向0/3/6', r['controls']), ('反向6/3/0', r['reverse_controls'])]:
    for x in rows:
        report.append(f'|{order}|{x["delay_ms"]:.0f}ms|{x["violations"]}/{x["initially_safe"]}|{x["min_nonexempt_mm"]:.3f}mm|{x["mean_joint_range"]:.2%}|')
report += [f'\n## 停止响应诊断\n\n{r["diagnostic_note"]}\n',
           '|方式|环境|对冷启动控制观测前缀一致|两停止方式观测前缀一致|对原第三cell q最大差rad|到达前/后重叠|到达后200ms移动/逆目标超调rad|\n|---|---:|---|---|---:|---|---:|']
for s in r['stop_diagnostics']:
    for x in s['trajectories']:
        report.append(f'|{s["mode"]}|{x["env"]}|{x["cold_control_observed_prefix_exact"]}|{x["env"] in pair_exact}|{x["intervention_prefix_max_q_difference_rad"]:.6f}|{x["violation_before_stop_delivery"]}/{x["violation_after_stop_delivery"]}|{x["max_arm_joint_displacement_200ms_after_delivery_rad"]:.6f}/{x["max_further_travel_away_from_target_200ms_rad"]:.6f}|')
report += [f'\n## 证据边界\n\n{r["protocol_note"]}\n',
           '- 13项停止探针/FIFO工具契约CPU检查通过，不是安全策略验证。上一轮98项记录仍保留，不以本轮重复计数扩充。',
           '- actor及全部生产源码/解析配置/协调器配置与上一轮逐项相同，四个关键核心源码SHA附于结果。全部仿真协议complete，初态无违规，命令/初态逐位匹配，延迟送达目标逐位符合FIFO。',
           '- 正反序6个对照192窗口加一个50ms冷启动cell32窗口，共224个相关对照窗口。主比较96窗口复用其中首cell，不能重复计数。两个停止探针64个诊断窗口，不计为新增独立安全证据。保留旧候选7/192原计数，不加总为独立安全置信界。',
           '- 未假设抢占队列、执行器侧运行或实机支持；无新视频、物体任务、接触力或连续碰撞证据。',
           f'- 实测q探针一次目标跳变{min(jumps):.6f}–{max(jumps):.6f}rad，普通输出增量上限{measured[0]["ordinary_increment_box_rad"]:.6f}rad/步；全部六条均超过普通箱约束。该探针是机制干预，不是经过原解析投影的可部署动作，尚未验证接触力/物体安全。',
           '- 多cell的factor_manifest只保存最后一个cell的延迟；每cell真实延迟见cell_00x.json.window、协议design及本轮FIFO逐位审计，不能把最后cell的值套到整个矩阵。',
           '- 尚待解决：当前线性历史目标预测与实际延迟动力学的偏差，前缀可重复性，以及在可部署检测器触发时的安全停机可行性。']
(OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
with (OUT / 'stop_observations.csv').open('w') as f:
    records = [dict(mode=s['mode'], **x) for s in r['stop_diagnostics'] for x in s['trajectories']]
    writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)

current = json.loads(DATA.read_text())
previous = deepcopy(current)
current['safety_queue_braking'] = r
current['updated'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
current['summary'] = r['summary']
link = dict(label='最新：队列与固定目标制动报告',
            url='https://asimfish.github.io/safeduo-dashboard/docs/safety_queue_braking_20261003/REPORT.md')
current['links'] = [link] + [x for x in current['links'] if x['url'] != link['url']]
for k in ['safety_forensics', 'safety_perturbation', 'safety_robustness', 'headline', 'historical_headline']:
    if k in previous:
        assert current[k] == previous[k], k
target = args.preview_json or DATA
target.write_text(json.dumps(current, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

public = DASH / 'docs' / OUT.name
public.mkdir(parents=True, exist_ok=True)
files = [*OUT.glob('*.py'), OUT / 'PLAN.md', OUT / 'REPORT.md', OUT / 'results.json',
         OUT / 'stop_schedule.json', OUT / 'stop_observations.csv', OUT / 'contract_tests.log']
files += [p for p in [OUT / 'VALIDATION.md', OUT / 'DELIVERY.json', OUT / 'panel.png',
                     OUT / 'browser_check.log', OUT / 'live_browser_check.log'] if p.exists()]
for file in files:
    shutil.copy2(file, public / file.name)
for name in ['delay_controls', 'reverse_controls', 'cold_50', 'stop_stored', 'stop_measured']:
    dest = public / name
    dest.mkdir(exist_ok=True)
    for file in (OUT / name).glob('*.json'):
        shutil.copy2(file, dest / file.name)
for name in ['counterfactual_runner.py', 'trace_runner.py', 'critical_rows.py']:
    deps = public / 'dependencies'
    deps.mkdir(exist_ok=True)
    shutil.copy2(OLD / name, deps / name)
print(json.dumps(dict(preview=bool(args.preview_json), output=str(target),
                      delay_violations=[x['violations'] for x in r['controls']],
                      diagnostic_trajectories=sum(len(x['trajectories']) for x in r['stop_diagnostics']),
                      historical_evidence_preserved=True)))
