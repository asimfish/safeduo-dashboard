"""Build a matched-condition report from completed frozen protocols only."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load_campaign(path):
    path = ROOT / path
    protocol = json.loads((path / 'protocol.json').read_text())
    if protocol['status'] != 'complete':
        raise ValueError(f'incomplete evidence: {path.name}')
    rows = [r for r in json.loads((path / 'episodes.json').read_text()) if r['method'] == 'system0']
    return rows, {'path': str(path.relative_to(ROOT)),
                  'protocol_sha256': hashlib.sha256((path / 'protocol.json').read_bytes()).hexdigest(),
                  'episodes_sha256': hashlib.sha256((path / 'episodes.json').read_bytes()).hexdigest(),
                  'args': protocol['args'], 'dt': protocol['dt'], 'steps': protocol['steps'],
                  'checkpoint_sha256': protocol['checkpoint_sha256']}


def aggregate(rows):
    keys = ('cross', 'self_F', 'self_U')
    return {'episodes': len(rows), 'violations': sum(r['violation'] for r in rows),
            'damaging': sum(r['damaging'] for r in rows),
            'arm_arm_min_mm': 1000 * min(min(r['min_margin_m'][k] for k in keys) for r in rows),
            'mean_joint_path_rad': float(np.mean([r['measured_joint_path_rad'] for r in rows])),
            'exec_command_l2_ratio': sum(r['executed_l2_sum'] for r in rows) / sum(r['command_l2_sum'] for r in rows),
            'mean_zero_exec_with_intent_rate': float(np.mean([r['zero_exec_with_intent_rate'] for r in rows])),
            'warning_exposed': sum(any(r['pair_warn_steps']) for r in rows),
            'near_5mm': sum(min(r['min_pair_margin_m']) < .005 for r in rows),
            'violations_by_class': {k: sum(r['violation_by_class'][k] for r in rows)
                                    for k in ('cross', 'self_F', 'self_U', 'table')}}


def identity(r):
    return r['flow'], r['amp'], r['seed'], r['env_id']


def compare(label, before, after):
    a, b = {identity(r): r for r in before}, {identity(r): r for r in after}
    if len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError('duplicate or unmatched experimental identities')
    fixed = ('pair', 'distance_band', 'target_gap_m', 'direction', 'source_seed', 'checkpoint_sha256', 'dt')
    for key in a:
        for field in fixed:
            if a[key].get(field) != b[key].get(field):
                raise ValueError(f'unmatched condition {key}/{field}')
        if key[0] == 'uniform_random' and a[key]['command_sha256'] != b[key]['command_sha256']:
            raise ValueError(f'random command tapes differ for {key}')
    pairs = [{'group': label, 'flow': key[0], 'amp': key[1], 'seed': key[2], 'env_id': key[3],
              'pair': a[key].get('pair', ''), 'band': a[key].get('distance_band', ''),
              'target_direction': a[key].get('direction', ''),
              'before_violation': a[key]['violation'], 'after_violation': b[key]['violation'],
              'before_min_self_F_mm': a[key]['min_margin_m']['self_F'] * 1000,
              'after_min_self_F_mm': b[key]['min_margin_m']['self_F'] * 1000,
              'before_joint_path_rad': a[key]['measured_joint_path_rad'],
              'after_joint_path_rad': b[key]['measured_joint_path_rad']} for key in sorted(a)]
    return {'label': label, 'before': aggregate(before), 'after': aggregate(after),
            'recovered': sum(a[k]['violation'] and not b[k]['violation'] for k in a),
            'new_failures': sum(not a[k]['violation'] and b[k]['violation'] for k in a)}, pairs


def main():
    groups = [
        ('组合防护 · 六配对压力 · 5 秒',
         ['artifacts/pair_stratified_matrix_v3_20261002'],
         ['artifacts/safety_refine_20261002/combined_matrix32']),
        ('六配对压力 · 延长到 10 秒',
         ['artifacts/safety_refine_20261002/baseline_long'], ['artifacts/safety_refine_20261002/combined_long']),
        ('四臂同时运动 · 工作空间目标流',
         ['artifacts/safety_refine_20261002/baseline_other'], ['artifacts/safety_refine_20261002/combined_diagnostic', 'artifacts/safety_refine_20261002/combined_directed']),
        ('纯 IID 均匀随机 · 10 秒',
         ['artifacts/safety_refine_20261002/baseline_random'], ['artifacts/safety_refine_20261002/combined_random']),
    ]
    rows, paired, manifests = [], [], []
    for label, before_paths, after_paths in groups:
        sides = []
        for paths in (before_paths, after_paths):
            records = []
            for path in paths:
                ep, manifest = load_campaign(path)
                records.extend(ep)
                manifests.append(manifest)
            sides.append(records)
        row, pairs = compare(label, *sides)
        rows.append(row)
        paired.extend(pairs)
        if '工作空间' in label:
            for flow in ('l1_full', 'directed_all'):
                detail, _ = compare('  ' + flow, *[[r for r in side if r['flow'] == flow] for side in sides])
                rows.append(detail)
    traces = []
    a = np.load(OUT / 'baseline_diagnostic/cell_001.npz')
    b = np.load(OUT / 'combined_matrix32/cell_005.npz')
    dt = float(json.loads(str(a['meta_json']))['dt'])
    for e in (6, 12, 24):
        traces.append({'label': f'F_L–F_R · seed 0 / env {e}', 'duration_s': 5,
                       'time_s': (np.arange(1, len(a['q']) + 1) * dt).round(6).tolist(),
                       'before_margin_mm': (a['margins'][:, e, 1] * 1000).round(4).tolist(),
                       'after_margin_mm': (b['margins'][:, e, 1] * 1000).round(4).tolist()})
    result = {'schema': 'safeduo.safety_refinement.v1', 'rows': rows, 'traces': traces,
              'candidate': 'duo_env_a31_lag_guard.yaml',
              'change': 'self/table 0.15 s; cross/permanent structural table rows retain 0.06 s; conditional contact rows are predicted',
              'claim': 'mitigation under measured simulator conditions, not safety certification',
              'statistics': 'correlated design repeats; no IID confidence claim',
              'comparison': 'uniform_random exact tape; other flows matched state-feedback source conditions',
              'manifests': manifests,
              'stage1': json.loads((OUT/'stage1_comparison.json').read_text())['rows']}
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (OUT / 'paired_outcomes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(paired)
    report = ['# System 0 提前制动策略验证', '',
              '状态：MITIGATED。冻结权重、指令幅度和物理场景，组合防护将 self/table 预测时间从 60 ms 延长至 150 ms；cross 和永久结构桌面行保持原有 60 ms。条件贴桌豁免行参与桌面预测，避免豁免速度条件失效后才开始制动。既有 a31 profile 保留，组合版使用显式配置 duo_env_a31_lag_guard.yaml。', '',
              '| 条件 | 原策略违规 | 新策略违规 | 原 / 新最小臂间球距 mm | 原 / 新平均关节路径 rad | 原 / 新执行量比例 | 新增失败 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        x,y = r['before'],r['after']
        report.append(f"| {r['label']} | {x['violations']}/{x['episodes']} | {y['violations']}/{y['episodes']} | {x['arm_arm_min_mm']:.3f} / {y['arm_arm_min_mm']:.3f} | {x['mean_joint_path_rad']:.3f} / {y['mean_joint_path_rad']:.3f} | {x['exec_command_l2_ratio']:.3f} / {y['exec_command_l2_ratio']:.3f} | {r['new_failures']} |")
    report += ['', '## 首版回归与第二阶段', '',
               '首版只增强 self 行，在完整历史配对矩阵中 22/768 → 0/768，长窗口 3/32 → 0/32；但四臂目标流 7/64 → 5/64，同时出现两个新增桌面失败（env 14、23）。因此没有将首版作为通用默认策略，保留 stage1_comparison.json 和逐条件转归作为负面证据。', '',
               '桌面 trace 在 env 14 / row 8833 捕获 d=41.70 mm、J·qd=-0.253 m/s、目标积压的线性闭合量约 47.11 mm；60 ms 预测 26.50 mm，150 ms 预测 3.69 mm，后者在有效 d_min=20 mm 前触发更强制动。威胁行已存在，穿过零间距时投影残差为零。env 23 / row 8996 的条件贴桌豁免在速度越过 0.05 m/s 后从 5 mm 边界切回 20 mm；它属于动态接触豁免，不能与永久肩部球结构豁免一起延迟预测。', '',
               '组合版的 L1 诊断复测为 0/32，含两个新增失败在内的五个桌面违规均未再出现，最小非豁免桌面球距约 1.49 mm；该组最小臂间球距约 0.78 mm，余量仍小。完整组合回归结果见上表。', '',
               '## 因果证据与反事实', '',
               '在冻结诊断条件 seed 0 / amp 0.015 / 32 env / 5 s 下，原策略复现 env 6、12、24 的 3 次违规，最深 -6.410 mm。威胁行 6081 在 step 61 已有效：d=148.692 mm、J·qd=-0.731 m/s、J·(target-q)=-140.652 mm。60 ms 预测为 104.828 mm，150 ms 为 39.032 mm；40 mm 行参与门因此提前约 6 帧触发。原轨迹虽然随后满足增量投影，实际闭合仍持续，最终穿过零间距。诊断 trace 记录决策前状态和投影后残差，违规发生时残差为零。', '',
               '同条件候选诊断为 0/32，最小 self_F +12.488 mm。它支持提前制动可缓解这类执行滞后的机制，不能证明所有非线性制动过程都被固定 150 ms 覆盖。env 12 前期还有约 0.09 mm 的短暂投影残差，仍应保留对不可行约束/求解残差的后续审计。', '',
               '## 有效范围与限制', '',
               '- 球层非豁免 margin < 0 算违规；不是现实接触/力/夹持安全保证。最小臂间距列仅涵盖 cross/self_F/self_U，桌面违规另计入总违规。',
               '- 调度 seed 轮换同一几何设计，重复样本相关；尝试次数不等于独立场景数量，不报告 IID 安全置信界。',
               '- warn/near/contact/penetration 是 EE 目标中心间距设计，方向也是目标意图；recede 包含从 home 接近近目标的阶段。',
               '- 工作空间目标流让四臂同时运动，但不是已完成的四臂物体协同任务；纯随机保留同一命令 tape，其他方法对照为状态反馈源的匹配条件。',
               '- 运动量与执行量比例只能排查整体停住的退化，不能代替任务成功率。',
               '- Isaac 启动仍有 GPU frontend bad-state 警告；本轮成功完成 PhysX/Torch 数值模拟且无非有限状态，仍需在干净运行环境复核。', '',
               '## 可复现材料', '',
               '`run_validation.py` 保存第一阶段实验安排，`run_combined_validation.py` 串行执行组合版回归；L1 诊断为同一评测器的 --flows l1_full --seeds 8 --num-envs 32 --duration-s 5 --amps 0.015 --causal-trace。`comparison.json` 记录协议/episode SHA256、配置和分组统计；`paired_outcomes.csv` 保留每个匹配设计条件的转归。原始 NPZ 与完整协议保存在各运行目录，已有历史结果未覆盖。', '',
               '相关安全/旁通/结构豁免/覆盖回归共 103 项通过。捕获闭合行的 CPU 回归在禁用对应 horizon 时为 RED，启用时为 GREEN；它是接口级回归，实际安全结果以完整 Isaac 复测为准。测试也验证永久结构行响应不变，条件贴桌行不能被永久结构豁免屏蔽，以及安全区/正在远离桌面时保留原响应。', '',
               '下一步：优先分析仍残留的轨迹，建立按 qd/目标积压/可制动能力确定的自适应屏障；再扩大初始姿态、目标几何、速度和扰动分布，并将夹持、物体运动、接触力与任务成功加入评测。']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
