"""Aggregate only complete frozen campaigns; verify random tape identities."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/liyufeng/safeduo')
OUT = ROOT / 'artifacts/safety_robustness_20261002'
METHOD_NAMES = {'raw': '无保护', 'backstop_only': '仅解析兜底', 'system0': 'System 0'}
FLOW_NAMES = {'uniform_random': '逐步 IID 随机', 'held_random': '纯随机保持'}
RUNS = ['legacy_held', 'candidate_random', 'candidate_regression', 'candidate_pair_long',
        'candidate_directed', 'candidate_held_long', 'priority_candidate']


def load(name):
    p = OUT / name
    protocol = json.loads((p / 'protocol.json').read_text())
    if protocol['status'] != 'complete' or protocol['completed_cells'] != len(protocol['design']):
        raise RuntimeError(f'{name}: incomplete evidence cannot enter final report')
    return protocol, json.loads((p / 'episodes.json').read_text())


def summarize(rows):
    exposure = np.array([r['pair_warn_steps'] for r in rows]) > 0
    return dict(
        episodes=len(rows), violations=sum(r['violation'] for r in rows),
        damaging=sum(r['damaging'] for r in rows),
        initial_violations=sum(r['initial_violation'] for r in rows),
        by_class={k: sum(r['violation_by_class'][k] for r in rows)
                  for k in ('cross', 'self_F', 'self_U', 'table')},
        arm_arm_min_mm=1000 * min(min(r['min_margin_m'][k] for k in ('cross','self_F','self_U')) for r in rows),
        mean_joint_range_fraction=float(np.mean([r['joint_range_fraction_mean'] for r in rows])),
        joint_range_fraction_quantiles=np.quantile([r['joint_range_fraction_mean'] for r in rows], [0,.5,1]).tolist(),
        mean_joint_path_rad=float(np.mean([r['measured_joint_path_rad'] for r in rows])),
        exec_command_l2_ratio=sum(r['executed_l2_sum'] for r in rows)/sum(r['command_l2_sum'] for r in rows),
        pair_union_exposed=int(exposure.any(0).sum()),
        exposed_episodes_by_pair=exposure.sum(0).tolist(),
        all_six_pairs_exposed_episodes=int(exposure.all(1).sum()),
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['priority','bounded'],default='bounded')
    args=parser.parse_args()
    bounded=args.phase=='bounded'
    names={'candidate_random':'bounded_final_random','candidate_regression':'bounded_final_regression',
           'candidate_pair_long':'bounded_pair_long','candidate_directed':'bounded_directed',
           'candidate_held_long':'bounded_final_long','priority_candidate':'bounded_l1'} if bounded else {}
    campaigns = {name: load(names.get(name,name)) for name in RUNS}
    controls=[]
    if bounded:
        for alias in ['candidate_random','candidate_held_long']:
            old_protocol,old_rows=load(alias)
            controls.append(old_protocol)
            new_config=dict(campaigns[alias][0]['resolved_config'])
            old_config=old_protocol['resolved_config']
            new_config['safety']=dict(new_config['safety'])
            assert new_config['safety'].pop('project_target_limits') is True
            assert new_config==old_config, 'raw reuse requires identical physics and only an ignored safety flag'
            campaigns[alias][1].extend(e for e in old_rows if e['method']=='raw')
    manifests = [p for name,(p,_) in campaigns.items() if name!='legacy_held']
    ckpt = {p['checkpoint_sha256'] for p in manifests}
    assert len(ckpt) == 1
    keys = ['src/safeduo/safety/geometry.py', 'src/safeduo/safety/backstop.py',
            'src/safeduo/safety/sphere_distance.py', 'src/safeduo/envs/duo_env.py']
    for key in keys:
        # priority_candidate uses the exact safety implementation later frozen
        # for the campaign; trace-only changes may differ in evaluation code.
        assert len({p['source_sha256'][key] for p in manifests}) == 1, key
    random_rows = campaigns['candidate_random'][1]
    long_rows = campaigns['candidate_held_long'][1]
    groups = defaultdict(list)
    for r in random_rows + long_rows:
        groups[(r['flow'],r['method'],r['hold_steps'],r['dt'])].append(r)
    result_rows = []
    for (flow, method, hold, dt), rows in groups.items():
        duration = (10 if hold == 90 else 5)
        result_rows.append(dict(label=f'{FLOW_NAMES[flow]} · {METHOD_NAMES[method]}',
            flow=flow, method=method, hold_steps=hold,
            condition=f"seed {','.join(map(str, sorted({r['seed'] for r in rows})))} · {duration}s · 保持{hold*dt:.2f}s",
            **summarize(rows)))
    by_case = defaultdict(dict)
    for r in random_rows + long_rows:
        key = (r['flow'], r['seed'], r['amp'], r['env_id'], r['hold_steps'])
        by_case[key][r['method']] = r
    paired = []
    for case, methods in by_case.items():
        assert len({r['command_sha256'] for r in methods.values()}) == 1, case
        paired.append(dict(flow=case[0], seed=case[1], amp=case[2], env_id=case[3], hold_steps=case[4],
            command_sha256=next(iter(methods.values()))['command_sha256'],
            raw_violation=methods['raw']['violation'],
            system0_violation=methods['system0']['violation'],
            backstop_violation=methods['backstop_only']['violation'] if 'backstop_only' in methods else 'not_tested'))
    legacy = campaigns['legacy_held'][1]
    now = [r for r in random_rows if r['flow'] == 'held_random' and r['method'] == 'system0']
    key = lambda r:(r['seed'],r['env_id'],r['amp'],r['hold_steps'])
    legacy_by_key = {key(r): r for r in legacy}
    assert len(legacy_by_key) == len(now)
    new_failures = []
    for r in now:
        old = legacy_by_key[key(r)]
        assert r['command_sha256'] == old['command_sha256']
        if r['violation'] and not old['violation']:
            new_failures.append(key(r))
    safe_rows = [r for name, (_, rows) in campaigns.items() if name != 'legacy_held'
                 for r in rows if r['method'] == 'system0']
    baseline_npz = np.load(OUT/'com_diagnostic/cell_001.npz')
    final_npz = np.load(OUT/names.get('priority_candidate','priority_candidate')/'cell_001.npz')
    np.testing.assert_array_equal(baseline_npz['q_initial'], final_npz['q_initial'])
    # Same frozen initial poses across random methods, not just the same seed.
    for name in ['candidate_random','candidate_held_long']:
        p = campaigns[name][0]
        initial_by_seed = {}
        for i, cell in enumerate(p['design'], 1):
            x = np.load(OUT/names.get(name,name)/f'cell_{i:03d}.npz')
            key0 = (cell['flow'],cell['seed'],cell['amp'])
            if key0 in initial_by_seed:
                np.testing.assert_array_equal(initial_by_seed[key0], x['q_initial'])
            initial_by_seed[key0] = x['q_initial']
    diagnostic = np.load(OUT/'com_diagnostic/cell_001.npz')
    valid = diagnostic['pre_row_valid']
    error_before = np.abs(diagnostic['pre_row_full_rate'][valid]-diagnostic['pre_row_body_rate'][valid])
    error_after = np.abs(diagnostic['pre_row_com_rate'][valid]-diagnostic['pre_row_body_rate'][valid])
    times, vals = [], [[],[],[]]
    for t in range(120,171):
        k = np.flatnonzero(diagnostic['pre_row_id'][t,2] == 7276)
        assert len(k) == 1
        times.append(t/60)
        for target, source in zip(vals,['pre_row_full_rate','pre_row_com_rate','pre_row_body_rate']):
            target.append(float(diagnostic[source][t,2,k[0]])*1000)
    traces = [dict(label='同一运动状态：原预测 / 质心预测 / 实测速度', unit='mm/s',
        time_s=times, x_min=times[0], x_max=times[-1], y_min=-40., y_max=100.,
        series=[dict(label=label,color=color,values=v) for label,color,v in zip(
            ['原 link 预测','COM 预测','body 实测'],['#dc2626','#15803d','#2563eb'],vals)],
        caption='seed8 / env2 / row7276；三条速度取自同一原策略轨迹。负数为接近。COM 曲线是重新计算的预测，并非另一条轨迹的速度。')]
    t = ((np.arange(300)+1)/60).tolist()
    traces.append(dict(label='四臂漫游：env2 的实测 self_U 最小余量', unit='mm',
        time_s=t,x_min=0.,x_max=5.,y_min=0.,
        y_max=float(np.ceil(max(baseline_npz['margins'][:,2,2].max(),final_npz['margins'][:,2,2].max())*1000/10)*10),
        series=[dict(label='原延迟防护',color='#dc2626',values=(baseline_npz['margins'][:,2,2]*1000).tolist()),
                dict(label='组合修正',color='#15803d',values=(final_npz['margins'][:,2,2]*1000).tolist())],
        caption=f"同初态 / seed8 / amp0.015 / 5秒；原余量0.78mm，组合修正{final_npz['margins'][:,2,2].min()*1000:.2f}mm。self_U 包含UR各臂自身及两臂间球对。l1_full 为状态反馈源，后续指令不保证相同；该曲线不代表全组最小值。"))
    replay_protocol, replay_ep = load('random_pair_trace')
    replay_raw = [e for e in replay_ep if e['method']=='raw']
    env_id = int(np.argmin([e['min_pair_margin_m'][0] for e in replay_raw]))
    xr=np.load(OUT/'random_pair_trace/cell_001.npz')
    xs=np.load(OUT/'bounded_final_random/cell_006.npz' if bounded else OUT/'random_pair_trace/cell_002.npz')
    np.testing.assert_array_equal(xr['q_initial'],xs['q_initial'])
    np.testing.assert_array_equal(xr['cmd'],xs['cmd'])
    km=json.loads(str(xr['kinematic_meta_json']))
    pt=np.array(km['pair_sphere_idx'][:km['robot_pair_count']])
    arms=np.array(km['sphere_arm_id']);radius=np.array(km['sphere_radii_m'])
    pt=pt[((arms[pt[:,0]]==0)&(arms[pt[:,1]]==1))|((arms[pt[:,0]]==1)&(arms[pt[:,1]]==0))]
    def measured_fr_pair(x):
        c=x['sphere_centers'][:,env_id]
        return 1000*((np.linalg.norm(c[:,pt[:,0]]-c[:,pt[:,1]],axis=-1)
                      -radius[pt[:,0]]-radius[pt[:,1]]).min(axis=-1))
    raw_pair,new_pair=measured_fr_pair(xr),measured_fr_pair(xs)
    traces.append(dict(label=f'纯随机保持：F_L–F_R 实测球距 · seed9 / env{env_id}',unit='mm',
        time_s=t,x_min=0.,x_max=5.,y_min=float(np.floor(min(raw_pair.min(),new_pair.min())/20)*20-20),
        y_max=float(np.ceil(max(raw_pair.max(),new_pair.max())/20)*20+20),
        series=[dict(label='无保护',color='#dc2626',values=raw_pair.tolist()),
                dict(label='System 0',color='#15803d',values=new_pair.tolist())],
        caption=f'四臂26关节同时纯随机，0.5秒保持；同初态与完全相同命令tape。此例原球距最小{raw_pair.min():.2f}mm，保护后{new_pair.min():.2f}mm。曲线只包含F_L与F_R之间的球对；展示最小原F_L–F_R距离窗口的补充重播，不增加独立样本数。'))
    regression_rows=[]
    for name,label,condition in [
        ('priority_candidate','四臂同时工作空间漫游','seed8 · 5s · amp0.015'),
        ('candidate_regression','六臂对分层压力','seed0–3 · 5s · amp0.005/0.015'),
        ('candidate_pair_long','六臂对延长窗口','seed0 · 10s · amp0.015'),
        ('candidate_directed','四臂定向冲突','seed8 · 5s · amp0.015')]:
        regression_rows.append(dict(label=label,condition=condition,**summarize(campaigns[name][1])))
    result = dict(candidate='duo_env_a31_bounded_guard.yaml' if bounded else 'duo_env_a31_priority_guard.yaml', rows=result_rows,
        regression_rows=regression_rows,diagnostic_manifests=[replay_protocol],
        reused_raw_controls=controls,
        protected_total=summarize(safe_rows), legacy_held=summarize(legacy), new_held=summarize(now),
        new_held_failures=new_failures, manifests=manifests, checkpoint_sha256=next(iter(ckpt)),
        rate_error_p99_before_m_s=float(np.quantile(error_before,.99)),
        rate_error_p99_after_m_s=float(np.quantile(error_after,.99)), traces=traces,
        counts_note='窗口相关且各分布不等价；总数用于审计完成量，不是独立采样置信证据。')
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'paired_random.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(paired)
    report = ['# 质心参考点、约束覆盖与纯随机压力测试', '',
        f"最新候选 `{result['candidate']}`。checkpoint SHA256 `{result['checkpoint_sha256']}`。", '',
        '修正球心雅可比参考点、150ms危险行排序、条件接触阻尼，并在投影内编码目标限位，避免后置裁剪把开离指令变成闭合。actor checkpoint及原32行观测、碰撞球与速度上限不变。', '',
        '## 冻结随机组', '',
        '|分布/方法|条件|违规/窗口|最小臂/手球距mm|平均关节范围|六臂对暴露|执行/指令|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in result_rows:
        report.append(f"|{r['label']}|{r['condition']}|{r['violations']}/{r['episodes']}|{r['arm_arm_min_mm']:.3f}|{100*r['mean_joint_range_fraction']:.2f}%|{r['pair_union_exposed']}/6|{r['exec_command_l2_ratio']:.3f}|")
    s=result['protected_total']
    report += ['', '## 回归与审计', '',
        f"新候选共完成{s['episodes']}个保护窗口，违规{s['violations']}，超过5mm的非豁免球重叠{s['damaging']}，初态违规{s['initial_violations']}。按通道：{s['by_class']}。最小臂/手球距{s['arm_arm_min_mm']:.3f}mm。",
        f"短保持随机的旧/新 System0 为 {result['legacy_held']['violations']}/{len(legacy)} → {result['new_held']['violations']}/{len(now)}；新增失败条件{len(new_failures)}。所有纯随机配对的命令SHA和初态均逐条验证相同。",
        '保留仅COM修正导致4/32违规及priority_guard长保持1/32违规的负结果。重复诊断不增加独立样本数。旋转点、条件接触和目标限位均保存RED→GREEN日志，相关CPU回归78项通过。', '',
        '## 范围与限制', '',
        '平均关节范围 = 各关节实测max-min / 该关节soft-limit跨度，再对关节及窗口平均。六臂对暴露是至少一步进入80mm的跨窗口并集，不等于每条轨迹覆盖六对。',
        '最小臂/手球距取 cross/self_F/self_U 三通道，包含本臂手部与本臂连杆，并不只统计臂与另一条臂。桌面使用非豁免违规计数，不能与该距离混用。',
        '执行/指令是安全层输出目标增量范数的比例；无保护输出仍会被后置执行器限位裁剪，因此该比例不等于实际关节运动或任务成功。实际运动量另外来自测量q的路径。',
        '逐步IID噪声与纯随机保持是不同分布，保持时间为0.5秒及1.5秒；后者时间相关。所有关节方向随机且无目标点引导，但不等于姿态空间或危险状态均匀采样。',
        '初态、桌面/基座几何与动力学仍固定；未覆盖随机初始姿态、参数扰动、通信延迟、真实物体协作成功、接触力和实机。零球层违规不是全面安全证明。',
        '输入仅随机控制26个臂关节，未随机控制手指闭合。headless图形层仍有GPU枚举警告；本轮数值状态完整且有限，仍需干净环境复现和独立接触/视频验证。',
        'bounded active set仍可能遗漏复杂多接触状态；保留最小余量和运动量，不能仅用违规率判断改进。', '',
        '完整清单、源码hash及每窗口配对见results.json、protocols与paired_random.csv；诊断过程见DIAGNOSIS.md。']
    if bounded:
        old=np.load(OUT/'target_bounds_diagnostic/cell_001.npz')
        new=np.load(OUT/'bounded_final_long/cell_001.npz')
        error=lambda x:1000*np.abs(x['projected_margin_delta']-x['applied_margin_delta']).max(-1)[:,3]
        old_error,new_error=error(old),error(new)
        traces.append(dict(label='目标限位：投影与实际施加的最大行差异 · env3',unit='mm/step',
            time_s=((np.arange(600)+1)/60).tolist(),x_min=0.,x_max=10.,y_min=0.,
            y_max=float(np.ceil(max(old_error.max(),new_error.max()))),
            series=[dict(label='后置目标裁剪',color='#dc2626',values=old_error.tolist()),
                    dict(label='限位纳入投影',color='#15803d',values=new_error.tolist())],
            caption='每种方法各自活跃行的最大 |J·projected − J·applied|；度量控制指令一致性，不是物理间距。旧模型约8mm级误差会翻转开离方向，新模型降到数值舍入误差。'))
        result['target_bounds_fix']=dict(before_violation_episodes=1,after_violation_episodes=0,episodes=32,
            old_env3_min_self_F_mm=float(old['margins'][:,3,1].min()*1000),
            new_env3_min_self_F_mm=float(new['margins'][:,3,1].min()*1000),
            max_row_delta_error_before_mm=float(1000*np.abs(old['projected_margin_delta']-old['applied_margin_delta']).max()),
            max_row_delta_error_after_mm=float(1000*np.abs(new['projected_margin_delta']-new['applied_margin_delta']).max()))
        result['traces']=traces
        (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        report += ['', '## 目标限位修复', '',
            'phase2长保持32窗口仍有1条F_L–F_R违规，最深32.68mm。row6193/pre-step513的投影开离+1.245mm被目标限位裁成闭合−6.439mm，尽管投影残差几乎为零。',
            f"目标位移限位与速度箱在同一次Dykstra投影中求解，保留原alpha预算方向；同条件复测1/32→0/32。原失败窗口self_F余量−32.68→{result['target_bounds_fix']['new_env3_min_self_F_mm']:.3f}mm。",
            '最新所有保护窗口来自bounded_guard重新执行；无保护控制因RawShim忽略此唯一新增安全开关而复用，已核对其余解析配置、初态及命令SHA。复用控制不增加独立样本数量。']
    report += ['', '## 回归分组', '',
               '|分组|条件|违规/窗口|最小臂/手球距mm|平均关节路径rad|', '|---|---|---:|---:|---:|']
    for row in regression_rows:
        report.append(f"|{row['label']}|{row['condition']}|{row['violations']}/{row['episodes']}|{row['arm_arm_min_mm']:.3f}|{row['mean_joint_path_rad']:.3f}|")
    report += ['', '## 可视化', '',
               '![同状态速度诊断和匹配轨迹余量](kinematics_and_margin.png)', '',
               '![纯随机操作覆盖](random_coverage.png)', '',
               '![同命令的F_L–F_R安全对照](random_pair_safety.png)', '',
               '补充重播用于取完整球心以画臂间球距，未加入独立样本总数；选择规则为seed9中原F_L–F_R球距最小的窗口。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(s,ensure_ascii=False))


if __name__ == '__main__':
    main()
