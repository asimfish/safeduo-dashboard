"""Build a public, scoped diagnostic from completed independent receipts."""
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_motion import merge
from bind_first_failures import sha

P = Path(__file__).resolve().parent
W = Path('/home/liyufeng/safeduo-dashboard-motion-audit-20261008')
DEST = W / 'docs' / 'safety_motion_response_audit_20261008'


def read(path):
    return json.loads(path.read_text())


def main():
    motion = [read(d / 'MOTION_FORECAST_AUDIT.json') for d in [P, P / 'block1']]
    native = [read(d / 'FIRST_SCORE_EVENT_AUDIT.json') for d in [P, P / 'block1']]
    paired = read(P / 'PAIRED_GEOMETRY.json')
    prior_force_path = Path('/home/liyufeng/safeduo/artifacts/safety_velocity_arrival_20261008_1519/BLOCK0_CLOSED_RESULT.json')
    prior_force = read(prior_force_path)
    shutil.copyfile(prior_force_path, P / 'PRIOR_BLOCK0_VECTOR_RESULT.json')
    assert all(m['original_post_score_all_frames_exact'] for m in motion)
    assert all(m['windows'] == 64 and m['control_steps'] == 960 and m['rows'] == 9021 for m in motion)
    assert paired['new_candidate_windows'] == 128 and len(paired['comparisons']['adaptive_joint']['new_failures']) == 2
    assert paired['summaries']['velocity_arrival']['strict_windows'] == 9
    assert paired['summaries']['adaptive_joint']['strict_windows'] == 12
    assert len(paired['comparisons']['adaptive_joint']['rescued']) == 5
    combined = {str(h): {} for h in [1, 6, 7, 18]}
    grouped = {str(h): {} for h in [1, 6, 7, 18]}
    events = []
    for block, report in enumerate(motion):
        for h in combined:
            cleaned = {k: {key: value for key, value in v.items() if key != 'mean_absolute_error_m'}
                       for k, v in report['metrics'][h].items()}
            merge(combined[h], cleaned)
            for cls, bins in report['metrics_by_class'][h].items():
                cleaned = {k: {key: value for key, value in v.items() if key != 'mean_absolute_error_m'}
                           for k, v in bins.items()}
                merge(grouped[h].setdefault(cls, {}), cleaned)
        for event in native[block]['events']:
            events.append(dict(block=block, **event))
    for horizon in combined.values():
        for values in horizon.values():
            values['mean_absolute_error_m'] = values['absolute_error_sum_m'] / values['observations'] if values['observations'] else None
    late_rows = [r for e in events if not e['initial_geometry_negative'] for r in e['rows']]
    assert len(events) == 9 and len(late_rows) == 10
    assert all(r['selected'] and r['J_actually_applied_offset_m'] > 0 and min(r['J_pending_offsets_m']) > 0 for r in late_rows)
    kinds = {kind: sum(r['event_kind'] == kind for e in events for r in e['rows']) for kind in
             ['RAW_DISTANCE_CROSSING', 'INITIAL_NEGATIVE', 'NEGATIVE_DISTANCE_EXEMPTION_REVOKED']}
    residuals = {k: max(e['returned_residuals'][k] for e in events) for k in events[0]['returned_residuals']}
    candidate = paired['summaries']['velocity_arrival']
    fraction = candidate['joint_mean_span_fraction']
    result = dict(status='COMPLETE_KNOWN_BANK_PASSIVE_MOTION_DIAGNOSTIC',
                  adoption='REJECTED_TWO_NEW_PRIMARY_FAILURES',
                  distinct_initial_cases=128, new_candidate_windows=128, reused_comparator_windows=256,
                  audited_control_macros=128 * 960, geometry_rows_per_frame=9021,
                  original_post_geometry_all_frames_exact=True, original_first_event_native_clocks_exact=True,
                  first_scored_failure_windows=9, initially_negative_windows=1,
                  late_first_scored_failure_windows=8, late_rows=10, first_scored_row_kinds=kinds,
                  all_late_rows_selected_and_applied_pending_locally_away=True,
                  summaries=paired['summaries'], comparisons=paired['comparisons'],
                  mean_per_window_marginal_joint_span_percent=[100 * min(fraction), 100 * max(fraction)],
                  passive_constant_speed_metrics=combined, metrics_by_class=grouped,
                  first_score_events=events, maximum_returned_residuals=residuals,
                  contact_point_scalar_full_trials='UNKNOWN_NOT_RECORDED',
                  previously_reported_block0_vector_peak_N=prior_force['candidate']['peak_vector_normal_n'],
                  prior_block0_vector_receipt_sha256=sha(prior_force_path),
                  contact_audit_scope='Prior closed block0 vector receipt only; this motion audit does not independently re-audit contact streams.',
                  native_physics_trials_launched_by_this_audit=0, fresh_holdout=False,
                  continuous_time_safety_certified=False, hardware_approved=False,
                  evidence={str(d.relative_to(P) / name): sha(d / name) for d in [P, P / 'block1'] for name in
                            ['REGISTRATION.json', 'MOTION_FORECAST_AUDIT.json', 'FIRST_FAILURE_NATIVE_BINDING.json', 'FIRST_SCORE_EVENT_AUDIT.json']},
                  limitations=['All rows/frames are correlated; metrics are passive forecast observations, not independent experiments or controller failure rates.',
                               'Eligibility requires nonnegative origin gap and origin/future nonexemption; exemption revocation is separately reported.',
                               'Constant-speed prediction excludes the actual combined FIFO/admission/projector mechanism; a forecast miss is not a reconstructed admission miss.',
                               'Exact first-event native bindings do not isolate actuator inertia, nonlinear geometry or any particular repair as the cause.',
                               'Raw original native streams remain at registered local roots; this small page distributes derived evidence and selected original images, not a complete public raw-stream archive.'])
    (P / 'gates.json').write_text(json.dumps(result, indent=2) + '\n')
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    with np.load(P / 'ENV49_TRACE.npz', allow_pickle=False) as trace:
        d, velocity, dt = trace['d'][:, 1], trace['velocity'][:, 1], float(trace['dt'])
    x = np.arange(len(d))
    predicted = np.full(len(d), np.nan)
    predicted[7:] = d[:-7].astype(float) + 7 * dt * velocity[:-7]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(x, d * 1000, label='Actual raw gap', color='#155d98', linewidth=2)
    axes[0].plot(x, predicted * 1000, label='Frozen-speed forecast from 7 steps earlier', color='#c75528', linestyle='--')
    axes[0].axhline(0, color='#ad173b', linewidth=1, label='Raw zero gap')
    axes[0].axhline(20, color='#777777', linestyle=':', label='20 mm descriptive marker')
    axes[0].axvline(895, color='#ad173b', linestyle=':')
    axes[0].set_ylim(-12, 85)
    axes[0].set_ylabel('Signed raw gap (mm)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title('Known-bank block 0 / env 49 / U_L forearm-table row 8887')
    axes[1].plot(x, velocity, color='#175e4c', label='Recorded J qdot (event 894 native-bound)')
    axes[1].axhline(0, color='#777777', linewidth=1)
    axes[1].axvline(895, color='#ad173b', linestyle=':', label='First negative pre boundary 895 = post macro 894')
    axes[1].set_ylabel('Signed current gap rate (m/s)')
    axes[1].set_xlabel('Original pre-control boundary index')
    axes[1].set_xlim(858, 902)
    axes[1].legend(loc='lower left', fontsize=8)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(P / ('env49_motion.' + extension), dpi=170)
        if extension == 'svg':
            vector = P / 'env49_motion.svg'
            vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines()) + '\n')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    for offset, (mode, label, color) in enumerate([
            ('adaptive_joint', 'Sealed primary baseline', '#8697ad'),
            ('velocity_arrival', 'Velocity/FIFO candidate', '#155d98')]):
        ax.bar(np.arange(26) + (offset - .5) * .36,
               np.asarray(paired['summaries'][mode]['joint_mean_span_fraction']) * 100,
               width=.36, label=label, color=color)
    ax.set_xticks(range(26), [f'{arm}{j + 1}' for arm, width in [('FL', 7), ('FR', 7), ('UL', 6), ('UR', 6)] for j in range(width)], rotation=60)
    ax.set_ylabel('Mean per-window span / soft-limit range (%)')
    ax.set_title('128 reused initial cases: marginal motion span, not full workspace coverage')
    ax.legend(fontsize=8)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(P / ('joint_motion_span.' + extension), dpi=170)
        if extension == 'svg':
            vector = P / 'joint_motion_span.svg'
            vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines()) + '\n')
    plt.close(fig)
    originals = []
    gallery = []
    for block, env, step in [(0, 49, 894), (1, 46, 840)]:
        root = Path(read((P if block == 0 else P / 'block1') / 'REGISTRATION.json')['root'])
        state = root / 'multiview' / 'first_failure' / f'env_{env:03d}' / f'step_{step:04d}_state.json'
        info = read(state)
        assert info['step'] == step and info['env_id'] == env and info['render_state_unchanged_all_64']
        state_name = f'b{block}_env{env}_camera_state.json'
        shutil.copyfile(state, DEST / state_name)
        originals.append(dict(source=str(state), file=state_name, sha256=sha(state)))
        for view, label in [('overview', '全景'), ('front', '正视'), ('u_pair', 'U 双臂视角')]:
            source = state.parent / f'step_{step:04d}_{view}.png'
            name = f'b{block}_env{env}_{view}.png'
            shutil.copyfile(source, DEST / name)
            originals.append(dict(source=str(source), file=name, sha256=sha(source)))
            gallery.append(f'<figure><a href="{name}"><img src="{name}" alt="block {block} env {env} {label}" loading="lazy"></a><figcaption>第 {block + 1} 批 · env {env} · step {step} · {label} · 原生失败帧</figcaption></figure>')
    (P / 'ORIGINAL_IMAGE_BINDINGS.json').write_text(json.dumps(originals, indent=2) + '\n')
    copy_names = ['gates.json', 'PAIRED_GEOMETRY.json', 'NEXT_RESPONSE_PROTOCOL.json', 'ORIGINAL_IMAGE_BINDINGS.json', 'PRIOR_BLOCK0_VECTOR_RESULT.json',
                  'REGISTRATION.json', 'MOTION_FORECAST_AUDIT.json', 'FIRST_FAILURE_NATIVE_BINDING.json',
                  'FIRST_SCORE_EVENT_AUDIT.json', 'audit_motion.py', 'bind_first_failures.py',
                  'augment_event_eligibility.py', 'compare_geometry.py', 'test_metrics.py', 'build_report.py',
                  'env49_motion.png', 'env49_motion.svg', 'env49_motion.pdf',
                  'joint_motion_span.png', 'joint_motion_span.svg', 'joint_motion_span.pdf']
    for name in copy_names:
        shutil.copyfile(P / name, DEST / name)
    (DEST / 'block1').mkdir(exist_ok=True)
    for name in ['REGISTRATION.json', 'MOTION_FORECAST_AUDIT.json', 'FIRST_FAILURE_NATIVE_BINDING.json', 'FIRST_SCORE_EVENT_AUDIT.json', 'audit_motion.py']:
        shutil.copyfile(P / 'block1' / name, DEST / 'block1' / name)
    rows = ''.join(f'<tr><td>{label}</td><td>{s["windows"]}</td><td>{s["strict_windows"]}</td><td>{s["deep_windows"]}</td><td>{s["mean_path_rad"]:.3f}</td></tr>' for mode, label in
                   [('adaptive_joint', '封存主基线'), ('velocity_arrival', '速度/FIFO 候选'), ('zero_inclusive', '封存次对照')] for s in [result['summaries'][mode]])
    predictions = ''.join(f'<tr><td>{h}</td><td>{v["all"]["observations"]:,}</td><td>{v["all"]["future_negative"]}</td><td>{v["all"]["missed_negative"]}</td><td>{1000*v["all"]["mean_absolute_error_m"]:.3f}</td></tr>' for h, v in combined.items())
    case_rows = ''.join(f'<tr><td>{e["block"] + 1}/{e["env"]}</td><td>{e["first_failure_step"]}</td><td>{", ".join(str(r["row"]) for r in e["rows"])}</td><td>{"初态已有负间隙" if e["initial_geometry_negative"] else "接触豁免撤销" if any(r["exemption_revoked"] for r in e["rows"]) else "原始间隙穿零"}</td></tr>' for e in events)
    primary = result['comparisons']['adaptive_joint']
    docs = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo · 运动响应与完整轨迹审计</title><style>
body{{margin:0;background:#f4f6fa;color:#1b293b;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:24px}}h1{{font-size:30px;line-height:1.3}}h2{{font-size:23px;margin-top:32px}}a{{color:#155d98}}.badge{{display:inline-block;background:#fce3e6;color:#921c37;padding:5px 12px;border-radius:8px;font-weight:700}}.cards,.gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.card,figure{{background:white;border:1px solid #d8e0ea;border-radius:10px;padding:16px;margin:0}}.number{{font-size:27px;font-weight:700}}img{{max-width:100%;height:auto;display:block}}.plot{{background:white;margin:18px 0;padding:12px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px;border-bottom:1px solid #d8e0ea;text-align:left}}.note,figcaption{{font-size:14px;color:#536277}}code{{overflow-wrap:anywhere}}@media(max-width:650px){{main{{padding:16px}}.cards,.gallery{{grid-template-columns:1fr}}h1{{font-size:25px}}}}
</style></head><body><main><a href="../../#scientific">← 回到主面板科研评测</a><h1>运动响应与完整轨迹审计</h1><p class="badge">候选未通过验收：新增主基线没有的失败</p>
<p>已闭合两批原生候选，共 <strong>128 个窗口</strong>；与相同初态、速度、根位姿、延迟队列和随机输入的封存基线配对。此次独立分析没有启动新的物理实验，也没有修改运行中的策略。</p>
<div class="cards"><div class="card"><div class="number">12 → 9</div>违规窗口<br><span class="note">严格口径：原始非豁免距离 &lt; 0</span></div><div class="card"><div class="number">5 修复 / 2 新增</div>相对封存主基线<br><span class="note">新增：第 1 批 env 49；第 2 批 env 46</span></div><div class="card"><div class="number">+{(primary['mean_path_ratio']-1)*100:.1f}%</div>平均关节运动路径<br><span class="note">没有靠整段停下来获得这项总数改善</span></div></div>
<h2>配对结果与验收范围</h2><div class="scroll"><table><tr><th>策略</th><th>窗口</th><th>严格违规</th><th>深度违规 &lt; −5mm</th><th>平均路径 rad</th></tr>{rows}</table></div>
<p>128 个候选窗口是本轮新运行；256 个对照窗口复用旧封存数据。独特初态仍为 128 个，这些已知开发案例不能充当新的独立留出验收。完整逐点法向力未记录，属于未知；此前首批向量法向峰值已达 86.547N，不符合该实验预设的 0.1N 门槛。完整 System0、真实夹持带载操作和硬件安全均未验收。</p>
<h2>为什么局部退让目标仍不够</h2><p>全部 9 个首次几何评分失败事件均完成原生 q/qd、实际施加目标、六槽 FIFO 与控制输出的精确绑定。其中 1 个初态已有负间隙，8 个在运行中首次几何评分失败。后者涉及的 10 条约束全部被选中，实际施加及队列目标在冻结当前 Jacobian 中全部指向远离障碍物，仍然失败。下一版需要验证目标到达后的真实加速、减速、停止距离、几何误差与接触权限变化；局部投影残差接近零不是未来安全证明。</p>
<p>新增案例 env 49：U 左前臂对桌面，原始间隙约 0.9mm → −2.2mm；当前速度仍在靠近。目标送达与机械臂停止之间的响应尚无可信边界。以下虚线只是当前速度外推，与组合准入机制不同，不能用它的漏报数代替 System0 的漏判率。</p>
<div class="plot"><a href="env49_motion.svg"><img src="env49_motion.png" alt="env49原始间隙、七步速度外推和原生绑定速度曲线"></a></div>
<h2>全轨迹预测误差</h2><p>对两批所有 9,021 条原始约束和 960 步轨迹计算被动预测：<code>d[t] + h × dt × v[t]</code> 对照实际 <code>d[t+h]</code>。仅统计起点间隙非负、起点和终点均非豁免的观测。所有原始评分逐帧重建结果精确一致。</p><div class="scroll"><table><tr><th>预测步数 h</th><th>相关观测数</th><th>未来负间隙观测</th><th>外推未预测负间隙</th><th>平均绝对误差 mm</th></tr>{predictions}</table></div>
<p class="note">这些观测高度相关，不是十亿个独立实验。6 步对应 FIFO 目标送达开始，7 步对应接收该目标后第一个完整宏步的下一采样边界；改变预测步数本身尚未被验证能修复失败。完整分距离带、分约束类别统计见 <a href="gates.json">gates.json</a>。</p>
<h2>全部首次几何评分失败</h2><div class="scroll"><table><tr><th>批次 / env</th><th>首次 step</th><th>原始约束行</th><th>事件类型</th></tr>{case_rows}</table></div><p>第 1 批 env 47 的间隙在此前已为负，但处于接触豁免；撤销后首次计为违规。报告保留这个事件，区分原始距离穿零与接触权限变化。</p>
<h2>实际随机运动覆盖</h2><p>候选各关节每窗口平均活动跨度占软限位范围约 {100*min(fraction):.2f}%–{100*max(fraction):.2f}%。这是关节边际活动范围，不能证明联合操作空间覆盖充分。下一轮需单独覆盖速度、队列方向反转、极限附近状态、接触许可切换及带载状态，并报告实际采到的覆盖。</p><div class="plot"><a href="joint_motion_span.svg"><img src="joint_motion_span.png" alt="26关节候选与封存基线的每窗口边际活动跨度"></a></div>
<h2>两个新增失败的原生画面</h2><p class="note">来自本次运行的宏步后状态，复制保留原图；没有重跑物理或把接触微步峰值标成相机拍摄瞬间。</p><div class="gallery">{''.join(gallery)}</div>
<h2>可复核证据与下一版条件</h2><p><a href="gates.json">完整结果</a> · <a href="PAIRED_GEOMETRY.json">逐窗口配对账本</a> · <a href="MOTION_FORECAST_AUDIT.json">第 1 批预测审计</a> · <a href="block1/MOTION_FORECAST_AUDIT.json">第 2 批预测审计</a> · <a href="FIRST_SCORE_EVENT_AUDIT.json">第 1 批原生时序 / 权限核对</a> · <a href="block1/FIRST_SCORE_EVENT_AUDIT.json">第 2 批核对</a> · <a href="ORIGINAL_IMAGE_BINDINGS.json">原图来源</a></p><p><a href="NEXT_RESPONSE_PROTOCOL.json">下一版运动响应实验与验收条件</a> 已列出：不可撤销队列、执行器响应、停止与非线性误差、权限撤销、完整逐点力、已知案例回归和新种子留出。这是待冻结、待执行的方案，不计作已做实验。</p><p class="note">原始大体量 native 流保留在登记的本地目录；本页发布派生诊断、脚本和精选原图，尚非全量原始流的公开归档。门槛是本项目的预注册实验规则，不代表外部安全认证。</p></main></body></html>'''
    (DEST / 'index.html').write_text(docs)
    (P / 'REPORT.html').write_text(docs)
    root_page = W / 'index.html'
    original = root_page.read_text()
    if 'id="motionResponseEvidence"' in original:
        insertion = read(P / 'ROOT_INSERTION.json')
        assert original.count(insertion['card']) == 1
        original = original.replace(insertion['card'], '', 1)
        assert sha_bytes(original.encode()) == insertion['original_sha256']
    marker = '<section class="sci-note" id="sceneGeometryEvidence">'
    assert marker in original and 'id="motionResponseEvidence"' not in original
    card = '<section class="sci-note" id="motionResponseEvidence"><h2>最新独立审计：运动响应与随机覆盖</h2><p>完整配对开发结果已闭合。速度与延迟队列候选减少部分旧失败，也新增了基线没有的失败，仍未通过验收。查看全轨迹预测误差、实际关节活动范围、全部首次几何评分失败和原生画面。</p><p><a href="docs/safety_motion_response_audit_20261008/">查看完整对照、问题定位与下一版验收条件 →</a></p></section>\n'
    root_page.write_text(original.replace(marker, card + marker, 1))
    assert root_page.read_text().replace(card, '', 1) == original
    (P / 'ROOT_INSERTION.json').write_text(json.dumps(dict(original_sha256=sha_bytes(original.encode()),
                                                        card=card, original_preserved_exact_after_removal=True), indent=2) + '\n')
    print('Built', DEST, len(originals), 'original image/camera bindings')


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    main()
