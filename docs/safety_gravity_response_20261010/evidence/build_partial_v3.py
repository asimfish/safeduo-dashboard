"""Publish measured short128/long64; never count the64 resource rejects as safe."""
import csv
import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parent
H = P.parent
W = Path('/home/liyufeng/safeduo-dashboard-response-probe-20261009')
G, L, Q = [H / s for s in ['gravity_hold128_v1', 'gravity_long128_v1', 'queue_response128_v2']]
SLUG = 'docs/safety_gravity_response_20261010'
OUT = W / SLUG


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, d): p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def load(p):
    with np.load(p) as z: return {k: z[k] for k in z.files}


def export(fig, name):
    def signature(blob):
        def sig(x):
            return (x.tag, sorted((k, ' '.join(v.split())) for k, v in x.attrib.items()),
                    ' '.join((x.text or '').split()), ' '.join((x.tail or '').split()), [sig(c) for c in x])
        return sig(ET.fromstring(blob))
    for ext in ['png', 'svg', 'pdf']:
        path = OUT / 'plots' / (name + '.' + ext)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        if ext == 'svg':
            old = path.read_bytes(); new = b'\n'.join(line.rstrip(b' \t') for line in old.split(b'\n'))
            assert signature(old) == signature(new)
            path.write_bytes(new)
    plt.close(fig)


def main():
    result = dict(short=read(G / 'GRAVITY_HOLD128_RESULT_V1.json'), long=read(L / 'GRAVITY_LONG_FIRST64_RESULT_V1.json'))
    finish = read(G / 'FINISH_EXECUTION_V1.json')
    assert finish['status'] == 'complete' and all(s['actual_exit'] == 0 for s in finish['steps'])
    assert read(L / 'PARTIAL_AUDIT_ACTUAL_EXIT_V1.json')['actual_exit'] == 0
    assert result['long']['inputs_completed'] == 64 and result['long']['remaining_inputs_not_run'] == 64
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W)
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=W).split(b'\0')
    preserved = {x.decode(): sha(W / x.decode()) for x in tracked if x and x != b'index.html'}
    write(P / 'PRESERVED_BEFORE_BUILD_V1.json', preserved)
    OUT.mkdir(exist_ok=False); (OUT / 'plots').mkdir(); (OUT / 'evidence').mkdir()
    sources = []

    def copy(source, name):
        target = OUT / 'evidence' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); assert sha(target) == sha(source)
        sources.append(dict(source=str(source), path='evidence/' + name, sha256=sha(target), bytes=target.stat().st_size))

    groups = [('short', G, 2, ['off', 'on']), ('long', L, 1, ['on'])]
    for label, folder, batches, modes in groups:
        for source in sorted(folder.glob('*.py')): copy(source, label + '/' + source.name)
        names = ['REGISTRATION_V1.json', 'FINISH_EXECUTION_V1.json', 'PREFLIGHT_GATE_V1.json']
        names += ['GRAVITY_HOLD128_RESULT_V1.json', 'OFF_BASELINE_REPRODUCTION_V1.json'] if label == 'short' else ['GRAVITY_LONG_FIRST64_RESULT_V1.json', 'PARTIAL_AUDIT_ACTUAL_EXIT_V1.json']
        for name in names: copy(folder / name, label + '/' + name)
        native_inventory = []
        for batch in range(batches):
            for mode in modes:
                for name in [f'GRAVITY_ORACLE_batch{batch}_{mode}_V1.json', f'PAIRED_METRICS_batch{batch}_{mode}_V1.npz']:
                    copy(folder / name, label + '/' + name)
                leaf = Path(read(folder / 'REGISTRATION_V1.json')['raw']) / f'paired_batch{batch}_{mode}_v8'
                for source in sorted(s for s in leaf.rglob('*') if s.is_file()):
                    native_inventory.append(dict(path=str(source), sha256=sha(source), bytes=source.stat().st_size))
                for name in ['response_protocol.json', 'actual_solver_roots.json', 'actual_body_gravity_properties.json',
                             'resolved_native_parameters.npz', 'gravity_feedforward_events.npz', 'response_stream.npz',
                             'all_raw_geometry_receipts.json', 'point_contact_receipts.json']:
                    copy(leaf / name, f'{label}/native_batch{batch}_{mode}/{name}')
        write(OUT / f'evidence/{label}/LOCAL_FULL_NATIVE_INVENTORY_V1.json', dict(
            files=native_inventory, total_files=len(native_inventory), raw_geometry_and_point_archives_public=False,
            selected_native_readbacks_public=True, fullSystem0_accepted=False))
    for name in ['PAIRED_METRICS_batch0_multirow_hold_fallback_V1.npz', 'STRONG_ORACLE_batch0_multirow_hold_fallback_V1.json']:
        copy(H / 'strong_long128_v1' / name, 'long_baseline/' + name)
    for source in [H / 'gravity_long128_v1_execution.json', H / 'gravity_long128_v1_batch1_on_resource_gate.json',
                   Q / 'REGISTRATION_V1.json', Q / 'PREFLIGHT_GATE_V1.json', Q / 'MODEL_BOX_DIAGNOSTIC_V1.json',
                   Q / 'common_bounded_goal_tape.npz', H / 'queue_response128_v1/PREPARATION_FAILURE_V1.json']:
        copy(source, source.name if source.parent != Q else 'registered_motion/' + source.name)
    for source in sorted(Q.glob('*.py')): copy(source, 'registered_motion/' + source.name)
    copy(Path(__file__), 'build_partial_v3.py')
    matrix = read(H / 'full_requirements_v1/EVIDENCE_MATRIX_V1.json')
    copy(H / 'full_requirements_v1/EVIDENCE_MATRIX_V1.json', 'EVIDENCE_MATRIX_V1.json')
    matrix.update(schema='safeduo.full_requirements.evidence_matrix.v2', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    matrix['rows'][0].update(status='PASS_FIRST64_LONG_REMAINING64_RESOURCE_NOCHILD',
        measured='短程开启128/128；首64条长程关闭60/64、开启64/64；4条救回，新增失败0',
        missing='剩余64条因显存门槛没有启动；完整长程、正式留出、未来目标保持和有效操作仍待验收')
    matrix['rows'][1]['measured'] += '；新队列动作候选只登记，没有原生动作结果'
    matrix['rows'][-1].update(measured='本页发布新128短程、64长程和9项完整验收缺口，无本轮新视频', status='NUMERIC_REPORT_UPDATED_TASK_VIDEO_PENDING')
    write(P / 'EVIDENCE_MATRIX_V2.json', matrix); write(OUT / 'evidence/EVIDENCE_MATRIX_V2.json', matrix)
    write(OUT / 'data.json', result)
    write(OUT / 'evidence/BUNDLE_RECEIPT_V1.json', dict(sources=sources, actual_960_microstep_long_oracle=True,
        long_inputs=64, remaining64_not_run=True, all128_long_complete=False, raw_full_native_archives_public=False,
        original_velocities_and_FIFO_preserved=True, no_new_video=True, fullSystem0_accepted=False))
    fields = ['horizon', 'mode', 'input_id', 'cell', 'primary_failure', 'geometry_failure', 'force_failure',
              'minimum_raw_gap_m', 'peak_all_arm_scalar_N', 'controlled_path_rad',
              'supplementary_max_all74_hard_violation_rad', 'supplementary_max_all74_velocity_exceedance_rad_s']
    with (OUT / 'all384_hold_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore', lineterminator='\n'); writer.writeheader()
        for hz in result:
            for mode, rows in result[hz]['states'].items():
                for row in rows: writer.writerow(dict(horizon=hz, mode=mode, **row))
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    counts = [result[h]['counts'][m] for h in ['short', 'long'] for m in ['off', 'on']]
    names = ['16 controls\nOff / 128', '16 controls\nOn / 128', '480 controls\nOff / 64', '480 controls\nOn / 64']
    colours = ['#c74243', '#17785b'] * 2
    for ax, key, ylabel, title in [(axes[0], 'joint_and_primary_pass', 'Passed inputs', 'Full gates; denominators shown below'),
                                   (axes[1], 'peak_normal_N', 'Peak scalar normal force [N]', 'Same original velocities and FIFO')]:
        values = [c[key] for c in counts]; ax.bar(names, values, color=colours)
        ax.set(ylabel=ylabel, title=title)
        for i, value in enumerate(values): ax.text(i, value + 1, f'{value:.4g}', ha='center')
        ax.set_ylim(0, max(values) * 1.15 + 1)
    export(fig, 'holding_summary')
    data_by_horizon = {hz: {m: {} for m in ['off', 'on']} for hz in result}
    for hz, folder, batches, modes in groups:
        for batch in range(batches):
            for mode in ['off', 'on']:
                source = H / 'strong_long128_v1' / f'PAIRED_METRICS_batch{batch}_multirow_hold_fallback_V1.npz' if hz == 'long' and mode == 'off' else folder / f'PAIRED_METRICS_batch{batch}_{mode}_V1.npz'
                data = load(source)
                for lane, input_id in enumerate(data['global_input_id']):
                    data_by_horizon[hz][mode][int(input_id)] = {k: data[k][:, lane] for k in ['gap_m', 'force_N', 'hard_violation_rad', 'controlled_q']}
    bindings = []
    for input_id in result['short']['paired_changes']['full74_rescues']:
        hz = 'long' if input_id in data_by_horizon['long']['on'] else 'short'; controls = 480 if hz == 'long' else 16
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
        for mode, colour in zip(['off', 'on'], colours[:2]):
            d = data_by_horizon[hz][mode][input_id]; time = (np.arange(controls * 2) + 1) * .008333
            axes[0].plot(time, d['gap_m'] * 1000, color=colour, label='No feedforward' if mode == 'off' else 'Gravity feedforward')
            axes[1].plot(time, d['force_N'], color=colour); axes[2].plot(time, d['hard_violation_rad'], color=colour)
            axes[3].plot(np.arange(controls + 1) * .016666, abs(d['controlled_q'] - d['controlled_q'][0]).max(-1), color=colour)
        axes[0].axhline(0, color='#222', ls='--', lw=.7); axes[0].legend(loc='upper right')
        axes[0].set(ylabel='Minimum9021 gap [mm]', title=f'Input{input_id}: measured{controls} controls; held positions')
        axes[1].set(ylabel='Scalar normal [N]'); axes[2].set(ylabel='Max74 hard breach [rad]')
        axes[2].axhline(1e-5, color='#222', ls='--', lw=.7)
        axes[3].set(ylabel='Max26 departure [rad]', xlabel='Native simulation time [s]')
        export(fig, 'input_' + str(input_id))
        bindings.append(dict(input_id=input_id, controls=controls, plot=f'plots/input_{input_id}.png',
            note=f'输入 {input_id}：补偿后{controls}周期通过。首64条有长程，剩64条长程未测；位置收敛不是操作成功。'))
    assert len(bindings) == 10; write(P / 'PUBLIC_CASE_BINDINGS.json', bindings)
    default = next(c for c in bindings if c['input_id'] == 1887)
    count_rows = ''.join(f'<tr><td>{h}</td><td>{m}</td><td>{c["joint_and_primary_pass"]}/{c["inputs"]}</td><td>{c["geometry_failures"]}</td><td>{c["force_failures"]}</td><td>{c["all74_hard_bad"]}</td><td>{c["all74_speed_bad"]}</td></tr>' for h in ['short', 'long'] for m, c in result[h]['counts'].items())
    matrix_rows = ''.join('<tr>' + ''.join('<td>' + html.escape(row[k]) + '</td>' for k in ['requirement', 'status', 'measured', 'missing']) + '</tr>' for row in matrix['rows'])
    long_map = {m: {r['input_id']: r for r in result['long']['states'][m]} for m in ['off', 'on']}
    def verdict(r):
        if r is None: return '未测试'
        return '失败' if r['primary_failure'] or r['supplementary_max_all74_hard_violation_rad'] > 1e-5 or r['supplementary_max_all74_velocity_exceedance_rad_s'] > 1e-5 else '通过'
    rows = []
    for a, b in zip(result['short']['states']['off'], result['short']['states']['on']):
        assert a['input_id'] == b['input_id']
        pair = [a, b, long_map['off'].get(a['input_id']), long_map['on'].get(a['input_id'])]
        rows.append(f'<tr><td>{a["input_id"]}</td><td>{a["cell"]}</td>' + ''.join('<td>' + verdict(r) + '</td>' for r in pair) + '</tr>')
    options = ''.join(f'<option value="{c["input_id"]}" {"selected" if c["input_id"]==1887 else ""}>{c["input_id"]} · {c["controls"]}周期</option>' for c in bindings)
    case_links = ''.join(f'<li>输入{c["input_id"]}，{c["controls"]}周期：<a href="{c["plot"]}">PNG</a> · <a href="{c["plot"].replace(".png", ".svg")}">SVG</a> · <a href="{c["plot"].replace(".png", ".pdf")}">PDF</a></li>' for c in bindings)
    links = ''.join(f'<li><a href="{f.relative_to(OUT)}">{f.relative_to(OUT)}</a></li>' for f in sorted((OUT / 'evidence').rglob('*')) if f.is_file())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo 重力补偿与完整验收</title><style>
body{{margin:0;background:#f5f7fa;color:#192b3d;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:28px 18px}}h1{{font-size:30px}}h2{{margin-top:32px}}a,li{{overflow-wrap:anywhere}}a{{color:#13588d}}.box{{background:white;padding:20px;border-radius:12px;margin:20px 0;border:1px solid #dce4ec}}.good{{border-left:5px solid #17785b}}.pending{{border-left:5px solid #ba7020}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;vertical-align:top;border-bottom:1px solid #dce4ec;padding:9px;overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}select{{font:inherit;max-width:100%}}</style>
<main><a href="../../">← 返回主面板</a><h1>重力补偿、首64条长程与完整验收缺口</h1><p>2026-10-10 · 原始初态速度与位置延迟6周期保留 · 已知开发集</p>
<div class="box good"><b>保持机制有实测改善：</b>短程128条，关闭补偿118/128、开启128/128，救回全部10条失败。首64条长程，关闭60/64、开启64/64，救回4条失败；新增失败0。额外力矩在每个物理子步与原生读回一致。</div>
<div class="box pending"><b>完整 System 0 安全验收未通过。</b>剩余64条长程被显存门槛挡在启动前，不能计为通过。保持收敛不是任务成功；选择性暂停/恢复、真实夹持、四臂共同操作和独立留出仍待验收。</div>
<h2>相同输入的保持对照</h2><div class="scroll"><table id="counts"><thead><tr><th>时长</th><th>补偿</th><th>联合通过</th><th>几何失败</th><th>接触失败</th><th>限位失败</th><th>速度失败</th></tr></thead><tbody>{count_rows}</tbody></table></div><img src="plots/holding_summary.png" alt="不同分母的短长程保持通过数和接触峰值">
<p>短程16控制周期约0.267秒；首64条长程480周期约8秒。独立审计覆盖每个物理子步的全部9021球对、82碰撞刚体接触归因和74关节硬限位/速度。短程关闭组逐位复现原长程前32子步；首64条新长程开启组逐位复现已有开启短程。不是全网格、摩擦或硬件安全认证。</p>
<h2>未执行部分和动作机制候选</h2><p>第一批原生子进程退出0。整体计划在第二批启动前退出1：GPU空闲17433MiB，小于预登记20480MiB；第二批没有子进程。新队列动作候选已冻结，共用保留目标方向的0.02rad步幅带、原生补偿和FIFO，围绕末端排队目标投影并计入额外补偿。目前仅有CPU名义约束诊断，原生动作结果为0条；完整128长程通过是其执行依赖，没有绕过。</p><p><a href="evidence/gravity_long128_v1_batch1_on_resource_gate.json">资源拒绝回执</a> · <a href="evidence/gravity_long128_v1_execution.json">实际执行记录</a> · <a href="evidence/registered_motion/REGISTRATION_V1.json">动作候选登记</a></p>
<h2>全部10条原保持失败的对照</h2><label for="case">输入： </label><select id="case">{options}</select><p id="case-note">{default['note']}</p><img id="case-plot" src="{default['plot']}" alt="记录时长明确的补偿关闭与开启原生曲线"><p><a id="case-svg" href="{default['plot'].replace('.png','.svg')}">下载SVG</a> · <a id="case-pdf" href="{default['plot'].replace('.png','.pdf')}">下载PDF</a></p><details><summary>全部10个对照，无JavaScript也可查看</summary><ul id="failure-cases">{case_links}</ul></details>
<h2>完整要求与验收状态</h2><p>以下9项共同验收，保持实验通过不会覆盖其他失败。32000个全软限位提案，经几何和物理筛选的128输入不代表全空间。手指动作、布局、对象、感知、延迟和故障覆盖仍有缺口。</p><div class="scroll"><table id="matrix"><thead><tr><th>要求</th><th>状态</th><th>实测</th><th>待通过内容</th></tr></thead><tbody>{matrix_rows}</tbody></table></div>
<p>旧物体任务是F双臂与U双臂各搬一物体的并行操作；两方案各1/4满足旧门槛，但通过槽位的U抬升仅发生在退离阶段，没有指定双手同时接触，不能算稳定夹持。正式100mm/2s夹持和四手共持或交接尚未通过。<a href="../safety_object_binding_20261006/">旧物体证据</a>。当前SafeDuo v7与NAS计划中的JAKA实际资产profile不同，本页不作该profile晋级。</p>
<details id="all-results"><summary>全部128输入：短程完整，长程未测明确列出</summary><p><a href="all384_hold_results.csv">下载384行已测结果CSV</a> · <a href="data.json">完整结果JSON</a></p><div class="scroll"><table><thead><tr><th>输入</th><th>组合</th><th>短关</th><th>短开</th><th>长关(匹配子集)</th><th>长开</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>
<h2>证据与画面范围</h2><p>公开逐子步指标、原生额外力矩读回、参数、运行和独审代码、输入判定与全部本地原始文件SHA清单。9021逐对大数组及原生点接触块完整保存在本地，未全部公开；仅下载本页无法复算所有原始接触/几何。本轮没有新原生视频。<a href="../safety_physical_random_128_20261009/">旧随机四视角视频</a>保持原实验标签。</p><details><summary>证据下载</summary><ul>{links}</ul></details></main>
<script>const cases={json.dumps(bindings,ensure_ascii=False)};document.querySelector('#case').addEventListener('change',e=>{{const c=cases.find(x=>String(x.input_id)===e.target.value);document.querySelector('#case-plot').src=c.plot;document.querySelector('#case-svg').href=c.plot.replace('.png','.svg');document.querySelector('#case-pdf').href=c.plot.replace('.png','.pdf');document.querySelector('#case-note').textContent=c.note;}});</script></html>'''
    (OUT / 'index.html').write_text(page)
    mainpath = W / 'index.html'; main = mainpath.read_text(); marker = '<div class="sci-note" id="holdFactorialEvidence">'
    assert main.count(marker) == 1 and 'id="gravityResponseEvidence"' not in main
    card = f'<div class="sci-note" id="gravityResponseEvidence"><b>10-10 最新：补偿短程128/128、首64长程64/64；完整验收未通过</b><p>原始速度和6周期延迟保留，保持失败有实测修复。剩64条长程被显存门槛挡在启动前，动作候选尚未执行；9项完整验收缺口、未测状态与全部失败单列。保持通过不等于操作成功。</p><a href="{SLUG}/">查看最新机制、实测效果与完整验收矩阵 →</a></div>\n'
    mainpath.write_text(main.replace(marker, card + marker))
    assert all(sha(W / name) == value for name, value in preserved.items())
    payload = [mainpath] + sorted(f for f in OUT.rglob('*') if f.is_file())
    write(P / 'PUBLIC_PAYLOAD_MANIFEST.json', dict(slug=SLUG, base_commit=base,
        files={str(f.relative_to(W)): sha(f) for f in payload}, total_files=len(payload), total_bytes=sum(f.stat().st_size for f in payload), fullSystem0_accepted=False))
    print('BUILT_PARTIAL_GRAVITY_RESPONSE', len(payload), sum(f.stat().st_size for f in payload), flush=True)


if __name__ == '__main__': main()
