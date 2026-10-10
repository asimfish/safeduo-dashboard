"""Build the closed gravity-support evidence and explicit full-requirement gaps."""
import csv
import datetime
import gzip
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
SLUG = 'docs/safety_gravity_complete_20261010'
OUT = W / SLUG
G = H / 'gravity_hold128_v1'
L = H / 'gravity_long128_v1'
Q = H / 'queue_response128_v2'
LABELS = {'off': 'No feedforward', 'on': 'Gravity feedforward'}
COLOURS = {'off': '#c74243', 'on': '#17785b'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def export(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        target = OUT / 'plots' / (name + '.' + ext)
        fig.savefig(target, dpi=160, bbox_inches='tight')
        if ext == 'svg':
            before = target.read_bytes()
            after = b'\n'.join(line.rstrip(b' \t') for line in before.split(b'\n'))
            def signature(blob):
                node = ET.fromstring(blob)
                def sig(x):
                    return (x.tag, sorted((k, ' '.join(v.split())) for k, v in x.attrib.items()),
                        ' '.join((x.text or '').split()), ' '.join((x.tail or '').split()), [sig(c) for c in x])
                return sig(node)
            assert signature(before) == signature(after)
            target.write_bytes(after)
    plt.close(fig)


def main():
    result = {name: read(folder / file) for name, folder, file in [
        ('short', G, 'GRAVITY_HOLD128_RESULT_V1.json'), ('long', L, 'GRAVITY_LONG128_RESULT_V1.json'),
        ('motion', Q, 'QUEUE_RESPONSE128_RESULT_V1.json')]}
    for folder, filename in [(G, 'FINISH_EXECUTION_V1.json'), (H / 'gravity_long_completion_v3', 'FINISH_EXECUTION_V1.json'), (Q, 'FINISH_EXECUTION_V6.json')]:
        finish = read(folder / filename)
        assert finish['status'] == 'complete' and all(s['actual_exit'] == 0 for s in finish['steps'])
    assert result['long']['long_hold_candidate_pass']
    assert all(not r['fullSystem0_accepted'] for r in result.values())
    command_effect = read(Q / 'COMMAND_EFFECT_DESCRIPTION_V1.json')
    assert command_effect['first6_physical_controls_bitexact_with_same_gravity_holding']
    assert not command_effect['full_system0_accepted']
    result['command_effect_description'] = command_effect
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W)
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=W).split(b'\0')
    preservation = {n.decode(): sha(W / n.decode()) for n in tracked if n and n != b'index.html'}
    write(P / 'PRESERVED_BEFORE_BUILD_V1.json', preservation)
    OUT.mkdir(exist_ok=False)
    (OUT / 'plots').mkdir()
    (OUT / 'evidence').mkdir()
    sources = []

    def copy(source, relative):
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(target) == sha(source)
        sources.append(dict(source=str(source), path=relative, sha256=sha(source), bytes=source.stat().st_size))

    for group, folder, prefix, modes, batches in [
        ('short', G, 'GRAVITY_HOLD128', ['off', 'on'], 2),
        ('long', L, 'GRAVITY_LONG128', ['on'], 2),
        ('motion', Q, 'QUEUE_RESPONSE128', ['bounded_raw', 'queue_projection'], 2)]:
        for source in sorted(folder.glob('*.py')):
            copy(source, f'evidence/{group}/{source.name}')
        for name in ['REGISTRATION_V1.json', prefix + '_RESULT_V1.json', 'FINISH_EXECUTION_V1.json', 'PREFLIGHT_GATE_V1.json']:
            source = folder / name
            if name == 'FINISH_EXECUTION_V1.json' and group == 'long':
                source = H / 'gravity_long_completion_v3/FINISH_EXECUTION_V1.json'
            if name == 'FINISH_EXECUTION_V1.json' and group == 'motion':
                source = Q / 'FINISH_EXECUTION_V6.json'
            copy(source, f'evidence/{group}/{name}')
        for batch in range(batches):
            for mode in modes:
                for name in [f'GRAVITY_ORACLE_batch{batch}_{mode}_V1.json', f'PAIRED_METRICS_batch{batch}_{mode}_V1.npz']:
                    copy(folder / name, f'evidence/{group}/{name}')
        raw = Path(read(folder / 'REGISTRATION_V1.json')['raw'])
        native_inventory = []
        for batch in range(batches):
            for mode in modes:
                leaf = raw / f'paired_batch{batch}_{mode}_v8'
                if group == 'long' and batch == 1:
                    leaf = Path(read(H / 'gravity_long_completion_v3/RECOVERY_REGISTRATION_V1.json')['raw']) / f'paired_batch{batch}_{mode}_v8'
                files = sorted(s for s in leaf.rglob('*') if s.is_file())
                for source in files:
                    native_inventory.append(dict(path=str(source), sha256=sha(source), bytes=source.stat().st_size))
                for name in ['response_protocol.json', 'actual_body_gravity_properties.json', 'actual_solver_roots.json',
                             'resolved_native_parameters.npz', 'gravity_feedforward_events.npz',
                             'all_raw_geometry_receipts.json', 'point_contact_receipts.json', 'response_stream.npz']:
                    copy(leaf / name, f'evidence/{group}/native_batch{batch}_{mode}/{name}')
        write(OUT / f'evidence/{group}/LOCAL_FULL_NATIVE_INVENTORY_V1.json', dict(
            status='COMPLETE_LOCAL_NATIVE_INVENTORY', files=native_inventory, total_files=len(native_inventory),
            total_bytes=sum(x['bytes'] for x in native_inventory), all_raw_geometry_public=False,
            raw_point_archives_public=False, selected_native_readbacks_public=True, fullSystem0_accepted=False))
    for batch in range(2):
        for name in [f'PAIRED_METRICS_batch{batch}_multirow_hold_fallback_V1.npz', f'STRONG_ORACLE_batch{batch}_multirow_hold_fallback_V1.json']:
            copy(H / 'strong_long128_v1' / name, f'evidence/long_baseline/{name}')
    for name in ['OFF_BASELINE_REPRODUCTION_V1.json']:
        copy(G / name, 'evidence/short/' + name)
    for source in [Q / 'MODEL_BOX_DIAGNOSTIC_V1.json', Q / 'common_bounded_goal_tape.npz',
                   H / 'queue_response128_v1/PREPARATION_FAILURE_V1.json', H / 'full_requirements_v1/EVIDENCE_MATRIX_V1.json']:
        copy(source, 'evidence/' + source.name)
    copy(Q / 'COMMAND_EFFECT_DESCRIPTION_V1.json', 'evidence/motion/COMMAND_EFFECT_DESCRIPTION_V1.json')
    for folder_name in ['grasp_sequence_diagnosis_v1', 'payload_feedback_v1']:
        for source in sorted((H / folder_name).iterdir()):
            if source.suffix in ['.json', '.py']:
                copy(source, f'evidence/{folder_name}/{source.name}')
    copy(H / 'PAYLOAD_FEEDBACK_READINESS_V1.json', 'evidence/PAYLOAD_FEEDBACK_READINESS_V1.json')
    for source in [H / 'gravity_long128_v1_execution.json', H / 'gravity_long_completion_v2_execution.json',
                   H / 'gravity_long_completion_v3_execution.json', H / 'queue_response128_v4_execution.json', H / 'queue_response128_v6_execution.json', H / 'queue_response128_v7_execution.json',
                   H / 'GRAVITY_LONG_COMPLETION_PLAN_V3.json', H / 'QUEUE_RESPONSE128_PLAN_V4.json', H / 'QUEUE_RESPONSE128_PLAN_V5.json', H / 'QUEUE_RESPONSE128_PLAN_V6.json', H / 'QUEUE_RESPONSE128_PLAN_V7.json']:
        copy(source, 'evidence/execution_history/' + source.name)
    for folder_name in ['gravity_long_completion_v2', 'gravity_long_completion_v3']:
        for source in sorted((H / folder_name).iterdir()):
            if source.suffix in ['.json', '.py']:
                copy(source, f'evidence/execution_history/{folder_name}/{source.name}')
    copy(Path(__file__), 'evidence/build_report_v1.py')

    matrix = read(H / 'full_requirements_v1/EVIDENCE_MATRIX_V1.json')
    matrix['schema'] = 'safeduo.full_requirements.evidence_matrix.v2'
    matrix['created_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    matrix['rows'][0].update(status='PASS_LONG_DEVELOPMENT_PENDING_HOLDOUT',
        evidence='evidence/long/GRAVITY_LONG128_RESULT_V1.json',
        measured='同128输入、480控制周期/960子步：关闭118/128，重力补偿128/128；10条救回、新增失败0',
        missing='正式独立留出、当前速度下未来目标保持、有效操作仍待验收')
    matrix['rows'][1]['measured'] += '；新候选2048/2048回退、模型满足0，实际轨迹与保持相同，四臂命令诱发运动0/128；有效操作明确失败'
    diagnosis = read(H / 'grasp_sequence_diagnosis_v1/RECORDED_TASK_DIAGNOSIS_V1.json')
    assert diagnosis['tallies']['descriptive_grasp_2s_objects'] == 0
    for row in matrix['rows']:
        if '夹持' in row['requirement'] or '抓持' in row['requirement']:
            row['measured'] += '；旧16物体记录连续100mm/2s且姿态/双手接触/离桌组合0/16，张手后拇指仍接触的故障已定位；任务观察器9项软件测试通过，原生闭环控制待验证'
    matrix['rows'][-1].update(status='PASS_THIS_NUMERIC_REPORT_TASK_VIDEO_PENDING',
        measured='本页发布新的保持长短程、队列动作实测和9项完整要求矩阵；没有本轮新原生视频',
        missing='四臂协作操作的连续三视角视频与完整状态/接触/干预/任务侧车')
    write(OUT / 'evidence/EVIDENCE_MATRIX_V2.json', matrix)
    write(P / 'EVIDENCE_MATRIX_V2.json', matrix)
    write(OUT / 'data.json', result)
    write(OUT / 'evidence/BUNDLE_RECEIPT_V1.json', dict(sources=sources,
        all128_denominators_retained=True, actual_960_microstep_long_oracle=True,
        original_velocities_and_FIFO_preserved=True, raw_full_native_archives_public=False,
        fullSystem0_accepted=False, no_new_video=True))

    with (OUT / 'all512_hold_results.csv').open('w', newline='') as stream:
        fields = ['horizon', 'mode', 'input_id', 'cell', 'primary_failure', 'geometry_failure', 'force_failure',
                  'minimum_raw_gap_m', 'peak_all_arm_scalar_N', 'controlled_path_rad',
                  'supplementary_max_all74_hard_violation_rad', 'supplementary_max_all74_velocity_exceedance_rad_s']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for horizon in ['short', 'long']:
            for mode, rows in result[horizon]['states'].items():
                for row in rows:
                    writer.writerow(dict(horizon=horizon, mode=mode, **row))
    with (OUT / 'all256_motion_results.csv').open('w', newline='') as stream:
        fields = ['mode', 'input_id', 'cell', 'primary_failure', 'geometry_failure', 'force_failure',
                  'minimum_raw_gap_m', 'peak_all_arm_scalar_N', 'controlled_path_rad', 'four_arms_moved',
                  'supplementary_max_all74_hard_violation_rad', 'supplementary_max_all74_velocity_exceedance_rad_s']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for mode, rows in result['motion']['states'].items():
            for row in rows:
                writer.writerow(dict(mode=mode, **row))

    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    names, passes, forces = [], [], []
    for horizon, hzlabel in [('short', '16 controls'), ('long', '480 controls')]:
        for mode in ['off', 'on']:
            names.append(hzlabel + '\n' + LABELS[mode]); passes.append(result[horizon]['counts'][mode]['joint_and_primary_pass']); forces.append(result[horizon]['counts'][mode]['peak_normal_N'])
    ax[0].bar(np.arange(4), passes, color=[COLOURS[m] for m in ['off', 'on', 'off', 'on']])
    ax[0].set(xticks=np.arange(4), xticklabels=names, ylim=(0, 140), ylabel='Passed inputs / 128', title='All geometry, force, hard and speed gates')
    for i, n in enumerate(passes): ax[0].text(i, n + 2, str(n), ha='center')
    ax[1].bar(np.arange(4), forces, color=[COLOURS[m] for m in ['off', 'on', 'off', 'on']])
    ax[1].set(xticks=np.arange(4), xticklabels=names, ylabel='Peak attributed scalar normal force [N]', title='Same original states; position FIFO = 6')
    export(fig, 'holding_summary')
    mcounts = result['motion']['counts']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    modes = ['bounded_raw', 'queue_projection']
    names = ['Bounded raw', 'Queue projection']
    axes[0].bar(names, [mcounts[m]['joint_and_primary_pass'] for m in modes], color=['#c74243', '#326ca8'])
    axes[0].set(ylabel='Passed inputs / 128', ylim=(0, 140), title='New16-control development probe')
    axes[1].bar(names, [mcounts[m]['total_controlled_path_rad'] for m in modes], color=['#c74243', '#326ca8'])
    axes[1].set(ylabel='Sum of measured26-joint paths [rad]', title='Includes initial inertia; no manipulation task')
    export(fig, 'motion_summary')

    metrics = {'off': {}, 'on': {}}
    for batch in range(2):
        for mode in metrics:
            path = (H / 'strong_long128_v1' / f'PAIRED_METRICS_batch{batch}_multirow_hold_fallback_V1.npz') if mode == 'off' else L / f'PAIRED_METRICS_batch{batch}_on_V1.npz'
            data = load(path)
            for lane, input_id in enumerate(data['global_input_id']):
                metrics[mode][int(input_id)] = {k: data[k][:, lane] for k in ['gap_m', 'force_N', 'hard_violation_rad', 'velocity_exceedance_rad_s', 'controlled_q']}
    failures = result['short']['paired_changes']['full74_rescues']
    assert len(failures) == 10
    bindings = []
    for input_id in failures:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
        for mode in ['off', 'on']:
            d = metrics[mode][input_id]
            t = (np.arange(960) + 1) * .008333
            axes[0].plot(t, d['gap_m'] * 1000, label=LABELS[mode], color=COLOURS[mode])
            axes[1].plot(t, d['force_N'], color=COLOURS[mode])
            axes[2].plot(t, d['hard_violation_rad'], color=COLOURS[mode])
            axes[3].plot(np.arange(481) * .016666, abs(d['controlled_q'] - d['controlled_q'][0]).max(-1), color=COLOURS[mode])
        axes[0].axhline(0, color='#222', lw=.7, ls='--')
        axes[0].set(ylabel='Minimum9021 gap [mm]', title=f'Input{input_id}: held positions over480 controls (~8 s)')
        axes[0].legend(loc='upper right')
        axes[1].set(ylabel='Peak scalar normal [N]')
        axes[2].axhline(1e-5, color='#222', lw=.7, ls='--')
        axes[2].set(ylabel='Max74 hard violation [rad]')
        axes[3].set(ylabel='Max26 departure [rad]', xlabel='Recorded native simulation time [s]')
        export(fig, 'input_' + str(input_id))
        bindings.append(dict(input_id=input_id, plot=f'plots/input_{input_id}.png',
            note=f'输入 {input_id}：原保持失败，补偿后完整480周期通过。曲线中的运动是初始速度收敛，不是操作任务成功。'))
    write(P / 'PUBLIC_CASE_BINDINGS.json', bindings)
    default = next(x for x in bindings if x['input_id'] == 1887)
    count_rows = ''.join(f'<tr><td>{"短程16" if hz=="short" else "长程480"}</td><td>{"关闭补偿" if mode=="off" else "开启补偿"}</td><td>{d["joint_and_primary_pass"]}/128</td><td>{d["geometry_failures"]}</td><td>{d["force_failures"]}</td><td>{d["all74_hard_bad"]}</td><td>{d["all74_speed_bad"]}</td></tr>' for hz in ['short','long'] for mode,d in result[hz]['counts'].items())
    motion_rows = ''.join(f'<tr><td>{html.escape(mode)}</td><td>{d["joint_and_primary_pass"]}/128</td><td>{d["primary_failures"]}</td><td>{d["all74_hard_bad"]}</td><td>{d["all74_speed_bad"]}</td><td>{d["total_controlled_path_rad"]:.4f}</td><td>{d["all4_arms_moved"]}</td></tr>' for mode,d in mcounts.items())
    matrix_rows = ''.join(f'<tr><td>{html.escape(row["requirement"])}</td><td>{html.escape(row["status"])}</td><td>{html.escape(row["measured"])}</td><td>{html.escape(row["missing"])}</td></tr>' for row in matrix['rows'])
    table = []
    for i in range(128):
        rs = [result[hz]['states'][mode][i] for hz in ['short', 'long'] for mode in ['off', 'on']]
        assert len({r['input_id'] for r in rs}) == 1
        table.append('<tr><td>'+str(rs[0]['input_id'])+'</td><td>'+str(rs[0]['cell'])+'</td>'+''.join('<td>'+('失败' if r['primary_failure'] or r['supplementary_max_all74_hard_violation_rad']>1e-5 or r['supplementary_max_all74_velocity_exceedance_rad_s']>1e-5 else '通过')+'</td>' for r in rs)+'</tr>')
    options = ''.join(f'<option value="{x["input_id"]}" {"selected" if x["input_id"]==1887 else ""}>{x["input_id"]}</option>' for x in bindings)
    case_links = ''.join(f'<li>输入{x["input_id"]}：<a href="{x["plot"]}">PNG</a> · <a href="{x["plot"].replace(".png",".svg")}">SVG</a> · <a href="{x["plot"].replace(".png",".pdf")}">PDF</a></li>' for x in bindings)
    evidence_links = ''.join(f'<li><a href="{html.escape(str(f.relative_to(OUT)))}">{html.escape(str(f.relative_to(OUT)))}</a></li>' for f in sorted((OUT/'evidence').rglob('*')) if f.is_file())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo 重力补偿与完整验收</title>
<style>body{{margin:0;background:#f5f7fa;color:#192b3d;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:28px 18px}}h1{{font-size:30px}}h2{{margin-top:32px}}a{{color:#13588d;overflow-wrap:anywhere}}.box{{background:white;padding:20px;border-radius:12px;margin:20px 0;border:1px solid #dce4ec}}.good{{border-left:5px solid #17785b}}.pending{{border-left:5px solid #ba7020}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;vertical-align:top;border-bottom:1px solid #dce4ec;padding:9px;overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}select{{font:inherit;max-width:100%}}li{{overflow-wrap:anywhere}}#matrix td:first-child{{min-width:110px}}#matrix td:nth-child(2){{max-width:150px}}code{{overflow-wrap:anywhere}}</style>
<main><a href="../../">← 返回主面板</a><h1>完整128条长程保持、动作对照与真实夹持缺口</h1>
<p>2026-10-10 · 同128个已知开发输入 · 全26臂关节初态支持 · 原始非零速度 · 位置延迟6周期</p>
<div class="box good"><b>保持机制有实测改善：</b>补偿关闭118/128通过，开启128/128通过；短程16与长程480周期都救回原来的全部10条失败，新增失败0。额外力矩每个物理子步与原生读回一致；原有重力、速度、关节限位和FIFO保留。</div>
<div class="box pending"><b>完整 System 0 安全验收未通过。</b> 新动作候选2048/2048次回退，与保持轨迹相同，没有执行新操作。本轮证明开发集上的位置保持，不能将初始速度收敛当作操作成功。选择性暂停/恢复、真实夹持、四臂共同操作、独立留出和任务视频仍待通过。</div>
<h2>相同输入的保持对照</h2><div class="scroll"><table id="counts"><thead><tr><th>控制周期</th><th>补偿</th><th>联合通过</th><th>几何失败</th><th>接触失败</th><th>限位失败</th><th>速度失败</th></tr></thead><tbody>{count_rows}</tbody></table></div>
<img src="plots/holding_summary.png" alt="相同128输入的短长程通过数和接触峰值"><p>每条长程480控制周期约8秒，完整记录960物理子步。每步9021球对、82碰撞刚体接触归因、全部74关节硬限位和速度都由独立审计计算。关闭补偿的长程来自已闭环的原保持基线；新短程关闭组逐位复现基线前32子步，新长程开启组逐位复现此前开启短程。</p>
<h2>动作修复开发试验</h2><p>两模式共同使用事前冻结、每臂每周期最大分量0.02rad且保留目标方向的目标带，并共同施加同样补偿。队列候选围绕最后排队目标限制候选范围，将实际额外补偿计入名义响应，模型不满足时重复末端目标。它仍未证明到达状态预测或安全回退，也不是正式alpha策略。这里改变了命令分布，不能把它标作原全软限位跳变指令的安全成绩。</p>
<div class="scroll"><table id="motion-counts"><thead><tr><th>模式</th><th>联合通过</th><th>首要失败</th><th>限位失败</th><th>速度失败</th><th>实际路径总和(rad)</th><th>四臂均有运动</th></tr></thead><tbody>{motion_rows}</tbody></table></div>
<img src="plots/motion_summary.png" alt="新队列候选短程安全与实际运动量"><p>模型满足 {result['motion']['model_satisfied_lane_controls']} /2048 次，模型失败回退 {result['motion']['fallback_lane_controls']} 次。路径与四臂运动包括初始惯性，16周期短程没有操作任务，不能据此通过有效动作验收。所有失败均保留在256行下载中。</p>
<p>补充的事后动作诊断与同输入、同补偿的保持轨迹比较，前6个实际控制周期逐位相同。四臂轨迹相对保持均出现超过1mrad差异的输入：原始有界命令 {command_effect['counts']['bounded_raw']['all4_trajectory_differences_gt1mrad']}/128，队列候选 {command_effect['counts']['queue_projection']['all4_trajectory_differences_gt1mrad']}/128。该差异剔除了共用初始惯性，但仍可能包含不安全运动；1mrad是展示阈值，不是正式操作成功门槛。<a href="evidence/motion/COMMAND_EFFECT_DESCRIPTION_V1.json">完整诊断与来源</a></p>
<h2>原10条保持失败的完整长程曲线</h2><label for="case">输入： </label><select id="case">{options}</select><p id="case-note">{default['note']}</p>
<img id="case-plot" src="{default['plot']}" alt="原保持与重力补偿的长程原生指标"><p><a id="case-svg" href="{default['plot'].replace('.png','.svg')}">下载SVG</a> · <a id="case-pdf" href="{default['plot'].replace('.png','.pdf')}">下载PDF</a></p>
<details><summary>全部10个对照图，无JavaScript也可打开</summary><ul id="failure-cases">{case_links}</ul></details>
<h2>完整要求与验收状态</h2><p>以下9项共同构成验收；保持实验通过不会覆盖其他失败。随机初态/命令、手指、布局、物体、感知、延迟和故障支持分别说明，不能把32000个提案或128个筛选输入当作全空间覆盖。</p><div class="scroll"><table id="matrix"><thead><tr><th>要求</th><th>当前状态</th><th>实测证据</th><th>待通过内容</th></tr></thead><tbody>{matrix_rows}</tbody></table></div>
<p>已有物体任务是F双臂与U双臂各搬一物体的并行操作；两方案各1/4满足旧门槛，其中通过槽位的U抬升只出现在退离阶段，不能证明双手稳定搬运。补充阶段诊断中，旧16个物体记录没有一个连续保持100mm/2s且同时满足姿态稳定、双手接触和离桌；归零张手指令后仍有拇指卡住。新增任务观察器通过9项软件测试，并拒绝旧16条不完整记录；它尚未接入原生物理控制，不能称夹持修复成功。正式100mm/2s夹持和四手共持/交接均待通过。<a href="../safety_object_binding_20261006/">查看旧物体任务证据</a>。当前SafeDuo v7实验与NAS计划中的JAKA实际资产profile不同，本页不作该资产profile晋级。</p>
<details id="all-results"><summary>全部128输入的四列保持判定</summary><p><a href="all512_hold_results.csv">下载512行保持结果CSV</a> · <a href="all256_motion_results.csv">下载256行动作结果CSV</a> · <a href="data.json">完整结果JSON</a></p><div class="scroll"><table><thead><tr><th>输入</th><th>组合</th><th>短关</th><th>短开</th><th>长关</th><th>长开</th></tr></thead><tbody>{''.join(table)}</tbody></table></div></details>
<h2>证据范围与复核</h2><p>本页公开完整逐子步指标、原生额外力矩读回、实际参数、运行与独审代码、所有输入判定和完整本地原始文件SHA清单。9021逐对原始大数组和原生点接触块保存在本地，未在本页完整公开；不能声称仅下载本页就可复算全部原始接触/几何。没有本轮新原生视频。<a href="../safety_physical_random_128_20261009/">旧随机四视角视频</a>保持原实验标签。</p>
<details><summary>全部证据下载</summary><ul>{evidence_links}</ul></details></main>
<script>const cases={json.dumps(bindings,ensure_ascii=False)};document.querySelector('#case').addEventListener('change',e=>{{const c=cases.find(x=>String(x.input_id)===e.target.value);document.querySelector('#case-plot').src=c.plot;document.querySelector('#case-svg').href=c.plot.replace('.png','.svg');document.querySelector('#case-pdf').href=c.plot.replace('.png','.pdf');document.querySelector('#case-note').textContent=c.note;}});</script></html>'''
    (OUT / 'index.html').write_text(page)
    main_path = W / 'index.html'
    main = main_path.read_text()
    marker = '<div class="sci-note" id="gravityResponseEvidence">'
    assert main.count(marker) == 1 and 'id="gravityCompleteEvidence"' not in main
    card = f'<div class="sci-note" id="gravityCompleteEvidence"><b>10-10 最新完整补测：重力补偿长程保持128/128通过，完整验收仍未通过</b><p>同128输入、原始速度和6周期延迟；16与480周期补偿关闭118/128、开启128/128，救回全部10条失败且无新增失败。动作候选2048/2048回退、有效操作失败；真实夹持诊断与9项完整验收缺口单列；两次资源失败原样保留，首64条复用；保持通过不等于夹持和四臂操作成功。</p><a href="{SLUG}/">查看最新机制、长程效果与完整验收矩阵 →</a></div>\n'
    main = main.replace(marker, card + marker)
    main_path.write_text(main)
    assert all(sha(W / name) == expected for name, expected in preservation.items())
    payload = [main_path] + sorted(s for s in OUT.rglob('*') if s.is_file())
    write(P / 'PUBLIC_PAYLOAD_MANIFEST.json', dict(slug=SLUG, base_commit=base,
        files={str(s.relative_to(W)): sha(s) for s in payload}, total_files=len(payload),
        total_bytes=sum(s.stat().st_size for s in payload), fullSystem0_accepted=False))
    print('BUILT_GRAVITY_RESPONSE', len(payload), sum(s.stat().st_size for s in payload), flush=True)


if __name__ == '__main__':
    main()
