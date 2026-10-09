"""Publish a closed early-failure diagnostic without changing prior evidence."""
import csv
import datetime
import hashlib
import gzip
import html
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parent
H = P.parent
F = H / 'hold_factorial128_v1'
W = Path('/home/liyufeng/safeduo-dashboard-response-probe-20261009')
SLUG = 'docs/safety_hold_factorial_20261009'
OUT = W / SLUG
FACTORS = ['baseline', 'zero_arm_velocity', 'matched_hand_target', 'both']
LABELS = ['Original hold', 'Zero initial arm velocity', 'Match hand targets', 'Both interventions']
ZH = ['原样保持', '只将初始臂速度置零', '只对齐夹爪目标', '两项同时改变']
COLORS = ['#c64737', '#e39b22', '#3776bd', '#278558']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save(fig, name):
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / 'plots' / (name + '.' + suffix), dpi=170, bbox_inches='tight')
    plt.close(fig)


def figure(name, caption):
    return f'<figure><img src="plots/{name}.png" alt="{html.escape(caption)}"><figcaption>{caption} · <a href="plots/{name}.svg">SVG</a> · <a href="plots/{name}.pdf">PDF</a></figcaption></figure>'


def main():
    result = read(F / 'FACTORIAL128_RESULT_V1.json')
    finished = read(F / 'FINISH_EXECUTION_V1.json')
    execution = read(H / 'hold_factorial128_v1_execution.json')
    reg = read(F / 'REGISTRATION_V1.json')
    assert result['baseline_exact_reproduction'] and result['all128_each_factor_retained']
    assert finished['status'] == 'complete' and len(finished['steps']) == 10
    assert all(s['actual_exit'] == 0 for s in finished['steps'])
    assert execution['status'] == 'complete' and len(execution['jobs']) == 8
    assert all(j['actual_exit'] == 0 for j in execution['jobs'])
    assert not result['runtime_braking'] and not result['fullSystem0_accepted']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W).strip()
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip()
    assert base == 'a714bd092526fc3b9b75803c8db5e16ed3a57068'
    assert not OUT.exists()
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=W).decode().split('\0')
    preservation = {n: sha(W / n) for n in names if n and n != 'index.html'}
    write(P / 'PRESERVED_BEFORE_BUILD_V1.json', preservation)
    OUT.mkdir()
    (OUT / 'plots').mkdir()
    (OUT / 'evidence').mkdir()
    public_sources = []
    def copy(source, dest):
        target = OUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        public_sources.append(dict(source=str(source), path=dest, sha256=sha(source), bytes=source.stat().st_size))
    for name in ['REGISTRATION_V1.json', 'FACTORIAL128_RESULT_V1.json', 'BASELINE_REPRODUCTION_V1.json',
                 'FINISH_EXECUTION_V1.json', 'PREFLIGHT_GATE_V1.json', 'SOURCE_CLONE_RECEIPT_V1.json',
                 'native_factorial_v1.py', 'independent_oracle16_v1.py', 'analyse_factorial_v1.py', 'finish_factorial_v1.py']:
        copy(F / name, 'evidence/' + name)
    copy(H / 'HOLD_FACTORIAL128_PLAN_V1.json', 'evidence/HOLD_FACTORIAL128_PLAN_V1.json')
    copy(Path(__file__), 'evidence/build_report_v2.py')
    copy(Path(reg['bank']), 'evidence/selected_inputs_and_targets.npz')
    copy(Path(reg['fixture']), 'evidence/hand_margin_fixture_v1.npz')
    public_native = []
    for batch in range(2):
        for factor in FACTORS:
            leaf = Path(reg['raw']) / f'paired_batch{batch}_{factor}_v8'
            dest = f'evidence/native_batch{batch}_{factor}'
            # Include all closed32-event raw archives, their identity, dynamics and issue streams.
            for source in sorted(leaf.rglob('*')):
                if source.is_dir():continue
                if source.name=='native_contact_identity.json':
                    target=OUT/(dest+'/'+source.name+'.gz')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(gzip.compress(source.read_bytes(),compresslevel=6,mtime=0))
                    assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest()==sha(source)
                    public_sources.append(dict(source=str(source),path=str(target.relative_to(OUT)),sha256=sha(target),bytes=target.stat().st_size,encoding='gzip',decoded_sha256=sha(source),original_bytes=source.stat().st_size))
                elif source.suffix in ['.json','.npz']:
                    copy(source,dest+'/'+str(source.relative_to(leaf)))
            for name in [f'FACTOR_ORACLE_batch{batch}_{factor}_V1.json', f'PAIRED_METRICS_batch{batch}_{factor}_V1.npz']:
                copy(F / name, 'evidence/' + name)
            public_native.append(dict(batch=batch, factor=factor, directory=dest,
                                      receipt=dest + '/point_contact_receipts.json',
                                      geometry_receipt=dest + '/all_raw_geometry_receipts.json',
                                      identity_gzip=dest+'/native_contact_identity.json.gz',actual_exit=0, controls=16, microsteps=32))
    write(OUT / 'evidence/BUNDLE_RECEIPT_V1.json', dict(status='PASS_EXACT_CLOSED_FACTORIAL128_EVIDENCE_BUNDLE',
          utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), sources=public_sources, native=public_native,
          raw32_events_all8_public=True, previous480_raw_archives_included=False, fullSystem0_accepted=False))
    rows = [dict(row) for factor in FACTORS for row in result['states'][factor]]
    fields = list(rows[0])
    with (OUT / 'all512_factor_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 512
    metrics = {f: [load(F / f'PAIRED_METRICS_batch{b}_{f}_V1.npz') for b in range(2)] for f in FACTORS}
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})
    fig, ax = plt.subplots(figsize=(10, 4.7))
    x = np.arange(4)
    for i, (key, label) in enumerate([('primary_failures', 'Geometry or contact failures'), ('all74_hard_bad', 'Hard-limit failures'), ('all74_speed_bad', 'Speed-limit failures')]):
        bars = ax.bar(x + (i - 1) * .22, [result['counts'][f][key] for f in FACTORS], width=.2, label=label)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(x, ['Original', 'Zero velocity', 'Match hands', 'Both'])
    ax.set_ylabel('Failed inputs / 128')
    ax.set_ylim(0, 128)
    ax.legend(loc='upper left')
    ax.set_title('Matched initial-hold diagnostic: 16 controls / 32 native substeps')
    ax.grid(axis='y', alpha=.18)
    fig.tight_layout()
    save(fig, 'factor_failures')
    matrix = np.array([[int(not r['full74_and_primary_pass']) for r in result['states'][f]] for f in FACTORS])
    fig, ax = plt.subplots(figsize=(11, 3.5))
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=ListedColormap(['#deeee6', '#c64737']), vmin=0, vmax=1)
    ax.set_yticks(range(4), LABELS)
    ax.set_xlabel('All128 retained inputs (sorted input ID); red = joint or primary failure')
    ax.set_title('Same development cohort; no replacement or outcome exclusions')
    fig.tight_layout()
    save(fig, 'all128_matrix')
    failures = result['original10_onset_failures_reproduced']
    assert len(failures) == 10
    cases = []
    for input_id in failures:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        keys = [('gap_m', 'Raw minimum clearance\n(m)', 0.), ('force_N', 'Native scalar normal force\n(N)', .1),
                ('hard_violation_rad', 'All74 hard-limit violation\n(rad)', 1e-5), ('velocity_exceedance_rad_s', 'All74 speed exceedance\n(rad/s)', 1e-5)]
        for f, label, color in zip(FACTORS, LABELS, COLORS):
            batch = next(b for b in range(2) if input_id in metrics[f][b]['global_input_id'])
            m = metrics[f][batch]
            lane = int(np.flatnonzero(m['global_input_id'] == input_id)[0])
            t = (np.arange(32) + 1) / 120.
            for ax, (key, ylabel, threshold) in zip(axes.flat, keys):
                ax.plot(t, m[key][:, lane], color=color, label=label)
        for ax, (key, ylabel, threshold) in zip(axes.flat, keys):
            ax.axhline(threshold, color='#777', linestyle='--', linewidth=1)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Time after reset (s)')
            ax.set_yscale('symlog', linthresh=.001 if key == 'gap_m' else 1e-5 if key != 'force_N' else .1)
            ax.grid(alpha=.2)
            if key != 'gap_m':
                ax.set_ylim(bottom=0.)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper center', ncol=2, bbox_to_anchor=(.5, 1.065))
        fig.suptitle(f'Original hold failure input {input_id}: matched four-factor native traces', y=1.13)
        name = f'case_{input_id}'
        save(fig, name)
        cases.append(dict(input_id=input_id, plot='plots/' + name + '.png'))
    write(P / 'PUBLIC_CASE_BINDINGS.json', cases)
    default = next(c for c in cases if c['input_id'] == 1887)
    table = ''.join(f'<tr><td>{ZH[i]}</td><td>{result["counts"][f]["joint_and_primary_pass"]} /128</td><td>{result["counts"][f]["geometry_failures"]}</td><td>{result["counts"][f]["force_failures"]}</td><td>{result["counts"][f]["all74_hard_bad"]}</td><td>{result["counts"][f]["all74_speed_bad"]}</td><td>{result["counts"][f]["peak_normal_N"]:.6g}</td></tr>' for i, f in enumerate(FACTORS))
    # State the measured paired result; do not infer untested long-horizon or task effects.
    same_failures = all(not result['paired_changes'][f]['rescued_primary_ids'] and not result['paired_changes'][f]['new_primary_failure_ids'] for f in FACTORS[1:])
    assert same_failures, 'This report explanation is specific to the closed paired result'
    finding = f'<section class="alert"><h2>本轮诊断结论</h2><p>四组均为 118/128 联合通过，同样的 10 条几何失败和 1 条接触失败仍存在。两项干预单独或同时实施均未解除任何一条原始失败。初始臂速度置零后峰值法向力从 {result["counts"]["baseline"]["peak_normal_N"]:.2f} N 降至 {result["counts"]["zero_arm_velocity"]["peak_normal_N"]:.2f} N，但仍超过 0.1 N 判据。不能把降低接触力当作通过安全验收。</p><p>需要继续检查保持过程的实际动力学、执行器响应和碰撞位置；这轮没有验证新的可执行制动策略。</p></section>'
    options = ''.join(f'<option value="{c["input_id"]}"'+(' selected' if c == default else '')+f'>{c["input_id"]}</option>' for c in cases)
    links = ''.join(f'<li>输入 {c["input_id"]}：<a href="{c["plot"]}">PNG</a> · <a href="{c["plot"].replace(".png", ".svg")}">SVG</a> · <a href="{c["plot"].replace(".png", ".pdf")}">PDF</a></li>' for c in cases)
    evidence = ''.join(f'<li><a href="{html.escape(s["path"])}">{html.escape(s["path"])}</a> ({s["bytes"]:,} bytes)</li>' for s in public_sources)
    paired = ''.join('<tr><td>'+str(i)+'</td>'+''.join(f'<td>{"通过" if result["states"][f][j]["full74_and_primary_pass"] else "失败"}</td>' for f in FACTORS)+'</tr>' for j, i in enumerate([r['input_id'] for r in result['states']['baseline']]))
    css = 'body{margin:0;background:#f5f7fa;color:#182433;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:28px 18px}h1{line-height:1.3}a{color:#1764ac}section,figure{background:white;padding:20px;border-radius:12px;margin:20px 0}figure{margin-inline:0}img{display:block;width:100%;height:auto}figcaption{font-size:14px;margin-top:12px}.alert{border-left:5px solid #c64737}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}select{max-width:100%;padding:8px}code{overflow-wrap:anywhere}li{overflow-wrap:anywhere}@media(max-width:600px){main{padding:16px 10px}section,figure{padding:14px}h1{font-size:25px}}'
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo 保持回退失败的匹配因子诊断</title><style>{css}</style><main>
<a href="../../#scientific">← 主面板科研评测</a><h1>保持回退为何仍会失败：128 个输入的四组匹配诊断</h1>
<section class="alert"><b>完整 System 0 安全验收未通过。</b><p>这轮仅诊断最初 16 个控制周期（32 个原生物理步，约 0.267 秒），不是 480 周期的安全改进结果。置零初始速度是仿真初态干预，不能当作运行时刹车。动作抑制不等于任务成功。</p><a href="../safety_strong_long128_20261009/">查看完整 128 条长轨迹：原样回退仍有 10 条失败 →</a></section>
<section><h2>比较的是什么</h2><p>沿用上一轮全部 128 个初始姿态与分组。每组始终发出初始臂关节位置，保留原始 6 周期指令延迟；只改变初始臂速度或夹爪保持目标。原样保持组已精确复现旧实验前 16 周期的全部关节状态、指令队列、几何与接触力指标，10 条早期失败均复现。四组原生执行与独立审计全部完成。</p><p>这是根据已发现失败设计的开发集机制诊断，非盲测或正式留出验收。位置迭代 64、速度迭代 0、夹爪限位余量 0.03 rad；同一批 128 输入均保留，没有结果筛选或替换。</p></section>
{finding}
<section><h2>全部输入的结果</h2><div class="scroll"><table id="counts"><thead><tr><th>干预</th><th>联合通过</th><th>几何失败</th><th>接触失败</th><th>74关节硬限位失败</th><th>74关节限速失败</th><th>峰值法向力 N</th></tr></thead><tbody>{table}</tbody></table></div><p>联合通过要求所有 32 个已记录物理步：9,021 条原始几何距离 ≥0；82 个碰撞刚体的原生点接触标量法向力 ≤0.1 N；全部 74 关节硬限位与限速超量 ≤10⁻⁵。离散步通过不能证明连续时间安全。</p></section>
{figure('factor_failures', '同一批 128 输入的四组失败数；纵轴分母固定为 128。')}
{figure('all128_matrix', '四组完整 128 输入的联合判据；红色为失败。')}
<section><h2>全部 10 条原始失败的逐物理步曲线</h2><label>选择输入：<select id="case">{options}</select></label><p id="case-note">输入 {default['input_id']}；四组均显示全部 32 个物理步。</p><figure><img id="case-plot" src="{default['plot']}" alt="四组对照的32物理步几何、接触力和关节违规"><figcaption><a id="case-svg" href="{default['plot'].replace('.png','.svg')}">SVG</a> · <a id="case-pdf" href="{default['plot'].replace('.png','.pdf')}">PDF</a></figcaption></figure><ul id="failure-cases">{links}</ul></section>
<section><details id="all-results"><summary>展开全部 128 输入的配对结果</summary><a href="all512_factor_results.csv">下载四组全部 512 行 CSV</a><div class="scroll"><table><thead><tr><th>输入</th>{''.join('<th>'+s+'</th>' for s in ZH)}</tr></thead><tbody>{paired}</tbody></table></div></details></section>
<section><h2>可复核证据</h2><p>下面包括八次原生运行的全部 32 步几何、原生点接触、动力学与指令记录，以及参数、初态、协议、独立审计和原样复现证明。身份 JSON 采用 gzip 无损压缩；解压后原始字节 SHA256 也列在清单中。旧 480 周期原始大归档仍按旧报告范围保留。</p><a href="evidence/BUNDLE_RECEIPT_V1.json">证据清单和逐文件 SHA256</a><details><summary>展开全部证据下载</summary><ul>{evidence}</ul></details></section>
<section><h2>仍需解决</h2><p>本轮改变仿真初始条件，只能帮助定位保持失败机制。实际系统需要在延迟队列与非零速度下验证可执行的制动方案、预测到达状态，并验证有用动作；随后还要测试真实物体感知、夹持与四臂协同任务。没有物体、训练策略、真实硬件或完整 System 0 验收结论。</p></section></main>
<script>const cases={json.dumps(cases)};const select=document.querySelector('#case');select.addEventListener('change',()=>{{const c=cases.find(x=>String(x.input_id)===select.value);document.querySelector('#case-plot').src=c.plot;document.querySelector('#case-svg').href=c.plot.replace('.png','.svg');document.querySelector('#case-pdf').href=c.plot.replace('.png','.pdf');document.querySelector('#case-note').textContent=`输入 ${{c.input_id}}；四组均显示全部 32 个物理步。`;}});</script></html>'''
    (OUT / 'index.html').write_text(body)
    main = (W / 'index.html').read_text()
    marker = '<div class="sci-note" id="strongRandomPrefixEvidence">'
    assert main.count(marker) == 1 and 'id="holdFactorialEvidence"' not in main
    counts = '、'.join(f'{ZH[i]} {result["counts"][f]["joint_and_primary_pass"]}/128' for i, f in enumerate(FACTORS))
    card = f'<div class="sci-note" id="holdFactorialEvidence"><b>10-09 新增：保持回退早期失败的四组匹配诊断</b><p>相同128输入、16控制周期；联合通过 {counts}。原样组精确复现全部10条早期失败；仿真初态干预不能当作运行时制动或长期安全。全部输入、四组曲线和原生证据可下载。</p><a href="{SLUG}/">查看保持失败机制与四组对照 →</a></div>\n'
    main = main.replace(marker, card + marker, 1)
    stale = '128条完整三策略轨迹测试正在运行；初态准入不等于长期或任务安全。'
    assert main.count(stale) == 1
    main = main.replace(stale, '128条完整三策略轨迹已完成，结果见上方长轨迹报告；初态准入不等于长期或任务安全。', 1)
    (W / 'index.html').write_text(main)
    assert all(sha(W / n) == value for n, value in preservation.items())
    payload = [W / 'index.html'] + sorted(q for q in OUT.rglob('*') if q.is_file())
    manifest = dict(slug=SLUG, base_commit=base, files={str(q.relative_to(W)): sha(q) for q in payload},
                    total_files=len(payload), total_bytes=sum(q.stat().st_size for q in payload),
                    scope='Closed16-control matched factorial diagnosis; original128 long results unchanged', fullSystem0_accepted=False)
    write(P / 'PUBLIC_PAYLOAD_MANIFEST.json', manifest)
    print('BUILT_FACTORIAL_REPORT', manifest['total_files'], manifest['total_bytes'], flush=True)


if __name__ == '__main__':
    main()
