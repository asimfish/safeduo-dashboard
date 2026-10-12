"""A new short full50 diagnostic, with actual movement separate from incoming range."""
from pathlib import Path
import collections,csv,hashlib,html,json,re,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_loaded_duration_20261012')
P=W/'docs/safety_full50_motion_20261013'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not P.exists();case=H/'fourhand_native_probe_v115'
    result=read(case/'INDEPENDENT_FULL50_RANDOM_MOTION_V1.json');replay=read(E/'FULL74_GEOMETRY_REPLAY_AUDIT_V2.json')
    reg=read(case/'REGISTRATION_V1.json');raw=Path(read(case/'RESOURCE_EXECUTION_V1.json')['out'])
    custody=read(case/'video_evidence_v6/VIDEO_CUSTODY_V1.json')
    assert replay['all_recorded_state_accuracy_gates_pass'] and len(custody['views'])==4
    P.mkdir();(P/'evidence').mkdir();(P/'source').mkdir();(P/'media').mkdir();(P/'raw').mkdir();(P/'figures').mkdir()
    write(P/'FULL50_RESULTS_V1.json',result);shutil.copyfile(E/'FULL74_GEOMETRY_REPLAY_AUDIT_V2.json',P/'GEOMETRY_REPLAY_V2.json')
    for f in case.glob('INDEPENDENT_*.json'):shutil.copyfile(f,P/'evidence'/f.name)
    for f in ['REGISTRATION_V1.json','RESOURCE_EXECUTION_V1.json']:shutil.copyfile(case/f,P/'evidence'/f)
    for f in E.glob('*.json'):shutil.copyfile(f,P/'evidence'/f.name)
    for f in E.glob('*.py'):shutil.copyfile(f,P/'source'/f.name)
    (P/'source/native115').mkdir()
    for f in case.glob('*.py'):shutil.copyfile(f,P/'source/native115'/f.name)
    for f in ['native_common_beam_v1.py','native_clock_binding_v1.py']:
        shutil.copyfile(H/'fourhand_native_probe_v114'/f,P/'source'/('failed114_'+f))
    oldres=read(H/'fourhand_native_probe_v114/RESOURCE_EXECUTION_V1.json');oldraw=Path(oldres['out'])
    for src,name in [(oldraw/'fourhand_protocol.json','NATIVE114_FAILED_PROTOCOL_V1.json'),(H/'fourhand_native_probe_v114/RESOURCE_EXECUTION_V1.json','NATIVE114_RESOURCE_V1.json')]:shutil.copyfile(src,P/'evidence'/name)
    shutil.copyfile(reg['motion_test']['command_bank'],P/'evidence/FULL50_FROZEN_INPUT_BANK_V1.npz')
    meta=read(reg['coupled_assets'])['assets']
    for a,x in meta.items():shutil.copyfile(x['original_URDF'],P/'source'/f'{a}_original_hand_source.urdf')
    with (P/'ACTUAL50_MOTOR_COVERAGE_V1.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['motor_rows'][0]));writer.writeheader();writer.writerows(result['motor_rows'])
    telemetry=[]
    for c in read(raw/'physics_chunks.json'):
        f=raw/c['path'];assert sha(f)==c['sha256']
        with np.load(f) as z:
            speed=np.max(np.concatenate([abs(z[a+'_native_qd'][:,0]) for a in ('F_L','F_R','U_L','U_R')],axis=1),axis=1)
            for i in range(len(speed)):telemetry.append([int(z['physics_event'][i]),float(z['state_time_s'][i]),float(z['all9021_raw_geometry'][i].min()),float(speed[i]),str(z['guard_mode'][i])])
    with (P/'PER_NATIVE_STEP_V1.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['event','native_time_s','minimum_gap_m','maximum74_speed_rad_s','guard_mode']);writer.writerows(telemetry)
    arr=np.asarray([r[:4] for r in telemetry],float);fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    axes[0].plot(arr[:,1],arr[:,2]*1000,color='#2459af');axes[0].axhline(0,color='black',ls='--');axes[0].set_ylabel('Minimum9021 gap (mm)')
    axes[1].plot(arr[:,1],arr[:,3],color='#2459af');axes[1].axhline(.02,color='black',ls='--');axes[1].set_ylabel('Max74 joint speed (rad/s)');axes[1].set_xlabel('Actual native time (s)')
    if result['brake']:
        for ax in axes:ax.axvline(result['brake']['trigger_time_s'],color='#bb3a35',ls=':')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(P/'figures/full50_native_trace.svg');plt.close(fig)
    media=[]
    for row in custody['views']:
        name=row['view']+'.mp4';shutil.copyfile(row['path'],P/'media'/name);assert sha(P/'media'/name)==row['sha256'];media.append(dict(row,public_path='media/'+name))
    write(P/'MEDIA_CUSTODY_V1.json',{'views':media,'native_timestamps_preserved':True})
    # Keep every original state/force/contact/PNG file available; each part below50MiB.
    archives=[];current=None;size=0
    try:
        for f in sorted(raw.rglob('*')):
            if not f.is_file() or f.suffix not in ('.npz','.json','.png','.usda'):continue
            if current is None or size+f.stat().st_size>48*1024**2:
                if current:current.close()
                path=f'raw/native_part_{len(archives)+1:02d}.zip';current=zipfile.ZipFile(P/path,'x',compression=zipfile.ZIP_STORED);size=0
                archives.append({'path':path,'entries':[]})
            name=str(f.relative_to(raw));current.write(f,name);size+=f.stat().st_size
            archives[-1]['entries'].append({'path':name,'bytes':f.stat().st_size,'sha256':sha(f)})
    finally:
        if current:current.close()
    for row in archives:
        f=P/row['path'];row.update(bytes=f.stat().st_size,sha256=sha(f));assert row['bytes']<50*1024**2
        with zipfile.ZipFile(f) as z:
            for entry in row['entries']:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
    write(P/'RAW_ARCHIVE_MANIFEST_V1.json',{'parts':archives,'all_entries_SHA256_verified':True})
    status='短窗原物理与模型门槛通过' if result['finite_preregistered_physical_and_model_pass'] else '短窗原物理或模型门槛未通过'
    motor_rows=''.join(f'<tr><td>{html.escape(r["coordinate"])}</td><td>{"手指" if r["finger_motor"] else "臂"}</td><td>{r["incoming_request_span_rad"]:.5f}</td><td>{r["ordinary_random_effective_target_span_rad"]:.6f}</td><td>{r["actual_span_during_ordinary_random_execution_rad"]:.6f}</td><td>{r["actual_whole_record_span_rad"]:.6f}</td><td>{r["maximum_actual_speed_rad_s"]:.5f}</td></tr>' for r in result['motor_rows'])
    videos=''.join(f'<figure><figcaption>{dict(top="俯视",front="正视",side="侧视",overview="全景")[r["view"]]} · 原生时间1×</figcaption><video src="{r["public_path"]}" controls playsinline preload="metadata"></video><a href="{r["public_path"]}">下载视频</a></figure>' for r in media)
    modes='；'.join(f'{html.escape(k)}：{v}控制周期' for k,v in result['guard_control_counts'].items())
    archive_links=''.join(f'<li><a href="{r["path"]}">原生分包{i+1} · {r["bytes"]/1024**2:.1f} MiB</a></li>' for i,r in enumerate(archives))
    maximum=max(r['maximum_full74_pair_error_m'] for r in replay['rows'])*1e6
    brake='本记录未触发制动。'
    if result['brake']:
        b=result['brake'];s=b['settling'];latency=f'{s["confirmation_latency_s"]:.6f}秒' if s['settled'] else '未达到连续停稳判据'
        brake=f'第{b["first_control"]}控制周期触发制动；连续停稳确认：{latency}。停止单独记账，不计作大范围运动成功。'
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo · 全50电机随机动作诊断</title>
<style>body{margin:0;background:#f4f6fa;color:#172332;font:16px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:28px 18px}h1{font-size:34px;line-height:1.3}a{color:#2459af}section{background:white;padding:20px;margin:18px 0;border-radius:12px;border:1px solid #d9e1ed}.alert{padding:18px;background:#fff8ee;border-left:5px solid #ba6e16}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.metric{padding:16px;background:#eaf0f8;border-radius:9px}.metric b{display:block;font-size:26px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dfe5ed;white-space:nowrap}.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img,video{width:100%;max-width:100%;display:block}figcaption{font-weight:600}.small{font-size:14px;color:#58697c}button{font:inherit;padding:7px 15px;border:1px solid #8ba6c4;border-radius:7px;background:white;color:#2459af}@media(max-width:650px){main{padding:18px 12px}h1{font-size:28px}.metrics{grid-template-columns:repeat(2,1fr)}.videos{grid-template-columns:1fr}section{padding:14px}}</style>
<main><a href="../../index.html#full50MotionEvidence">← 主面板</a><h1>全50电机随机动作诊断</h1><p>SafeDuo 开发诊断 · 2026-10-13 · 2 FR3/F2 + 2 UR5/DFX。学习System0关闭，正式留出0。</p>
<div class="alert"><b>%(status)s；长时间、大范围任意运动仍未验收。</b><p>本轮加入24个独立手指电机的随机目标。全50个目标经过真实FIFO6，预测器读取全部74个实际关节坐标。原始来指令范围、执行目标范围与实际动作范围分别列出；保护性停机不能替代动作覆盖。</p></div>
<div class="metrics"><div class="metric"><b>50</b>独立随机输入电机</div><div class="metric"><b>%(finger_updated)s / 24</b>普通随机执行中更新的手指目标</div><div class="metric"><b>%(duration)s</b>实际原生记录时长</div><div class="metric"><b>0</b>30/120/600秒物理长测完成</div></div>
<section><h2>手指几何进入防护预测</h2><p>旧预测器把手指固定在张开姿态，实际手指变化没有进入预测。新版本从全部74个实际关节坐标出发，用原始URDF手指运动学和原球对定义计算四臂及手指位置；不使用实测腕部位姿锚定，不拟合原生间隙偏置来通过精度审计。</p><p>在109/110/112/113的7920个已有原生子步上，逐项核对71446320条间隙，最大误差%(replay_error)s微米；旧固定手指模型最大差异约95.5毫米。UR基座固定坐标约定错误也已修复，原始失败和回归记录保留。</p><p>这证明了<b>已记录状态上的几何一致性</b>；未来动力学跟踪、制动距离、完整碰撞网格与硬件安全尚未认证。防护中的短时线性预测仍是开发机制。</p><p><a href="GEOMETRY_REPLAY_V2.json">完整几何精度审计</a> · <a href="evidence/LEGACY_FINGER_IGNORANCE_COUNTEREXAMPLE_V1.json">只改变手指、旧预测完全不变的反例</a></p></section>
<section><h2>实际执行与原门槛</h2><p>115：%(events)s个实际子步，%(duration)s，1920Hz物理、60Hz控制、FIFO6。同一初态、1条新开发IID50输入，种子202610130114；前4周期保持原公共前缀，后续每周期独立抽样。首次全50接入114在首个原生步后因任务观测器维度不匹配中断，归档完整控制周期0；该步没有可复核完整记录，明确不计安全试验。115仅修复两个臂目标观测表达式，原模型、质量、控制器、关节速度/限位、力与q/qd联动门槛保持一致。</p><p>原物理门槛：%(physical)s；原q且qd联动：%(model)s；四臂原实际运动门槛：%(movement)s；真实完整抓取任务：未通过。最小9021代理间隙：%(gap)s毫米。</p><p>新IID指令通过FIFO送达%(delivered)s周期；其中普通随机执行%(ordinary)s周期。%(brake)s</p><p class="small">%(modes)s。记录中包含公共前缀与自然动态；全部位移不能直接归因于随机输入。原物理最小间隙门槛0，q与qd关系容差均10⁻⁴。</p><img src="figures/full50_native_trace.svg" alt="全50原生运动的最小间隙和74关节最大速度"></section>
<section><h2>全50个电机：请求、执行、实际运动</h2><p>请求均匀覆盖每个独立电机与被动联动关节共同允许的关节区间。该输入范围不代表可到达、安全或实际执行范围。普通随机执行区间指第10周期起、随机目标实际到达FIFO出口后，且尚未锁存制动的周期。</p><div class="scroll"><table><thead><tr><th>坐标</th><th>类型</th><th>请求跨度/rad</th><th>普通执行目标跨度/rad</th><th>普通随机期间实际跨度/rad</th><th>全记录实际跨度/rad</th><th>最大实际速度/rad/s</th></tr></thead><tbody>%(motors)s</tbody></table></div><p><a href="ACTUAL50_MOTOR_COVERAGE_V1.csv">下载全部50维覆盖CSV</a>。四臂实际运动继续采用各臂偏移≥0.1 rad且腕路径≥0.02 m的原门槛；本轮没有将微小手指变化算作充分大范围覆盖。</p></section>
<section><h2>四视角原生视频</h2><p>俯视、正视、侧视和全景使用已审计的原生相机帧。保留真实帧时间，没有合成动作。</p><button id="slow">全部0.2×慢放</button> <button id="normal">恢复1×</button><div class="videos">%(videos)s</div><p class="small">慢放只改变观看速度。操作物体真实抬起并完成任务尚未通过；此处展示随机指令接入、实际小幅动作及安全干预。</p></section>
<section><h2>距离完整验收的缺口</h2><p>全50随机手指接入短窗的结果单独记账。30/120/600秒的连续输入60条已准备，但物理长测完成仍为0；连续原生记录需避免8192事件账本上限，且不能重置或拼接物理状态来代替长测。</p><p>下一阶段需要不同姿态与初速度下的制动距离、多运动过程的实际关节与工作空间覆盖、停机后的恢复执行，以及多种真实摩擦抓取完整任务。合格完整任务样本0，新增学习System0训练0，正式留出0，九项联合要求未验收。</p><p><a href="../safety_state_guard_20261012/">上一轮制动、长时间随机输入准备与完整九项缺口</a> · <a href="../safety_motion_random_20261012/">四臂实际退离动作和原始随机失败对照</a></p></section>
<section><h2>原始数据与源码</h2><ul><li><a href="FULL50_RESULTS_V1.json">全部结果、50维动作、FIFO与制动审计</a></li><li><a href="PER_NATIVE_STEP_V1.csv">逐原生步指标CSV</a></li><li><a href="MEDIA_CUSTODY_V1.json">四视角帧时间与SHA</a></li><li><a href="RAW_ARCHIVE_MANIFEST_V1.json">全部原始分包和逐文件SHA</a></li><li><a href="evidence/NATIVE114_FAILED_PROTOCOL_V1.json">114失败现场</a> · <a href="evidence/FULL50_FRONTIER_ACTUAL_EXITS_V1.json">实际退出状态记录</a></li><li><a href="source/full74_geometry_v2.py">全74几何预测</a> · <a href="source/full50_actual_guard_v1.py">全50实际状态防护</a> · <a href="source/audit_native115_full50_v1.py">独立原生审计</a></li>%(archives)s</ul></section>
</main><script>document.getElementById('slow').onclick=()=>document.querySelectorAll('video').forEach(v=>v.playbackRate=.2);document.getElementById('normal').onclick=()=>document.querySelectorAll('video').forEach(v=>v.playbackRate=1);</script></html>'''
    yn=lambda x:'通过' if x else '未通过'
    values=dict(status=status,finger_updated=result['ordinary_finger_motor_target_dimensions_updated'],duration=f'{result["duration_s"]:.3f}秒',replay_error=f'{maximum:.2f}',events=result['physics_events'],
        physical=yn(result['all_original_physical_gates_pass']),model=yn(result['original_q_AND_qd_residuals_pass']),movement=yn(result['original_all4_arm_movement_admission_pass']),
        gap=f'{result["native_minimum_gap_m"]*1000:.6f}',delivered=result['new_IID_packets_delivered_by_FIFO_controls'],ordinary=result['ordinary_controls_after_new_random_FIFO_delivery'],brake=brake,modes=modes,motors=motor_rows,videos=videos,archives=archive_links)
    (P/'index.html').write_text(re.sub(r'%\((\w+)\)s',lambda m:str(values[m[1]]),document))
    root=W/'index.html';before=root.read_bytes();marker=b'<div id="actualStateBrakeEvidence"';assert before.count(marker)==1 and b'id="full50MotionEvidence"' not in before
    (E/'DASHBOARD_INDEX_BEFORE_FULL50_V1.html').write_bytes(before)
    card=f'<div id="full50MotionEvidence" class="sci-note" style="margin-bottom:18px;border-left:4px solid #2459af"><strong>最新全50电机随机动作：{status}</strong><p>24个手指电机随机目标进入实际FIFO；全部74关节预测与原生间隙已核对。实际动作范围、保护性制动和失败分别列出；长时间大范围运动、完整任务与学习System0尚未验收。</p><a href="docs/safety_full50_motion_20261013/">查看全50动作、四视角视频与原始数据 →</a></div>\n'
    root.write_bytes(before.replace(marker,card.encode()+marker,1));assert root.read_bytes().replace(card.encode(),b'',1)==before
    write(E/'DASHBOARD_FULL50_INSERTION_V1.json',{'card':card,'old_bytes_SHA256':hashlib.sha256(before).hexdigest(),'all_old_content_preserved':True})
    files=[{'path':str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(P.rglob('*')) if f.is_file()]
    write(P/'PUBLIC_MANIFEST_V1.json',{'files':files,'total_bytes':sum(r['bytes'] for r in files),'source_builder_sha256':sha(__file__),'fullSystem0_accepted':False})
    print('FULL50_REPORT_BUILT',len(files),'files',sum(r['bytes'] for r in files),'bytes;4 native videos',flush=True)


if __name__=='__main__':main()
