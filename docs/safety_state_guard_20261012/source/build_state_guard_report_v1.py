"""Publishable evidence for closed native cases; safety claims remain narrow."""
from pathlib import Path
import ast,csv,datetime,hashlib,html,json,re,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_loaded_duration_20261012')
L=Path('/mnt/nas/data/lyf/double_hand/safeduo_long_random_motion_20261012')
P=W/'docs/safety_state_guard_20261012'
ARMS=('F_L','F_R','U_L','U_R')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,s:Path(p).write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')


def measurements(v):
    p=H/f'fourhand_native_probe_v{v}';reg=read(p/'REGISTRATION_V1.json')
    raw=Path(read(p/'RESOURCE_EXECUTION_V1.json')['out'])
    model=read(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    with np.load(raw/'native_parameters.npz') as z:names={a:z[a+'_joint_names'].tolist() for a in ARMS}
    parts=[]
    for c in read(raw/'physics_chunks.json'):
        f=raw/c['path'];assert sha(f)==c['sha256']
        with np.load(f) as z:
            qerr=np.zeros(len(z['physics_event']));verr=qerr.copy()
            for r in model['relations']:
                a=r['arm'];i=names[a].index(r['slave']);j=names[a].index(r['master'])
                qerr=np.maximum(qerr,abs(z[a+'_native_q'][:,0,i]-r['source_multiplier']*z[a+'_native_q'][:,0,j]-r['source_offset_rad']))
                verr=np.maximum(verr,abs(z[a+'_native_qd'][:,0,i]-r['source_multiplier']*z[a+'_native_qd'][:,0,j]))
            assert z['all9021_raw_geometry'].shape==(len(z['physics_event']),1,9021)
            parts.append({'event':z['physics_event'],'time':z['state_time_s'],'gap':z['all9021_raw_geometry'][:,0].min(axis=1),
                'speed':np.max(np.concatenate([abs(z[a+'_native_qd'][:,0]) for a in ARMS],axis=1),axis=1),
                'qerr':qerr,'verr':verr,'mode':z['guard_mode'],'hard':z['all74_hard_violation_rad'],
                'speed_excess':z['all74_speed_exceedance_rad_s']})
    return reg,raw,{k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    assert not P.exists()
    versions=(109,110,112,113);summary=[];native={}
    brake=read(H/'fourhand_native_probe_v113/INDEPENDENT_EMERGENCY_BRAKE_V1.json')
    for v in versions:
        p=H/f'fourhand_native_probe_v{v}';reg,raw,data=measurements(v);native[v]=(reg,raw,data)
        capture=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json');model=read(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
        motion=read(p/('INDEPENDENT_ACTUAL_STATE_GUARD_MOTION_V1.json' if v in (109,110) else f'INDEPENDENT_ACTUAL_STATE_GUARD_MOTION_V{2 if v==112 else 3}.json'))
        tol=model['predeclared_relation_tolerance']
        relation_pass=model['maximum_q_residual_rad']<=tol and model['maximum_qd_residual_rad_s']<=tol
        summary.append({'version':v,'physics_events':capture['physics_events'],'duration_s':capture['observed_duration_s'],
            'physics_hz':1/reg['physics_dt_s'],'physical_pass':capture['physical_gates_pass'],
            'original_q_AND_qd_residual_pass':relation_pass,'physical_AND_model_window_pass':model['model_window_pass'],
            'maximum_q_residual_rad':model['maximum_q_residual_rad'],
            'maximum_qd_residual_rad_s':model['maximum_qd_residual_rad_s'],
            'minimum_gap_m':capture['minimum_all9021_gap_m'],'complete':capture['maximum_window_complete'],
            'all4_actual_movement_admission_pass':motion['all4_actual_movement_admission_pass'],
            'actual_movement':motion['rows'],'genuine_full_task_pass':capture['four_hand_task_development_pass']})
    # Pair112/113 differs in braking at control15; validate the preceding native physics.
    raw112=native[112][1];raw113=native[113][1];prefix=0
    for c in read(raw112/'physics_chunks.json'):
        if c['stop']>480:break
        with np.load(raw112/c['path']) as old,np.load(raw113/c['path']) as new:
            for k in old.files:np.testing.assert_array_equal(new[k],old[k])
        prefix+=c['stop']-c['start']
    assert prefix==480
    P.mkdir();(P/'evidence').mkdir();(P/'media').mkdir();(P/'figures').mkdir();(P/'raw').mkdir();(P/'source').mkdir()
    results={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':summary,
        'startup111':{'status':'INVALID_STARTUP_NOT_A_PHYSICAL_TRIAL','measured_physics_events':0,'outer_exit':1,'child_exit':0},
        'same1920Hz_pair_first480_native_events_byte_equal':True,'brake':brake,
        'new_controlled_native_microsteps':sum(r['physics_events'] for r in summary),
        'random_input_sequences':1,'initial_states':1,'input_random_arm_dimensions':26,'input_random_finger_dimensions':0,
        'brake_actuated_motor_dimensions':50,'qualified_friction_tasks':0,'new_learned_training_jobs':0,
        'formal_holdouts':0,'native_long_random_30_120_600s_completed':0,'fullSystem0_accepted':False}
    write(P/'STATE_GUARD_RESULTS_V1.json',results)
    fieldnames=['version','event','time_s','minimum_proxy_gap_m','maximum_all74_speed_rad_s','maximum_original_mimic_q_residual_rad',
                'maximum_original_mimic_qd_residual_rad_s','hard_violation_rad','speed_excess_rad_s','mode']
    with (P/'PER_NATIVE_STEP_V1.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(fieldnames)
        for v in versions:
            z=native[v][2]
            for i in range(len(z['event'])):writer.writerow([v,int(z['event'][i]),*map(float,[z[k][i] for k in ['time','gap','speed','qerr','verr','hard','speed_excess']]),str(z['mode'][i])])
    media=[]
    for v in versions:
        p=H/f'fourhand_native_probe_v{v}'
        for f in p.glob('INDEPENDENT_*.json'):shutil.copyfile(f,P/'evidence'/f'v{v}_{f.name}')
        for filename in ['REGISTRATION_V1.json','RESOURCE_EXECUTION_V1.json']:
            shutil.copyfile(p/filename,P/'evidence'/f'v{v}_{filename}')
        custody=read(p/'video_evidence_v6/VIDEO_CUSTODY_V1.json')
        shutil.copyfile(p/'video_evidence_v6/VIDEO_CUSTODY_V1.json',P/'evidence'/f'v{v}_VIDEO_CUSTODY_V1.json')
        for row in custody['views']:
            src=Path(row['path']);name=f'v{v}_{row["view"]}.mp4';shutil.copyfile(src,P/'media'/name)
            assert sha(P/'media'/name)==row['sha256'];media.append(dict(row,version=v,public_path='media/'+name))
    write(P/'MEDIA_CUSTODY_V1.json',{'native_videos':media,'all_frame_timestamp_errors_below1_1_microseconds':True})
    p111=H/'fourhand_native_probe_v111';shutil.copyfile(p111/'RESOURCE_EXECUTION_V1.json',P/'evidence/v111_RESOURCE_EXECUTION_V1.json')
    raw111=Path(read(p111/'RESOURCE_EXECUTION_V1.json')['out']);shutil.copyfile(raw111/'fourhand_protocol.json',P/'evidence/v111_fourhand_protocol.json')
    long=read(L/'LONG_RANDOM_INPUT_AUDIT_V1.json');shutil.copyfile(L/'LONG_RANDOM_INPUT_AUDIT_V1.json',P/'evidence/LONG_RANDOM_INPUT_AUDIT_V1.json')
    shutil.copyfile(D/'REPAIR_CHAIN_ACTUAL_EXITS_V2.json',P/'evidence/REPAIR_CHAIN_ACTUAL_EXITS_V2.json')
    for f in D.glob('*.py'):shutil.copyfile(f,P/'source'/f.name)
    for v in versions:
        src=P/'source'/f'v{v}';src.mkdir()
        for name in ['native_common_beam_v1.py','actual_state_arm_guard_v1.py','native_clock_binding_v1.py','coupled_implicit_motor_effort_v1.py','coupled_motor_effort_v1.py']:
            shutil.copyfile(H/f'fourhand_native_probe_v{v}'/name,src/name)
    for filename in ['long_motion_stream.py','motion_episode_judge.py','test_motion_contracts.py']:shutil.copyfile(L/filename,P/'source'/filename)
    # Exact native snapshots, contacts, clocks and PNGs are downloadable, without size-limit bypasses.
    archives=[];current=None;size=0;archive_index=0
    try:
        for v in versions:
            for f in sorted(native[v][1].rglob('*')):
                if not f.is_file() or f.suffix not in ('.npz','.json','.png','.usda'):continue
                if current is None or size+f.stat().st_size>64*1024**2:
                    if current:current.close()
                    archive_index+=1;name=f'raw/native_part_{archive_index:02d}.zip'
                    current=zipfile.ZipFile(P/name,'x',compression=zipfile.ZIP_STORED);size=0
                    archives.append({'path':name,'entries':[]})
                name=f'v{v}/'+str(f.relative_to(native[v][1]));current.write(f,name);size+=f.stat().st_size
                archives[-1]['entries'].append({'path':name,'bytes':f.stat().st_size,'sha256':sha(f)})
    finally:
        if current:current.close()
    for row in archives:
        f=P/row['path'];row.update(bytes=f.stat().st_size,sha256=sha(f));assert row['bytes']<95*1024**2
        with zipfile.ZipFile(f) as z:
            for entry in row['entries']:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
    write(P/'RAW_ARCHIVE_MANIFEST_V1.json',{'parts':archives,'all_downloadable_entry_digests_verified':True})
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for v,color in [(112,'#bb3a35'),(113,'#167251')]:
        z=native[v][2];axes[0].plot(z['time'],z['gap']*1000,label=f'Native{v}',color=color)
        axes[1].plot(z['time'],z['speed'],label=f'Native{v}',color=color)
    axes[0].axhline(0,color='black',ls='--',lw=1);axes[0].set_ylabel('Minimum proxy gap (mm)');axes[0].legend()
    axes[1].axhline(.02,color='black',ls='--',lw=1);axes[1].set_ylabel('Maximum74 joint speed (rad/s)');axes[1].set_xlabel('Actual native time (s)')
    for ax in axes:ax.axvline(brake['brake_trigger_time_s'],color='#666',ls=':');ax.grid(alpha=.2)
    fig.savefig(P/'figures/brake_pair.svg');plt.close(fig)
    names={109:'实际状态/FIFO防护',110:'加入命令速率/加速度约束',112:'1920Hz积分对照，未锁存制动',113:'1920Hz对照，锁存全50电机制动'}
    yn=lambda b:'通过' if b else '未通过'
    tab=''.join(f'<tr><td>{r["version"]} · {names[r["version"]]}</td><td>{r["physics_events"]} / {r["duration_s"]:.3f}s</td><td>{r["physics_hz"]:.0f}</td><td>{yn(r["physical_pass"])}</td><td>{yn(r["original_q_AND_qd_residual_pass"])}</td><td>{r["minimum_gap_m"]*1000:.6f}</td><td>{r["maximum_qd_residual_rad_s"]:.3g}</td><td>{yn(r["all4_actual_movement_admission_pass"])}</td></tr>' for r in summary)
    videos=''
    for v in versions:
        cards=''.join(f'<figure><figcaption>{dict(top="俯视",front="正视",side="侧视",overview="全景")[m["view"]]} · 原生时间1×</figcaption><video controls playsinline preload="metadata" src="{m["public_path"]}"></video><a href="{m["public_path"]}">下载视频</a></figure>' for m in media if m['version']==v)
        videos+=f'<section class="case"><h3>{v} · {names[v]}</h3><div class="videos">{cards}</div></section>'
    families=''
    for family in long['process_families']:
        rows=[r for r in long['rows'] if r['process_family']==family]
        span=[r.get('minimum_requested_span_fraction') for r in rows]
        families+=f'<tr><td>{html.escape(family)}</td><td>30 / 120 / 600s，各2种子</td><td>6</td><td>0</td></tr>'
    matrix=read(W/'docs/safety_motion_random_20261012/EVIDENCE_MATRIX_V32.json')
    criteria=''.join(f'<li>{html.escape(r["requirement"])}：未联合验收</li>' for r in matrix['rows'])
    assert len(matrix['rows'])==9 and not any(r['joint_requirement_accepted'] for r in matrix['rows'])
    downloads=''.join(f'<li><a href="{r["path"]}">原生数据分包{i+1} · {r["bytes"]/1024**2:.1f} MiB</a></li>' for i,r in enumerate(archives))
    s=brake['settling_observation'];stop=f'{s["confirmation_latency_s"]:.3f}s' if s['settled'] else '尚未停稳'
    verdict='该已知场景的保护性制动通过' if brake['measured_emergency_brake_case_pass'] else '该已知场景的保护性制动未通过'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo · 实际状态防护与原生制动</title>
<style>body{margin:0;background:#f4f6fa;color:#172332;font:16px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:28px 18px}h1{font-size:34px;line-height:1.25}h2{margin-top:32px}a{color:#2459af}section,.panel{background:white;padding:20px;margin:18px 0;border-radius:12px;border:1px solid #d9e1ed}.alert{border-left:5px solid #ba6e16;padding:16px;background:#fff8ee}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:#eaf0f8;padding:16px;border-radius:9px}.metric b{display:block;font-size:26px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #dfe5ed;white-space:nowrap}.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}video,img{max-width:100%;width:100%;display:block}figcaption{font-weight:600}.small{font-size:14px;color:#58697c}li{margin:7px 0}button{font:inherit;padding:7px 15px;border:1px solid #8ba6c4;border-radius:7px;background:white;color:#2459af}@media(max-width:650px){h1{font-size:28px}.metrics{grid-template-columns:repeat(2,1fr)}.videos{grid-template-columns:1fr}main{padding:18px 12px}section{padding:14px}}</style>
<main><a href="../../index.html#actualStateBrakeEvidence">← 主面板</a><h1>实际状态防护与原生制动</h1><p>SafeDuo 开发诊断 · 2026-10-12 · 实验配置：2 FR3/F2 + 2 UR5/DFX。学习 System0 关闭，正式留出0。</p>
<div class="alert"><b>%(verdict)s；任意长时间、大范围运动仍未验收。</b><p>保护版本在没有合格候选时触发全电机制动，原生仿真继续运行。保护性停机单独计分；本轮四臂实际运动门槛和真实抓取完整任务均未通过。</p></div>
<div class="metrics"><div class="metric"><b>%(steps)s</b>新增实际原生子步</div><div class="metric"><b>%(stop)s</b>制动到连续停稳确认</div><div class="metric"><b>0</b>30/120/600秒物理随机完成</div><div class="metric"><b>0</b>新增训练与合格完整任务</div></div>
<section><h2>执行错误与修复</h2><p>旧分支返回“没有合格候选”后仍继续执行一个平滑目标。1920Hz对照112从第15控制周期开始出现该状态，随后最小代理间隙在0.766667秒变为负值。113在第15周期锁存紧急制动，每个物理子步用当前实际关节位置作为原50电机控制器的目标；没有写回关节位置、速度或物体状态，仿真继续推进。</p><p>112与113使用同一初态、同一随机臂指令及FIFO6，前480个原生子步的全部原字段逐值相同。原电机控制器、模型、限位、速度、力和位置/速度联动门槛保持一致。紧急制动明确覆盖普通命令的速率限制。</p><p><b>本轮随机输入仍只有26个臂关节；24个独立手指电机参与制动，没有随机手指输入。</b>单一场景的制动结果不能作为通用制动包络、遥操或硬件安全证明。</p></section>
<section><h2>全部新对照与失败</h2><div class="scroll"><table><thead><tr><th>候选</th><th>实际子步/时长</th><th>物理Hz</th><th>原物理门槛</th><th>原q且qd关系</th><th>最小间隙/mm</th><th>最大qd关系误差/rad/s</th><th>四臂实际运动</th></tr></thead><tbody>%(tab)s</tbody></table></div><p class="small">111启动时钟绑定失败，实际执行0个受控物理步，外层退出1、子进程退出0；不计物理试验或安全成功。112是修复时钟绑定后的执行复本，原失败保留。脚本退出0与安全门槛通过分开记录。</p></section>
<section><h2>同条件制动对照与停稳</h2><img src="figures/brake_pair.svg" alt="112与113原生时间下的最小间隙和全部74关节最大速度"><p>虚线标出制动触发时间；停稳要求全部74个实际关节速度连续0.3秒≤0.02 rad/s。原始速度及间隙逐物理子步审计。q与qd原关系容差均为10⁻⁴，最小代理间隙必须≥0。</p><p>制动后实际位移、停稳确认时刻及全部原门槛见<a href="evidence/v113_INDEPENDENT_EMERGENCY_BRAKE_V1.json">独立停稳记录</a>。这条曲线仅说明本次已知场景，不表示随机运动已充分覆盖。</p></section>
<section><h2>原生四视角视频</h2><p>每个版本包含俯视、正视、侧视和全景。视频保留实际帧时间，没有合成抓取动作；原生帧与编码时间最大偏差小于1.1微秒。</p><button id="slow" type="button">全部视频0.2×慢放</button> <button id="normal" type="button">恢复1×</button><p class="small">慢放只改变观看速度，实验时长以原生记录为准。这里展示短时修复和停止；大范围动作及完整抓取任务尚未取得通过证据。</p>%(videos)s</section>
<section><h2>长时间随机输入准备与物理执行</h2><p>已生成50维（26臂+24手指）连续输入：10种运动过程 × 30/120/600秒 × 每层2个独立开发种子，共60条、900060个请求样本、15000秒请求时长。种子和时钟贯穿整条轨迹，不靠拼接或每块重置。</p><p>逐周期独立IID在所有50个电机原关节范围上的请求跨度超过99%%。突发微振荡的跨度较小，单独记录；其余过程也按各自分布记账。<b>这些数值是指令支持，不是已执行运动范围或安全率。</b></p><div class="scroll"><table><thead><tr><th>运动过程</th><th>时长/种子</th><th>已生成连续输入</th><th>物理长测完成</th></tr></thead><tbody>%(families)s</tbody></table></div><p><a href="evidence/LONG_RANDOM_INPUT_AUDIT_V1.json">分布、范围、时间连续性与每条输入SHA</a>。下一步须验证含手指运动的预测几何与制动包络、持续原生记录，再执行长时间多场景随机压力测试；仅把手停住无法满足运动覆盖。</p></section>
<section><h2>联合验收与训练准入</h2><p>九项联合要求全部未完成。真实摩擦夹持的闭合、抬起、搬运、放置、释放与退离需形成合格完整任务；附着或姿态写入不计入任务数据。当前没有合格的新任务样本，不启动新学习System0训练。</p><ul>%(criteria)s</ul></section>
<section><h2>可复核数据</h2><ul><li><a href="STATE_GUARD_RESULTS_V1.json">全部结果和制动记录</a></li><li><a href="PER_NATIVE_STEP_V1.csv">逐原生子步指标CSV</a></li><li><a href="MEDIA_CUSTODY_V1.json">四视角视频帧时钟和SHA</a></li><li><a href="RAW_ARCHIVE_MANIFEST_V1.json">原始分包、每项文件SHA与索引</a></li><li><a href="../safety_motion_random_20261012/">原始107/108反例与106实际四臂运动</a></li><li><a href="source/audit_emergency_brake_v1.py">独立制动审计源码</a> · <a href="source/actual_state_arm_guard_v2.py">实际状态防护源码</a> · <a href="source/long_motion_stream.py">连续随机输入源码</a></li>%(downloads)s</ul><p class="small">原始分包含原生q/qd、全部9021代理间隙、接触、质量矩阵、实际提交目标、时钟和原始PNG。代理间隙审计不等于完整碰撞网格认证。</p></section>
</main><script>document.getElementById('slow').onclick=()=>document.querySelectorAll('video').forEach(v=>v.playbackRate=.2);document.getElementById('normal').onclick=()=>document.querySelectorAll('video').forEach(v=>v.playbackRate=1);</script></html>'''
    values=dict(verdict=verdict,steps=results['new_controlled_native_microsteps'],stop=stop,tab=tab,videos=videos,families=families,criteria=criteria,downloads=downloads)
    (P/'index.html').write_text(re.sub(r'%\((\w+)\)s',lambda m:str(values[m[1]]),doc.replace('99%%','99%')))
    root=W/'index.html';before=root.read_bytes();assert b'id="actualStateBrakeEvidence"' not in before
    (D/'DASHBOARD_INDEX_BEFORE_STATE_GUARD_V1.html').write_bytes(before)
    card=f'<div id="actualStateBrakeEvidence" class="sci-note" style="margin-bottom:18px;border-left:4px solid #167251"><strong>最新原生修复：{verdict}</strong><p>新增4条受控原生记录与1次无效启动；无合格候选时全50电机制动，实际停稳和全部失败可复核。全50随机手指、长时间大范围运动、完整抓取任务与学习System0仍未验收。</p><a href="docs/safety_state_guard_20261012/">查看制动对照、16路原生视频与原始数据 →</a></div>\n'
    marker=b'<div id="motionRandomEvidence"';assert before.count(marker)==1
    root.write_bytes(before.replace(marker,card.encode()+marker,1));assert root.read_bytes().replace(card.encode(),b'',1)==before
    write(D/'DASHBOARD_ROOT_INSERTION_V1.json',{'before_sha256':hashlib.sha256(before).hexdigest(),'inserted_card':card,'old_content_byte_preserved':True})
    files=[]
    for f in sorted(P.rglob('*')):
        if f.is_file():files.append({'path':str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':sha(f)})
    write(P/'PUBLIC_MANIFEST_V1.json',{'files':files,'total_bytes':sum(r['bytes'] for r in files),'source_builder_sha256':sha(__file__),'fullSystem0_accepted':False})
    print('STATE_GUARD_REPORT_BUILT',len(files),'files',sum(r['bytes'] for r in files),'bytes',len(media),'videos',results['new_controlled_native_microsteps'],'native_steps',flush=True)


if __name__=='__main__':main()
