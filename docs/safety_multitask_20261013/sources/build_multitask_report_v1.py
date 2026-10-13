"""Publish measured task development evidence; never substitute plans for outcomes."""
from pathlib import Path
import hashlib,html,json,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_loaded_duration_20261012')
P=W/'docs/safety_multitask_20261013'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not P.exists()
    outcomes=[read(K/f'NATIVE{v}_INDEPENDENT_PAIR_TASK_V1.json') for v in [118,119]]
    for v in [118,119]:
        r=read(H/f'fourhand_native_probe_v{v}/RESOURCE_EXECUTION_V1.json')
        assert r['actual_exit']==r['actual_native_exit']==0 and r['native_owned_process_reaped_and_absent']
    P.mkdir();(P/'data').mkdir();(P/'sources').mkdir();(P/'videos').mkdir()
    evidence=[]
    names=['NATIVE105_CORRECTED_OPPOSITION_V2.json','NATIVE117_CAPTURE_CAPACITY_COMPATIBILITY_V1.json',
        'PAIR_TASK_SCENE_ADMISSION_V1.json','PAIR_TASK_SCENE_ADMISSION_V2.json','PAIR_TASK_OPENING_ADMISSION_V1.json',
        'PAIR_TASK_OPENING_ADMISSION_V2.json','PAIR_TASK_RAISED_ADMISSION_V3.json','NATIVE118_PAIR_TASK_PROSPECTIVE_V1.json',
        'NATIVE119_RAISED_TASK_PROSPECTIVE_V1.json','NATIVE118_INDEPENDENT_PAIR_TASK_V1.json','NATIVE119_INDEPENDENT_PAIR_TASK_V1.json',
        'NATIVE118_INDEPENDENT_TASK_TRACE_V1.npz','NATIVE119_INDEPENDENT_TASK_TRACE_V1.npz','PAIR_TASK_NATIVE_SEED_V1.npz',
        'PAIR_TASK_RAISED_NATIVE_SEED_V3.npz','PAIR_TASK_IK_ENDPOINTS_V2.npz','ADR_CONTINUOUS_TASK_CAPTURE_V1.md',
        'CURRENT_TASK_CONTROLLER_BINDINGS_V1.json','NATIVE119_CONTACT_FORCE_FRONTIER_V1.json']
    for name in names:shutil.copyfile(K/name,P/'data'/name)
    for file in sorted(K.glob('*.py')):shutil.copyfile(file,P/'sources'/file.name)
    camera_count=0
    for v in [118,119]:
        probe=H/f'fourhand_native_probe_v{v}';raw=Path(read(probe/'RESOURCE_EXECUTION_V1.json')['out'])
        for name in ['REGISTRATION_V1.json','RESOURCE_EXECUTION_V1.json','INDEPENDENT_CAPTURE_AUDIT_V1.json',
            'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json','INDEPENDENT_FULL_MOTOR_EFFORT_V4.json']:
            shutil.copyfile(probe/name,P/'data'/f'native{v}_{name}')
        files=[file for file in raw.rglob('*') if file.is_file() and file.suffix in ['.npz','.json','.jsonl','.png']]
        # Compression handles the repetitive launch ledger; split compressed
        # archives at file boundaries and preserve each original file digest.
        rows=[];archive_index=0;archive=None;archive_name=None
        def close_archive():
            if archive is not None:archive.close()
        for file in sorted(files):
            if archive is None or (P/'data'/archive_name).stat().st_size>40*1024**2:
                close_archive();archive_name=f'native{v}_raw_part{archive_index:02d}.zip';archive_index+=1
                archive=zipfile.ZipFile(P/'data'/archive_name,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1)
            rel=file.relative_to(raw).as_posix();archive.write(file,rel)
            rows.append({'path':rel,'archive':archive_name,'bytes':file.stat().st_size,'sha256':sha(file)})
        close_archive()
        for name in {r['archive'] for r in rows}:
            assert (P/'data'/name).stat().st_size<49*1024**2
            with zipfile.ZipFile(P/'data'/name) as z:
                for row in [r for r in rows if r['archive']==name]:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
        (P/'data'/f'native{v}_raw_manifest.json').write_text(json.dumps({'source_version':v,'files':rows},indent=2)+'\n')
    video=read(H/'fourhand_native_probe_v119/video_evidence_v6/VIDEO_CUSTODY_V1.json')
    assert video['status']=='PASS_ALL4_NATIVE_FRAMES_AND_EXACT_VIDEO_TIMESTAMPS'
    for view in video['views']:shutil.copyfile(view['path'],P/'videos'/(view['view']+'.mp4'))
    shutil.copyfile(H/'fourhand_native_probe_v119/video_evidence_v6/VIDEO_CUSTODY_V1.json',P/'data/VIDEO_CUSTODY_V1.json')
    fig,axes=plt.subplots(3,2,figsize=(11,8),constrained_layout=True)
    for column,v in enumerate([118,119]):
        z=np.load(K/f'NATIVE{v}_INDEPENDENT_TASK_TRACE_V1.npz');t=z['time_s']
        axes[0,column].plot(t,z['mimic_qd_error_rad_s']);axes[0,column].axhline(1e-4,color='r',linestyle='--',label='Original1e-4 gate')
        axes[0,column].set_title(f'Native{v}: measured original mimic velocity error');axes[0,column].set_ylabel('rad/s');axes[0,column].legend(fontsize=8)
        for i,label in enumerate(['F_L','F_R','U_L','U_R']):axes[1,column].plot(t,z['hand_force_N'][:,i],label=label)
        axes[1,column].set_ylabel('Payload normal force / N');axes[1,column].legend(fontsize=8)
        axes[2,column].plot(t,z['lift_m']*1000);axes[2,column].axhline(100,color='r',linestyle='--');axes[2,column].set_ylabel('Object lift / mm');axes[2,column].set_xlabel('Native physical time / s')
        for ax in axes[:,column]:ax.grid(alpha=.25)
    fig.savefig(P/'task_physics_curves.svg');fig.savefig(P/'task_physics_curves.png',dpi=150);plt.close(fig)
    admission=read(K/'PAIR_TASK_SCENE_ADMISSION_V2.json');opposed=read(K/'NATIVE105_CORRECTED_OPPOSITION_V2.json')
    case_rows=''.join(f'<tr><td>Native{o["native_version"]}</td><td>{o["simulation_duration_s"]:.6f} s / {o["physics_events"]}步</td><td>{html.escape(o["protocol_task_status"])}</td><td>{o["max_object_lift_m"]*1000:.3f} mm</td><td>{"通过" if o["independent_stage_verdict"]["safety_pass"] else "未通过"}</td><td>{"完成" if o["independent_stage_verdict"]["task_pass"] else "未完成"}</td></tr>' for o in outcomes)
    direction_rows=''.join(f'<tr><td>{r["candidate"]}</td><td>{r["xy_displacement_m"][0]:+.3f}, {r["xy_displacement_m"][1]:+.3f}</td><td>{r["minimum_proxy_gap_m"]*1000:.3f} mm</td><td>{"源模型准入" if r["admitted"] else "拒绝；没有作为完整路线执行"}</td></tr>' for r in admission['candidates'])
    videos=''.join(f'<figure><video controls preload="metadata" src="videos/{v["view"]}.mp4"></video><figcaption>{html.escape(v["view"])} · {v["source_frames"]}个原生帧 · 最后物理时刻{v["source_last_time_s"]:.6f}s</figcaption></figure>' for v in video['views'])
    data_links=''.join(f'<li><a href="data/{html.escape(file.name)}">{html.escape(file.name)}</a></li>' for file in sorted((P/'data').iterdir()) if file.suffix in ['.json','.zip','.npz','.md'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo 多任务操作实验</title>
<style>body{{margin:0;background:#f4f6fa;color:#172333;font:16px/1.65 system-ui}}main{{max-width:1120px;margin:auto;padding:24px}}h1,h2{{line-height:1.3}}a{{color:#2459af;overflow-wrap:anywhere}}.alert{{background:#fff1df;border-left:5px solid #a75108;padding:16px}}.metrics,.videos{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.metric,section,figure{{background:white;padding:18px;border-radius:10px;margin:16px 0}}.metric b{{display:block;font-size:30px}}video,img{{width:100%;max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere}}@media(max-width:600px){{main{{padding:14px}}.metrics,.videos{{grid-template-columns:1fr}}}}
</style><main><a href="../../#multitaskTaskEvidence">← 主面板</a><h1>多任务操作：从真实接触到完整验收</h1>
<p>开发实验 Native118/119 · 单物体双手协同搬运 · 实验配置2FR3/F2＋2UR5/DFX。研究状态保留失败；尚无完整安全任务、学习System0或独立留出测试。</p>
<div class="alert">任务完成数仍为0。控制器、几何规划和录像均不能替代真实抓取、100毫米抬升保持2秒、200毫米搬运、支撑放置、实际释放及撤离。模拟窗口结束与真实机器人安全制动分别判断。</div>
<div class="metrics"><div class="metric"><b>3＋2</b>3类单物体角色控制；2种多物体调度组合。多物体原生观测适配尚未绑定。</div><div class="metric"><b>2</b>同一开发场景的两个原生任务试验；不是两个独立场景或留出样本。</div><div class="metric"><b>0</b>完整安全操作任务；合格训练任务数据0。</div><div class="metric"><b>2 / 8</b>八方向离散几何路线准入；源模型结果，不是物理成功率。</div></div>
<section><h2>已实现的操作与边界</h2><p>单手抓放、双手共同抬举、四手共同抬举共用按原生接触反馈推进的六阶段控制器。两组双手搬运、四手分别抓放通过多物体调度组合表达，检查物体身份和手臂占用冲突；当前原生任务执行适配仅绑定双手梁体。其余任务仍需各自物体场景、观测和完成条件。</p><p>任务目标、物体位姿、速度、尺寸、质量、接触法向、真实手形和角色进入控制输入。50电机意图经过真实FIFO6与实际74关节几何检查。没有运行时物体位姿写入、关节状态投影或手—物固定连接。</p></section>
<section><h2>原生结果与失败</h2><div class="scroll"><table><thead><tr><th>实验</th><th>实际记录窗口</th><th>状态</th><th>最大抬升</th><th>全部安全／模型门槛</th><th>完整任务</th></tr></thead><tbody>{case_rows}</tbody></table></div><p>Native118从原有接触状态初始化；Native119仅在初始化时把下方两手抬高60毫米，保留全部手指、速度、对象、动力学和控制器。这是两个启动／接近开发窗口，不能作为完整任务成功率或两个独立场景的统计样本。</p><p><strong>Native118：</strong>原始手指联动速度门槛失败。<strong>Native119：</strong>下方两臂实际关节运动约0.077rad，上方两臂保持；在1.11875秒，左手食指与梁顶部接触力由前一步2.89N跃升至63.29N，超过30N；这次联动q、qd本身通过原门槛。物体未被抬起。两种失败分别保留。</p><a href="task_physics_curves.svg">下载原始矢量图</a><img src="task_physics_curves.png" alt="两个原生任务试验的手指联动速度误差、接触力及物体抬升曲线"></section>
<section><h2>接触安全策略候选：尚未原生验证</h2><p>失败前实际手臂尚距抓取端点0.089／0.110rad，控制器已经推进手指加压。新增V2候选等待真实关节到位、速度及已提交端点连续稳定0.3秒，再推进压力；接近目标速率降至0.02rad/s，手指压力目标速率0.005rad/s。5项软件契约通过，原生验证0次。</p><p>12N只能触发停止意图；本样本在一个物理步内由2.89N跳至63.29N，因此反应阈值不能证明瞬时力安全。停止意图仍需在实际执行层绕过延迟FIFO并验证制动。下一步还需重新规划手—物插入路径、接触前速度与冲量预算；不把降速、停止指令或单个失败解释当成安全证明。</p></section>
<section><h2>四视角原生短窗</h2><p>下列录像来自Native119的俯视、正视、侧视和总览相机。只展示实际记录的启动／接近窗口，不计作已完成的抓取抬举。视频时间逐帧与物理时间核对。</p><div class="videos">{videos}</div></section>
<section><h2>可达性与路线筛选</h2><p>目标平移220毫米；八个方向在原始关节范围、固定末端姿态和全部9021几何对下检查。每条路线404个源模型状态。保留所有拒绝路线；选择预定角度顺序中第一个通过者。静态手形和离散采样不能证明连续带载碰撞安全。</p><div class="scroll"><table><thead><tr><th>候选</th><th>XY目标平移 / m</th><th>最小间隙</th><th>准入</th></tr></thead><tbody>{direction_rows}</tbody></table></div></section>
<section><h2>修正的测量与连续记录</h2><p>旧接触判断把长梁两端的同侧纵向坐标误当成“没有对向夹持”。原始Native105重算后，下方左右两手的对向接触分别持续{opposed['per_arm']['F_L']['longest_continuous_opposition_s']:.3f}秒和{opposed['per_arm']['F_R']['longest_continuous_opposition_s']:.3f}秒，上方两手仍为0。对向接触不是摩擦力闭合证明，原始完整任务失败保留。</p><p>宿主记录容量8192→65536，不改原生求解和16次额外迭代。14项CPU边界检查通过；Native117的832个物理步、6122个分块数组字段与原始115逐字节一致，旧联动失败保留。容量对应34.133秒的当前物理时钟；尚未通过长时任意运动验收，30／120／600秒随机实测均未因此获得成功计数。</p></section>
<section><h2>数据、方法与后续验收</h2><p>原始24项手指联动q、qd误差各≤1e−4；全部74关节硬限位／速度超量≤1e−5；9021代理几何间隙≥0；非预期原生接触≤0.1N；单手载荷法向≤30N。代理几何不是整个真实碰撞网格的完整证明。正式验收还需要真实任务语料、多场景种子、完整时间范围与独立冻结比较。</p><p><a href="../safety_full50_motion_20261013/">全50电机随机动作与接触故障证据</a> · <a href="../safety_motion_random_20261012/">随机运动安全／失败原始对照</a></p><details><summary>下载原始记录、独立判定、场景与协议</summary><ul>{data_links}</ul></details><a href="PUBLIC_MANIFEST_V1.json">完整交付SHA256清单</a></section></main></html>'''
    (P/'index.html').write_text(page)
    mainfile=W/'index.html';before=mainfile.read_text();anchor='<div id="full50MotionEvidence"'
    assert before.count(anchor)==1 and 'id="multitaskTaskEvidence"' not in before
    card='<div id="multitaskTaskEvidence" class="sci-note" style="margin-bottom:18px;border-left:4px solid #a75108"><strong>最新多任务操作：双手接近触发63N接触力失败</strong><p>单手／双手／四手阶段控制和多物体调度接口已补；原生实测为双手启动与接近，四手任务尚未验收。四视角录像、力曲线与候选策略分别列出，完整安全任务仍为0。</p><a href="docs/safety_multitask_20261013/">查看多任务操作、原生录像与独立验收 →</a></div>\n'
    mainfile.write_text(before.replace(anchor,card+anchor,1))
    assert mainfile.read_text().replace(card,'',1)==before
    files=[{'path':f.relative_to(P).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(P.rglob('*')) if f.is_file()]
    assert all(r['bytes']<49*1024**2 for r in files)
    (P/'PUBLIC_MANIFEST_V1.json').write_text(json.dumps({'report':'safety_multitask_20261013','files':files,'fullSystem0_accepted':False},indent=2)+'\n')
    (K/'MULTITASK_REPORT_BUILD_V1.json').write_text(json.dumps({'files':len(files)+1,'bytes':sum(r['bytes'] for r in files),
        'source_manifest':str(P/'PUBLIC_MANIFEST_V1.json'),'main_card_added_only':True,'complete_safe_task_successes':sum(o['complete_task_successes'] for o in outcomes),
        'fullSystem0_accepted':False},indent=2)+'\n')
    print('MULTITASK_REPORT_BUILT;full original raw custody;old main content preserved;no task or safety acceptance inferred',flush=True)


if __name__=='__main__':main()
