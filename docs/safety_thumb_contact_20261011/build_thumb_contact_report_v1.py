"""Publish closed short diagnoses, with original gates and complete raw custody."""
from pathlib import Path
import copy
import datetime
import hashlib
import html
import json
import shutil
import subprocess
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
P = H / 'thumb_contact_delivery_v1'
W = Path('/home/liyufeng/safeduo-dashboard-thumb-contact-20261011')
SLUG = 'docs/safety_thumb_contact_20261011'
BASE = '8e3f5e3c218592b31efe183ad803606e4394bbd1'
VERSIONS = [61, 62, 63, 64, 65, 66, 69, 70, 71, 73, 74, 75, 76, 77]
LABELS = {61:'原控制',62:'改为链式引用',63:'位置迭代128',64:'速度迭代255',65:'根速度迭代1',66:'六个手部PD置零',69:'六个手部PD×0.1',70:'六个手部PD×0.5',71:'GPU1原控制复核',73:'等价反向约束',74:'仅拇指屈曲PD置零',75:'仅拇指对掌PD置零',76:'根速度迭代0',77:'根与场景速度迭代上限0'}
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not P.exists() and not (W / SLUG).exists()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip() == BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W).strip()
    P.mkdir()
    out = W / SLUG
    out.mkdir()
    rows, original_sources, all_force, point_contact_bodies = [], [], [], set()
    baseline_raw = Path(read(H / 'fourhand_native_probe_v71/RESOURCE_EXECUTION_V1.json')['out'])
    for version in VERSIONS:
        probe = H / f'fourhand_native_probe_v{version}'
        actual = read(probe / f'ACTUAL_COMPLETION_EXIT_V{version}.json')
        assert all(type(actual[k]) is int and actual[k] == 0 for k in ['actual_outer_exit','actual_native_exit','actual_finite_exit','actual_launcher_subprocess_exit'])
        capture = read(probe / 'INDEPENDENT_CAPTURE_AUDIT_V1.json')
        assert capture['record_integrity_verified'] and capture['all_owned_native_children_reaped']
        model_name = 'INDEPENDENT_NATIVE_MIMIC_WINDOW_V8' if version == 73 else 'INDEPENDENT_NATIVE_MIMIC_WINDOW_V7' if version == 62 else 'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6'
        model = read(probe / (model_name + '.json'))
        effort_name = 'INDEPENDENT_SINGLE_THUMB_PD_EFFORT_V1' if version in [74,75] else 'INDEPENDENT_SHORT_HAND_PD_PAUSE_EFFORT_V1' if version == 66 else 'INDEPENDENT_SHORT_HAND_PD_SCALE_EFFORT_V1' if version in [69,70] else 'INDEPENDENT_FULL_MOTOR_EFFORT_V4'
        for name in ['INDEPENDENT_CAPTURE_AUDIT_V1',model_name,effort_name,'INDEPENDENT_SHORT_CONTACT_TARGET_TAPE_V1']:
            assert read(probe / (name + '_ACTUAL_EXIT.json'))['actual_outer_exit'] == 0
        raw = Path(read(probe / 'RESOURCE_EXECUTION_V1.json')['out'])
        for name in ['native_parameters.npz','actual_initial_seed_binding.npz']:
            with np.load(raw / name, allow_pickle=False) as candidate, np.load(baseline_raw / name, allow_pickle=False) as control:
                assert set(candidate.files) == set(control.files)
                for key in candidate.files:
                    np.testing.assert_array_equal(candidate[key], control[key], err_msg=key)
        case = out / f'native_v{version}'
        case.mkdir()
        shutil.copytree(raw, case / 'raw')
        for f in sorted(raw.rglob('*')):
            if f.is_file():
                original_sources.append(dict(source=str(f),public_path=(case/'raw'/f.relative_to(raw)).relative_to(W).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
        records = case / 'audit'
        records.mkdir()
        for f in probe.glob('*.json'):
            shutil.copyfile(f, records / f.name)
            original_sources.append(dict(source=str(f),public_path=(records/f.name).relative_to(W).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
        archive = case / 'source_archive.zip'
        member_rows = []
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(probe.iterdir()):
                if f.suffix in ['.py','.usda']:
                    z.writestr(f.name,f.read_bytes())
                    member_rows.append(dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)))
        with zipfile.ZipFile(archive) as z:
            for item in member_rows:
                value=z.read(item['name'])
                assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
        (case / 'SOURCE_ARCHIVE_CUSTODY_V1.json').write_text(json.dumps(dict(source_bytes_unchanged=True,archive_sha256=sha(archive),members=member_rows),indent=2)+'\n')
        forces=[]
        for chunk in read(raw / 'physics_chunks.json'):
            f=raw/chunk['path'];assert sha(f)==chunk['sha256']
            with np.load(f,allow_pickle=False) as z:forces.append(z['four_hand_normal_N'].copy())
        force=np.concatenate(forces);assert len(force)==capture['physics_events']
        assert not (force>.1).any(), 'Report assertion forbids accidentally labelling a loaded case as weak.'
        all_force.append(force)
        rows.append(dict(version=version,label=LABELS[version],physics_events=len(force),controls=capture['completed_controls'],
                         maximum_q_residual_rad=model['maximum_q_residual_rad'],maximum_qd_residual_rad_s=model['maximum_qd_residual_rad_s'],
                         original_model_pass=model['model_window_pass'],original_physical_pass=capture['physical_gates_pass'],
                         maximum_F_L_normal_N=float(force[:,0].max()),F_L_positive_events=int((force[:,0]>0).sum()),
                         all4_original_contact_window_s=capture['measured_stage_windows_s']['four_hand_close'],
                         original_task_pass=False,actual_outer_exit=0,all77_native_arrays_and13_initial_state_arrays_equal71=True))
    assert len(rows)==14 and sum(r['physics_events'] for r in rows)==304
    summary=dict(status='CLOSED_KNOWN_STATE_NATIVE_SHORT_DIAGNOSTICS_FULL_REQUIREMENTS_NOT_ACCEPTED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 rows=rows,native_closed_records=14,total_recorded_physics_events=304,unique_known_initial_cases=1,
                 maximum_physics_events_per_record=64,maximum_duration_s=.066664,original_q_and_qd_tolerance=1e-4,
                 original_four_hand_contact_normal_N=.1,original_four_hand_contact_window_s=.25,
                 original_physics_assets='Experimental2FR3/F2+2UR5/DFX; canonical2FR3+2JAKA+4F2 not qualified',
                 reset_object_writes_after_initialization=False,q_or_qd_state_projection=False,
                 known_case_solver_warm_start_not_restored=True,all9_jointly_accepted=False,
                 new_independent_native_random_cases=0,formal_holdout_count=0,learned_system0_tested=False,fullSystem0_accepted=False,
                 excluded_attempts=[dict(version=67,native_launches=0,outer_exit=241,reason='unlaunched scheduler cancellation'),dict(version=68,native_launches=0,reason='superseded before launch'),dict(version=72,physics_events=0,outer_exit=1,native_exit=0,reason='setup metadata getter incompatibility')])
    (out / 'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    extras=['THUMB_CONTACT_COORDINATES_POSTHOC_V2.json','THUMB_CONTACT_COORDINATES_POSTHOC_V2_ACTUAL_EXIT.json',
            'THUMB_CONTACT_COORDINATES_V1_FAILURE_PRESERVED.json','diagnose_thumb_contact_coordinates_v1.py','diagnose_thumb_contact_coordinates_v2.py',
            'diagnose_source_hand_gravity_v1.py','prepare_single_thumb_pd_isolation_v74.py','audit_single_thumb_pd_effort_v1.py','audit_single_thumb_pd_decision_v1.py',
            'SINGLE_THUMB_PD74_75_PROSPECTIVE_PLAN_V1.json','TGS_VELOCITY_STAGE76_PROSPECTIVE_PLAN_V1.json','TGS_SCENE_CAP77_PROSPECTIVE_PLAN_V1.json',
            'prepare_tgs_velocity_stage_isolation_v76.py','prepare_tgs_scene_cap_isolation_v77.py','SHORT_CONTACT_DEVELOPMENT_LEDGER_V1.json']
    for relative in extras:
        source=H/relative;shutil.copyfile(source,out/source.name)
        original_sources.append(dict(source=str(source),public_path=(out/source.name).relative_to(W).as_posix(),bytes=source.stat().st_size,sha256=sha(source)))
    for version, names in [(67,['UNLAUNCHED_SCHEDULER_CANCELLATION_PLAN_V1.json','UNLAUNCHED_SCHEDULER_CANCELLATION_ACTUAL_EXIT_V1.json']),
                           (68,['REGISTRATION_V1.json']),
                           (72,['ACTUAL_INCOMPLETE_SETUP_EXIT_V72.json','STATIC_USD_LOADER_UNAVAILABLE_V1.json'])]:
        destination=out/f'excluded_attempt_v{version}';destination.mkdir()
        for name in names:
            source=H/f'fourhand_native_probe_v{version}'/name
            assert source.is_file(),source
            shutil.copyfile(source,destination/name)
            original_sources.append(dict(source=str(source),public_path=(destination/name).relative_to(W).as_posix(),bytes=source.stat().st_size,sha256=sha(source)))
    matrix=copy.deepcopy(read(H/'full_requirements_v1/EVIDENCE_MATRIX_V25.json'))
    oldrows=copy.deepcopy(matrix['rows'])
    for row in matrix['rows']:
        row['joint_requirement_accepted']=False
        if row['id'] in ['grasp','cooperative_task','native_oracle','visual_delivery']:
            row['measured']=dict(prior_revision25=row['measured'],revision26_short_diagnostics=summary)
            row['status']='FAIL_LOADED_ORIGINAL_MODEL_FULLTASK_CANONICAL_AND_FORMAL_PENDING'
    for before,after in zip(oldrows,matrix['rows']):
        for key in ['id','requirement','gate']:assert before[key]==after[key]
    matrix.update(revision=26,updated_utc=summary['created_utc'],every_required_row_jointly_accepted=False,fullSystem0_accepted=False,formal_holdout_acceptance_count=0,
                  active_native_excluded=[],latest_development_frontier=summary,no_independent_native_random_input_added=True)
    (out/'EVIDENCE_MATRIX_V26.json').write_text(json.dumps(matrix,indent=2,ensure_ascii=False)+'\n')
    matrix_local=H/'full_requirements_v1/EVIDENCE_MATRIX_V26.json';assert not matrix_local.exists();shutil.copyfile(out/'EVIDENCE_MATRIX_V26.json',matrix_local)
    fig,axes=plt.subplots(1,2,figsize=(15,5.4),layout='constrained')
    labels=[str(r['version']) for r in rows];x=np.arange(len(rows))
    axes[0].bar(x,[r['maximum_qd_residual_rad_s'] for r in rows],color=['#b45309' if r['original_model_pass'] else '#b91c1c' for r in rows])
    axes[0].set_yscale('log');axes[0].axhline(1e-4,color='#111827',ls='--',label='Original qd gate 1e-4 rad/s');axes[0].legend()
    axes[0].set_title('Native original-mimic velocity residual');axes[0].set_ylabel('Maximum residual (rad/s)')
    axes[1].bar(x,[r['maximum_F_L_normal_N'] for r in rows],color='#2563eb');axes[1].axhline(.1,color='#111827',ls='--',label='Original per-hand contact minimum 0.1 N');axes[1].legend()
    axes[1].set_title('Weak left-thumb contact only');axes[1].set_ylabel('Peak F_L normal force (N)')
    for ax in axes:ax.set_xticks(x,labels);ax.set_xlabel('Diagnostic version (one known seed; not independent random trials)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('No loaded grasp qualified. Six-motor PD0 record66 loses contact. Formal System0 count=0.',fontsize=12)
    fig.savefig(out/'diagnostic_gate_comparison.png',dpi=160);fig.savefig(out/'diagnostic_gate_comparison.svg');plt.close(fig)
    gravity=read(H/'fourhand_native_probe_v71/POSTHOC_ORIGINAL_HAND_GRAVITY_ORACLE_V1.json')
    rowhtml=''.join('<tr>'+''.join(f'<td>{value}</td>' for value in [r['version'],html.escape(r['label']),r['physics_events'],f"{r['maximum_qd_residual_rad_s']:.9g}",'通过' if r['original_model_pass'] else '未通过',f"{r['maximum_F_L_normal_N']:.6f}",f"{r['F_L_positive_events']}/{r['physics_events']}",'未完成'])+'</tr>' for r in rows)
    matrixhtml=''.join(f"<tr><td>{html.escape(r['requirement'])}</td><td>{html.escape(r['gate'])}</td><td>联合未验收</td></tr>" for r in matrix['rows'])
    image_manifest=read(baseline_raw/'camera_frames.json')
    camera_record=image_manifest[-1]
    imagehtml=''
    for view,title in [('top','俯视'),('front','正视'),('side','侧视'),('overview','全景')]:
        candidates=sorted((baseline_raw/'frames').glob(f'*{view}*'))
        if not candidates:candidates=sorted((baseline_raw/'frames'/view).glob('*.png'))
        assert candidates and candidates[-1].is_file(),view
        relative=(out/'native_v71/raw'/candidates[-1].relative_to(baseline_raw)).relative_to(out).as_posix()
        imagehtml+=f'<figure><img src="{relative}" alt="原生诊断71的{title}静态帧" loading="lazy"><figcaption>{title} · 诊断71原生静态帧</figcaption></figure>'
    page=f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo · 拇指接触与原模型验收诊断</title>
<style>body{{font-family:system-ui,sans-serif;color:#172033;background:#f5f7fa;max-width:1200px;margin:auto;padding:24px;line-height:1.65}}a{{color:#1d4ed8}}h1{{font-size:clamp(22px,3vw,34px)}}section,.notice{{background:white;border:1px solid #d6dce7;border-radius:12px;padding:22px;margin:20px 0}}.notice{{border-left:6px solid #b91c1c}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d6dce7;text-align:left;vertical-align:top}}th{{background:#eef2f7}}img{{width:100%;height:auto}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0}}figcaption{{font-size:13px;color:#475569}}@media(max-width:600px){{body{{padding:12px}}section,.notice{{padding:16px}}.views{{grid-template-columns:1fr}}td,th{{padding:8px}}}}</style>
<p><a href="../../#scientific">← 主面板科研评测</a> · <a href="../safety_loaded_rate_20261011/">查看先前完整四路视频</a></p>
<h1>拇指接触、力矩与求解阶段的验收诊断</h1>
<div class="notice"><b>九项要求尚未全部达标。完整四臂任务未通过，正式System 0留出验收为0。</b><p>14条完成的短时诊断记录、合计304个物理子步，均来自同一个已知初态。原模型约束、质量和摩擦保持原验收要求；当前承载仅为左手拇指末节弱接触。原始记录、失败和实际退出码全部保留。</p><p>诊断66的短时模型约束通过，但仅2/64步存在正接触，随后失去接触；其四手夹持、抬升及完整任务未达标。其余13条记录的原模型速度约束均未通过。</p></div>
<section><h2>实测结果与固定验收线</h2><p>原始24条联动关系逐物理步检查：位置与速度残差均≤0.0001。正式抓取还要求四手各&gt;0.1 N连续≥0.25 s、摩擦余量以及后续100 mm抬升等完整阶段。这里最长单条记录只有0.066664 s；全部记录的四手有效抓取窗口均为0。</p><img src="diagnostic_gate_comparison.png" alt="14条诊断的速度残差和左手接触力，含原始验收线"><p><a href="diagnostic_gate_comparison.svg">下载矢量图</a> · <a href="SUMMARY_V1.json">完整机器可读结果</a></p><div class="scroll"><table id="results"><thead><tr><th>版本</th><th>单项变化</th><th>物理步</th><th>最大速度残差rad/s</th><th>原模型约束</th><th>左手峰值N</th><th>正接触步</th><th>完整任务</th></tr></thead><tbody>{rowhtml}</tbody></table></div><p>每条记录的原生几何、关节限位、速度限位与接触力上限检查通过；这仍未满足原模型与完整任务的联合门槛。PD变化仅为控制诊断，学习策略与System 0均关闭。</p></section>
<section><h2>已核查的机制证据</h2><p>用原始URDF全部26个手部零件的质量、质心和固定连接独立重建重力补偿，投影到六个独立电机后，最大差值为{gravity['maximum_reduced_hand_gravity_error_Nm']:.9g} Nm。未捕获原生局部质心，结论限于该已知轨迹的源模型比较。</p><p>全部正接触点归属于左拇指末节left_thumb_4。关节速度与原生身体角速度的独立核查已在先前报告完成，当前继续检查求解阶段和联动行之间的作用。</p><p>根速度迭代0的诊断76与根速度迭代1的诊断65，16步全部164个记录字段逐值一致。仅改根属性并未改变这条短轨迹。诊断77进一步限定场景速度迭代上限为0并记录各刚体迭代属性；配置元数据与实际二进制迭代次数仍需区分。</p><p>NVIDIA文档说明位置和速度迭代的方程及TGS阶段不同。这只是设计对照的依据；安装二进制与公开源码的精确提交对应关系尚未证明。<a href="https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/RigidBodyDynamics.html">官方求解器说明</a></p><p><a href="native_v71/audit/POSTHOC_ORIGINAL_HAND_GRAVITY_ORACLE_V1.json">重力补偿独立证据</a> · <a href="THUMB_CONTACT_COORDINATES_POSTHOC_V2.json">逐点接触与拇指坐标</a> · <a href="native_v76/audit/ROOT_VELOCITY0_VS1_ALL164_STATE_PARITY_V1.json">0与1迭代的逐字段对照</a></p></section>
<section><h2>四路原生静态帧与视频入口</h2><p>以下为诊断71的原生静态帧。短时诊断没有足够帧数展示完整任务；先前705周期的四路视频保持原版本与身份标记，可从顶部入口查看。</p><div class="views">{imagehtml}</div></section>
<section><h2>异常记录保留与实验范围</h2><p>67在等待资源阶段取消，未启动原生模拟；68未启动；72发生初始化观测API不兼容，物理步为0。这三次未计入14条完成记录。接触坐标读取器V1失败后，以V2重建折叠的固定掌部坐标；74力矩审计首次早于其前置审计，失败已保留，等待前置完成后运行原封存审计。</p><p>控制71在另一GPU上复核，与原控制61的全部164个物理记录字段一致。所有14条记录的77个原生参数数组及13个初态数组与71逐值相同；数值求解配置、约束表达或软件PD按各自预先声明改变。求解器暖启动缓存未恢复，不能称为完整动力学状态重放。</p><p>4万随机提案仍为软件空间采样，未新增独立原生随机案例；完整50维操作空间、标准2FR3+2JAKA+4F2、物体故障的原生整合及真正学习System 0同输入正式评测仍待完成。</p></section>
<section><h2>九项联合验收</h2><div class="scroll"><table id="matrix"><thead><tr><th>要求</th><th>原验收门槛</th><th>状态</th></tr></thead><tbody>{matrixhtml}</tbody></table></div><p><a href="EVIDENCE_MATRIX_V26.json">证据矩阵26</a> · <a href="PUBLIC_MANIFEST_V1.json">全部公开文件索引和哈希</a> · <a href="SOURCE_CUSTODY_V1.json">原始字节来源</a></p></section>
</html>'''
    (out/'index.html').write_text(page)
    root=W/'index.html';prior=root.read_text();needle='<div class="sci-note" id="loadedRateEvidence">';assert prior.count(needle)==1
    card='<div class="sci-note" id="thumbContactEvidence"><b>10-11 最新拇指接触诊断：原模型与完整四手任务仍未达标</b><p>14条同初态短时记录、304个物理子步；重力补偿、接触末节、单电机力矩和求解阶段逐项核查。全部失败及原门槛保留，正式System 0留出0。</p><a href="docs/safety_thumb_contact_20261011/">查看最新逐项诊断、原始记录和九项验收缺口 →</a></div>\n'
    root.write_text(prior.replace(needle,card+needle))
    assert root.read_text().replace(card,'',1)==prior
    (out/'SOURCE_CUSTODY_V1.json').write_text(json.dumps(dict(source_bytes_unchanged=True,source_entries=original_sources,no_new_random_cases=True,fullSystem0_accepted=False),indent=2)+'\n')
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    public_files=[dict(path=f.relative_to(W).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file()]
    manifest=dict(base_commit=BASE,slug=SLUG,public_files=public_files,old_tracked_paths=13041,previous_video_page_preserved=True,
                  native_closed_versions=VERSIONS,total_native_microsteps=304,unique_known_initial_cases=1,formal_holdout_count=0,all9_jointly_accepted=False,fullSystem0_accepted=False)
    (out/'PUBLIC_MANIFEST_V1.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (P/'BUILD_MANIFEST_V1.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('BUILT_THUMB_REPORT14_CLOSED_RECORDS304_STEPS',len(public_files)+1,sum(r['bytes'] for r in public_files),flush=True)


if __name__ == '__main__':
    main()
