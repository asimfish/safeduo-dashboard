"""Render numerical no-JS evidence and append one owned root card."""
from pathlib import Path
import argparse,json,html,re,hashlib,datetime
H=Path(__file__).resolve().parent
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_worktree_20261007')
A=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_assets_20261007')/H.name
def j(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(x):return html.escape(str(x))
def main(commit):
    r=j(H/'ACCEPTANCE_RESULT.json');peer=j(H/'ASTRA_ACCEPTANCE_REAL.json');c=j(H/'COVERAGE_RESULT.json');mechanism=j(H/'MECHANISM_RESULT.json');build=j(H/'PANEL_ASSET_BUILD.json')
    assert r['candidate_decision']==peer['candidate_decision'] and r['actual_windows_complete']==512
    base='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+commit+'/'+H.name+'/'
    titles=dict(joint_reference='原参考',zero_inclusive='零包含窄界',box_admission='完整指令界准入',adaptive_joint='联合可行性候选')
    scope='验收覆盖有限模拟中的原球模型、原生手部全伙伴聚合法向接触和实际运动量。候选采用须同时满足预登记门槛；整机、摩擦、构造阶段、未来非线性动态与硬件安全均未认证。'
    coverage=f"两批独立随机初态、128个不同配对初态、四方法512个完整窗口；每窗口960控制步/1920物理子步；七个风险分层。实际21视角原图{r['total_PNG']}张。初态经过旧球模型和零指令稳定性条件筛选；质量/摩擦/增益/布局/FIFO6固定，未进行动力学域随机化。"
    summary=dict(status='ADOPTION_FAILED' if r['candidate_decision']=='REJECTED' else 'ADOPTION_PASS',experiment_status=r['status'],candidate_decision=r['candidate_decision'],closed_windows=512,planned_windows=512,steps=960,unique_cases=128,modes=r['aggregated'],candidate_rejection_reasons=r['rejection_reasons'],scope=scope,coverage=coverage,queued_future_status='UNKNOWN',actual_motion_path_ratio=r['actual_motion_path_ratio'],actual_four_arm_movement_ratio=r['actual_four_arm_movement_ratio'],total_PNG=r['total_PNG'],criteria_sha256=r['criteria_sha256'],independent_decision=peer['candidate_decision'])
    galleries=j(A/'galleries.json');candidate=[g for g in galleries if g['mode']=='adaptive_joint'];assert candidate
    chosen=next((g for g in candidate if g['capture_kind']=='first_native_contact'),next((g for g in candidate if g['capture_kind']=='first_failure'),candidate[-1]))
    gallery='<p>默认实拍：'+esc(f"{chosen['job_id']} / env {chosen['env_id']} / step {chosen['step']} / {chosen['capture_kind']}")+'。原始状态绑定，完整21视角；原生接触报警图为宏步post状态。</p><div class="gallery-grid">'
    for im in chosen['images']:
        gallery+=f'<figure><a href="{base+im["path"]}" target="_blank" rel="noopener"><img src="{base+im["path"]}" width="1280" height="720" loading="lazy" style="width:100%;aspect-ratio:16/9;object-fit:contain" alt="{esc(im["label"])}"></a><figcaption>{esc(im["label"])}</figcaption></figure>'
    gallery+='</div><p><a href="'+base+chosen['state_url']+'">原始整体状态</a> · <a href="'+base+chosen['hand_state_url']+'">原始手部状态</a> · <a href="'+base+'canonical_raw_camera_mapping.json">原始路径与公开文件映射</a></p>'
    template=(H/'dashboard_template.html').read_text();content=template.replace('__ASSET_BASE__',base).replace('__SUMMARY_JSON__',json.dumps(summary,ensure_ascii=False,indent=2).replace('&','\\u0026').replace('<','\\u003c')).replace('__DEFAULT_GALLERY_JSON__',gallery)
    science='<section class="panel" id="scientific-figures"><h2>完整实验比较与实际覆盖</h2><p class="subtle">图表来自闭合的512窗口；下载PDF可查看完整统计。</p><div class="gallery-grid">'
    for name,title in [('outcomes','违规与原生接触'),('actual_motion','实际运动量'),('initial_coverage','随机初态覆盖')]:
        science+=f'<figure><a href="{base}figures/{name}.pdf"><img src="{base}figures/{name}.png" loading="lazy" style="width:100%;height:auto;object-fit:contain" alt="{title}"></a><figcaption>{title} · PDF</figcaption></figure>'
    science+='</div><p>当前联合LP不可行2071个环境步，扩宽1940个环境步；重新投影后仍有17376个当前线性残差超限环境步。未来动态验收为UNKNOWN。</p><p>'
    for name,title in [('ACCEPTANCE_RESULT.json','完整验收'),('ASTRA_ACCEPTANCE_REAL.json','Astra独立复核'),('COVERAGE_RESULT.json','实际覆盖统计'),('MECHANISM_RESULT.json','机制诊断'),('ADVERSE_CASE_DIAGNOSIS.md','8个新增失败时间线'),('ADVERSE_CASE_DIAGNOSIS.json','失败案例原始数据索引')]:
        science+=f'<a href="{base}{name}">{title}</a> · '
    science+='</p></section>'
    assert '<footer' in content
    content=content.replace('<footer',science+'<footer',1)
    rows=''
    for m in r['aggregated']:
        rows+='<tr><td class="method-name">'+titles[m['mode']]+'<small>'+esc(m['mode'])+'</small></td>'
        for key in ['windows','strict_windows','deep_windows','native_hand_raw_over_windows','native_hand_raw_peak_N','mean_q_l2_path','four_arms_moving_fraction']:
            v=m[key];rows+='<td data-field="'+key+'">'+esc(v)+'</td>'
        rows+='</tr>'
    content,count=re.subn(r'(<tbody id="method-rows">).*?(</tbody>)',lambda m:m[1]+rows+m[2],content,flags=re.S);assert count==1
    values={'window-count':'512 / 512','case-count':'128','step-count':'960','adoption-status':'拒绝采用' if r['candidate_decision']=='REJECTED' else '有限模拟验收通过','adoption-detail':scope,'latest-result':f"完整512窗口已闭合，候选结论：{r['candidate_decision']}。父审计与Astra独立原始数据验收一致。",'scope-content':scope,'coverage-content':coverage,'rejection-empty':'候选拒绝原因如下；实际科学结果全部保留。' if r['rejection_reasons'] else '候选在本轮有限模拟内通过所列门槛。'}
    for id,value in values.items():
        pattern=r'(<[^>]+id="'+re.escape(id)+r'"[^>]*>).*?(</[^>]+>)';content,n=re.subn(pattern,lambda m:m[1]+esc(value)+m[2],content,count=1,flags=re.S);assert n==1,id
    content,n=re.subn(r'(<ul id="rejection-list"[^>]*>).*?(</ul>)',lambda m:m[1]+''.join('<li>'+esc(x)+'</li>' for x in r['rejection_reasons'])+m[2],content,flags=re.S);assert n==1
    target=W/'docs'/H.name;target.mkdir(exist_ok=False);(target/'index.html').write_text(content)
    report=f"# 完整原生状态配对实验与验收\n\n完成512/512方法窗口，128个不同配对初态，四方法×两独立初态批次。每窗口960控制步、1920物理子步；候选结论 **{r['candidate_decision']}**，父分析与GPT‑6 Astra xhigh独立验收一致。\n\n"
    report+='| 方法 | strict/128 | deep/128 | 原生法向超0.1N/128 | 法向峰值N | 实际平均q路径 | 四臂均运动比例 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for m in r['aggregated']:report+=f"| {m['mode']} | {m['strict_windows']} | {m['deep_windows']} | {m['native_hand_raw_over_windows']} | {m['native_hand_raw_peak_N']:.9g} | {m['mean_q_l2_path']:.9g} | {m['four_arms_moving_fraction']:.9g} |\n"
    report+='\n预登记strict为原生float32间隙<0m，deep为<-0.005m，无评分epsilon。另要求候选实际初态全9021行无负值、无新增配对失败、整体与七分层路径至少保留零包含基线的90%、四臂运动比例至少90%、26关节各平均范围≥0.001rad，以及完整原生初态严格配对。手部全伙伴聚合法向0.1N为独立资格门槛，同手自碰全部保留。\n\n'
    report+='候选拒绝原因：\n\n'+(''.join('- '+x+'\n' for x in r['rejection_reasons']) if r['rejection_reasons'] else '- 本轮有限模拟内未触发拒绝原因。\n')
    report+=f"\n实际运动路径比 {r['actual_motion_path_ratio']}；四臂运动比例比 {r['actual_four_arm_movement_ratio']}。原球模型判定 {r['represented_sphere_decision']}；实际初始化几何判定 {r['initial_state_decision']}。\n\n"
    for comp in r['comparisons']:report+=f"- {comp['first']} → {comp['second']}：{json.dumps(comp['counts'],ensure_ascii=False)}。\n"
    report+=f"\n{coverage}\n\n全部64环境逐控制步保存三个原生边界；逐物理子步记录手部全伙伴法向/净力/计数、全部关节与实际位置目标。所有视图绑定同次运行，手部逐图核验全部64环境的关节、根状态、连杆、执行器目标和物理时钟逐字节不变。{build['gallery_groups']}组×21视角，共{build['original_PNG']}张原始1280×720图；未裁切、缩放或改图。0.06m连杆中心包围球在实际USD六个视锥平面均有≥0.05m余量；这不证明所有网格或无遮挡。\n\n"
    report+='公共U拇指初态/目标0.35rad在四方法一致，受控臂、演员、增益、自碰、原球豁免及FIFO6不变。初态资格采样仍在旧手部默认条件下完成，未据策略结局筛选或重采；构造阶段接触未观测。开发64窗口原生接触超限64/64→0/64只是开发证据；实际约3.6mrad追踪误差保留，正式结论只按新随机完整窗口计算。\n\n'
    report+='联合可行性机制只在当前联合LP返回不可行status2时扩宽到原可达界，并重新调用原30遍投影；LP见证不替代控制，未来队列始终UNKNOWN。当前残差与LP分类属于线性诊断，不能代替实际几何、接触或未来非线性安全。\n\n'
    report+=f'两批当前LP不可行2071个环境步、扩宽1940个环境步，最终仍有17376个当前线性残差超限环境步。[8个新增失败逐案例诊断]({base}ADVERSE_CASE_DIAGNOSIS.md)；时间上的先后与相关性不等于已证明因果。\n\n'
    report+='失败历史全部保留：短磁带协议错误（物理子进程0、监督1）、无子进程资源拒绝、旧25GiB等待器明确关闭actual−15/outer241。正式资源门槛预登记20GiB空闲、12GiB自身上限、6GiB运行空闲底线；实际退出与协议闭合分别检查，Kit退出0不代表成功。\n\n'
    report+=scope+'\n\n公开资源保存全部原图/状态、512案例、完整960/1920曲线、原始路径/SHA映射、验收登记与源码。全9021距离/Jacobian/预测及完整原生流保存在NAS原始实验与归档；公开曲线不替代未公开的大型原始文件。\n\n'
    report+=f"[固定资源清单]({base}asset_manifest.json) · [全部512案例]({base}case_table.json) · [实际随机覆盖]({base}COVERAGE_RESULT.json) · [独立验收]({base}ASTRA_ACCEPTANCE_REAL.json) · [完整原始文件SHA清单]({base}raw_experiment_manifest.json)\n\n"
    report+='固定源与接收规范：\n\n'+f"- criteria SHA `{r['criteria_sha256']}`\n- registration SHA `{r['acceptance_registration_sha256']}`\n- camera SHA `{r['camera_registration_sha256']}`\n- initial gate SHA `{r['initial_state_registration_sha256']}`\n"
    with (H/'REPORT.md').open('x') as f:f.write(report)
    (target/'REPORT.md').write_text(report);(target/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    root=W/'index.html';before=root.read_bytes();assert b'id="nativeAcceptanceEvidence"' not in before
    counts='；'.join(f"{titles[m['mode']]} {m['strict_windows']}/128" for m in r['aggregated'])
    card='<!-- native acceptance start --><section class="sci-note" id="nativeAcceptanceEvidence"><h2>最新完整验收：原生状态、手部接触与21视角</h2><p>128个新配对初态×四方法，512窗口全部闭合。strict：'+counts+'。候选：'+esc(r['candidate_decision'])+'。</p><p>原生接触与运动量分别验收；全部'+str(r['total_PNG'])+'张原图、512案例、960/1920步曲线、实际随机覆盖和GPT‑6 Astra xhigh独立验收可查。未来队列UNKNOWN，整机与硬件安全未认证。</p><p><a href="docs/'+H.name+'/">查看完整结果、拒绝原因与同状态原图 →</a></p></section><!-- native acceptance end -->\n'
    needle=b'<section class="sci-note"';at=before.index(needle);after=before[:at]+card.encode()+before[at:];root.write_bytes(after);assert after.replace(card.encode(),b'',1)==before
    files=[p for p in (W/'docs').rglob('*') if p.is_file()];size=sum(p.stat().st_size for p in files);assert size<1024**3,'registered main docs budget exceeded'
    record=dict(status='COMPLETE_BOUNDED_MAIN_PANEL_BUILD',asset_commit=commit,asset_base=base,original_root_sha256=hashlib.sha256(before).hexdigest(),new_root_sha256=sha(root),new_card_sha256=hashlib.sha256(card.encode()).hexdigest(),original_root_bytes_preserved_after_card_removal=True,main_docs_files=len(files),main_docs_bytes=size,remaining_to_1GiB=1024**3-size,new_panel_bytes=sum(p.stat().st_size for p in target.iterdir()),static_4method_numeric_noJS=True,default_gallery=chosen['job_id'],default_env=chosen['env_id'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/'PANEL_MAIN_BUILD.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(record,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset-commit',required=True);args=p.parse_args();main(args.asset_commit)
