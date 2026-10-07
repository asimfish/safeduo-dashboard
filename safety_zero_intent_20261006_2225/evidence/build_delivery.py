"""Compact Pages with the original scientific bundle on one fixed evidence commit."""
from pathlib import Path
import gzip,hashlib,json,shutil,sys
from candidate_decision import decide
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
WORK=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_worktree_20261006')
ASSET_WORK=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_assets_20261006')
ASSETS=ASSET_WORK/HERE.name
PUBLIC=WORK/'docs'/HERE.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def assets():
    result=load(HERE/'holdout_results.json');visual=load(HERE/'visual_verification.json')
    assert result['completed_method_windows']==576 and result['invalid_method_windows']==0
    for n in ['ASTRA_ZERO_FINAL_REVIEW.json','raw_evidence_manifest.json','ALL_ANALYSIS_EXECUTION.json','ZERO_INTENT_AUDIT.json']:
        assert load(HERE/n)['status'].startswith('PASS'),n
    assert visual['status']=='PASS_BOUNDED_ACTUAL_SIX_PLANE_CAMERA'
    ASSETS.mkdir(exist_ok=False);files={}
    def copy(src,rel):
        if rel in files:return
        dest=ASSETS/rel;dest.parent.mkdir(parents=True,exist_ok=True);s=sha(src)
        shutil.copy2(src,dest);assert sha(src)==sha(dest)==s
        files[rel]=dict(path=rel,bytes=dest.stat().st_size,sha256=s)
    for row in result['rows']:
        root=Path(row['path'])
        for r in load(root/'first_failure_receipts.json')['receipts']:
            src=root/r['path'];assert sha(src)==r['sha256'];copy(src,str(src.relative_to(RAW)))
        copy(root/'input_recipe.npz',str((root/'input_recipe.npz').relative_to(RAW)))
    for g in visual['galleries']:
        root=RAW/'visual'/g['mode']
        for rel,s in [(g['state'],g['state_sha256']),(g['before'],g['before_sha256']),(g['after'],g['after_sha256']),*[(im['path'],im['sha256']) for im in g['images']]]:
            src=root/rel;assert sha(src)==s;copy(src,'visual/'+g['mode']+'/'+rel)
    for row in load(HERE/'RANDOM_EXPERIMENT_DESIGN.json')['rows']:
        for n in ['bank.npz','metadata.json']:
            rel=f'banks/{row["initial_seed"]}/{n}';copy(RAW/rel,rel)
    for mode in ['joint_reference','zero_inclusive']:
        rel='visual/'+mode+'/input_recipe.npz';copy(RAW/rel,rel)
    # All closed original reviewer sources, failed attempts, audit results and
    # reproduction records remain downloadable without duplicating megabytes in Pages.
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.json','.md','.html','.log','.png','.pdf','.svg']:
            copy(p,'evidence/'+str(p.relative_to(HERE)))
    original=(HERE/'holdout_results.json').read_bytes();dest=ASSETS/'data/holdout_results.json.gz';dest.parent.mkdir()
    dest.write_bytes(gzip.compress(original,compresslevel=6,mtime=0));assert gzip.decompress(dest.read_bytes())==original
    rel=str(dest.relative_to(ASSETS));files[rel]=dict(path=rel,sha256=sha(dest),bytes=dest.stat().st_size,uncompressed_sha256=hashlib.sha256(original).hexdigest())
    assert all(r['bytes']<90_000_000 for r in files.values())
    receipt=dict(status='PASS_ORIGINAL_EVIDENCE_COPY',files=list(files.values()),count=len(files),bytes=sum(r['bytes'] for r in files.values()),pixel_transformation=False)
    write(ASSETS/'asset_manifest.json',receipt);write(HERE/'ASSET_BUILD.json',receipt)
    (ASSET_WORK/'README.md').write_text('SafeDuo zero-intent simulation evidence. Original images bind own actual states. Full methods, failures, independent sources and closed evidence are at https://asimfish.github.io/safeduo-dashboard/docs/'+HERE.name+'/ . No hardware approval.\n')
    print('ASSETS_COMPLETE',receipt['count'],receipt['bytes'],flush=True)
def pages():
    publication=load(HERE/'ASSET_PUBLICATION.json');assert publication['status']=='PASS_TWO_PUBLIC_BYTE_READS'
    result=load(HERE/'holdout_results.json');visual=load(HERE/'visual_verification.json');failure=load(HERE/'first_failure_audit.json')
    review=load(HERE/'ASTRA_ZERO_FINAL_REVIEW.json');camera_review=load(HERE/'ASTRA_ZERO_FINAL_CAMERA_REVIEW.json')
    assert review['status'].startswith('PASS') and camera_review['status'].startswith('PASS')
    base='https://raw.githubusercontent.com/asimfish/safeduo-dashboard/'+publication['commit']+'/'+HERE.name+'/'
    for row in failure['records']:row['snapshot_relative']=str(Path(row['snapshot_path']).relative_to(RAW))
    totals={r['mode']:r for r in result['totals']};a,b=totals['joint_reference'],totals['zero_inclusive'];decision=decide(result)
    independent=load(HERE/'ASTRA_ZERO_FINAL_SCORE.json');class_summary=[]
    for mode in result['totals']:
        c=independent['totals'][mode['mode']]
        assert mode['class_violations']==[c['strict']['class_failed_windows'][k] for k in ['cross','self_F','self_U','table']]
        for k in ['cross','self_F','self_U','table']:
            class_summary.append(dict(mode=mode['mode'],kind=k,strict_windows=c['strict']['class_failed_windows'][k],deep_windows=c['deep']['class_failed_windows'][k],strict_env_steps=c['strict']['class_env_steps'][k],deep_env_steps=c['deep']['class_env_steps'][k]))
    verdict=decision['text']+f"。原参考严格失败 {a['violations']}/192，保持目标候选 {b['violations']}/192；严格违规环境帧 {a['strict_env_steps']} → {b['strict_env_steps']}，深度违规环境帧 {a['deep_env_steps']} → {b['deep_env_steps']}。物理安全尚未验证。"
    zero=load(HERE/'ZERO_INTENT_AUDIT.json');candidate=next(r for r in zero['summary'] if r['mode']=='zero_inclusive')
    assert candidate['bounds_excluding_zero_env_steps']==0 and candidate['reference_prelimit_injection_env_steps']==0
    invariant=dict(candidate_env_steps=192*960,prefix_reference_injection_env_steps=candidate['reference_prelimit_injection_env_steps'],prefix_projector_nonzero_env_steps=candidate['original_projector_nonzero_return_env_steps'],scope='allframe bounds checked by parentdenseaudit+registeredrawzero audit; projector maynonzero')
    manifest=load(ASSETS/'asset_manifest.json')
    docs=[dict(title=row['path'][len('evidence/'):],path=base+row['path']) for row in manifest['files'] if row['path'].startswith('evidence/') and not row['path'].endswith(('.png','.pdf','.svg'))]
    docs.append(dict(title='asset_manifest.json',path=base+'asset_manifest.json'))
    PUBLIC.mkdir(parents=True,exist_ok=False)
    for name in ['REPORT.md','REPRODUCE.md','NEXT.md','RANDOM_EXPERIMENT_DESIGN.json','CANDIDATE_DECISION.json','ASTRA_ZERO_FINAL_REVIEW.json']:
        shutil.copy2(HERE/name,PUBLIC/name);assert sha(HERE/name)==sha(PUBLIC/name)
    shutil.copy2(HERE/'comparison.png',PUBLIC/'comparison.png')
    design=load(HERE/'RANDOM_EXPERIMENT_DESIGN.json')
    payload=dict(modes=design['modes'],blocks={r['command_seed']:r['block'] for r in design['rows']},assetBase=base,classSummary=class_summary,invariant=invariant,zeroSummary=zero['summary'],fixedSummary=load(HERE/'FIXED_FEASIBILITY_AUDIT.json')['summary'],bank=load(HERE/'bank_audit.json'),commands=load(HERE/'command_audit.json'),visual=visual,failure=failure,docs=docs,verdict=verdict,
        executionText='流程偏差：初始GPU内存门禁失败后父级误启动三组基线。早期基线按确切PID暂停，再在实际容量满足恢复规则后沿同一进程继续；没有重启、重置状态/FIFO或丢弃片段。基线独有的墙钟暂停与初始门禁违反均保留，限制干净预注册执行和效应因果归因。 原首尝448完成/128因USD初始化失败未开始；保留原Kit退出0但协议未完成及上层误记PASS，恢复只补未开始的两组候选，不重跑完整轨迹。曾事后登记改用CUDA1，但实际图形后端不支持且没有保存产物；确切自有相机被停止，原失败与未记录的wait码保留。最终按原CUDA0/九视角/样本/输入/state计划重新采集；远程传递USD资产跨尝试逐字节等价未证。',
        cameraText=f"完成 {visual['groups']} 组、{visual['images']} 张原始图像。128个单独相机窗口复用首银行/输入，不增加192初态/576数值分母。精确数值重放："+'；'.join(r['mode']+' '+r['binding_status']+'，最大q差'+str(r['q_max_difference'])+' rad' for r in visual['runs'])+'。所有图像绑定自己的相机状态。',
        reviewText=f"专门GPT‑6 Astra xhigh独立检查机制并复算全部576窗口严格/深度计分、持续时间和逐案例配对。全部{visual['groups']}组native状态及{visual['images']}张PNG哈希/尺寸检查，实际查看{camera_review['actual_original_pngs_viewed']}张原图。像素审查限于这一子集；具体边界及原始执行记录可下载。")
    (PUBLIC/'payload.json.gz').write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),compresslevel=6,mtime=0))
    shutil.copy2(HERE/'panel_template.html',PUBLIC/'index.html')
    added=sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file());total=sum(p.stat().st_size for p in (WORK/'docs').rglob('*') if p.is_file())
    assert added<=load(HERE/'CAPACITY_REGISTRATION.json')['new_initial_Pages_budget_bytes']
    assert total<1_073_741_824,('Pagesinternal1GiBceiling exceeded',total)
    write(HERE/'PANEL_BUILD.json',dict(status='PASS_COMPLETE_COMPACT_PANEL',cases=len(result['cases']),windows=576,images=visual['images'],new_bytes=added,total_docs_bytes=total,decimal_1GB_satisfied=total<1_000_000_000,one_GiB_satisfied=True,source_evidence_pinned_on_asset_branch=True,asset_commit=publication['commit']))
    print('PAGES_COMPLETE',added,total,flush=True)
if __name__=='__main__':{'assets':assets,'pages':pages}[sys.argv[1]]()
