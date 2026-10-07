from pathlib import Path
import json,gzip,numpy as np,hashlib
p=Path(__file__).resolve().parent;root=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
data={};gallery=[]
for run,regname,resultname in [('development','REGISTRATION_DEVELOPMENT.json','DEVELOPMENT_RESULTS.json'),('validation','REGISTRATION_VALIDATION.json','VALIDATION_FULL_RESULTS.json'),('paired','REGISTRATION_PAIRED.json','PAIRED_FULL_RESULTS.json')]:
 r=json.loads((p/regname).read_text());res=json.loads((p/resultname).read_text());raw=root/run;receipt=json.loads((raw/'recording_receipt.json').read_text());meta=json.loads((raw/'native_metadata.json').read_text());identity=json.loads((raw/'contact_identities.json').read_text());curves={}
 for ch in receipt['chunks']:
  with np.load(raw/ch['file']) as z:
   cyc=int(z['cycle'][0]);assert (z['cycle']==cyc).all();steps=z['step'];clock=((steps%720)+1)*r['physics_dt_s']
   for e,case in enumerate(r['cases']):
    for arm in ['U_L','U_R']:
     idx=cyc*r['profiles_per_cycle']+case['profile_slot'];key=(idx,case['reference_time_s'],arm,case.get('method'));c=curves.setdefault(key,dict(profile_index=idx,reference_time_s=case['reference_time_s'],arm=arm,method=case.get('method'),time_s=[],force=[],q1=[],q2=[],target1=[],target2=[]))
     paths=next(x for x in identity['hands'] if x['env']==e and x['arm']==arm);mask=np.asarray([[x in paths['sensors'] for x in fs] for fs in paths['filters']]);
     if run=='development':normal_value=np.linalg.norm(z[f'e{e}:{arm}:hand_partner_normal'],axis=-1)
     else:
      cnt=z[f'e{e}:{arm}:hand_normal_count'];st=z[f'e{e}:{arm}:hand_point_start'];force_raw=z[f'e{e}:{arm}:hand_point_force'];safe_st=np.where(cnt>0,st,0).astype(int);end_idx=safe_st+cnt.astype(int);prefix=np.pad(abs(force_raw).astype(np.float64).cumsum(1),((0,0),(1,0)));ix=np.arange(len(force_raw))[:,None,None];normal_value=prefix[ix,end_idx]-prefix[ix,safe_st]
     force=np.where(mask[None],normal_value,0).max(axis=(1,2))
     v=meta['arms'][arm];names=v['joint_names'];ji=[names.index(('left' if arm=='U_L' else 'right')+f'_thumb_{a}_joint') for a in [1,2]];q=z[arm+':q'][:,e,ji];tgt=z[arm+':target'][:,e,ji]
     for j in range(0,len(clock),12):
      end=min(j+11,len(clock)-1);c['time_s'].append(float(clock[end]));c['force'].append(float(force[j:end+1].max()));c['q1'].append(float(q[end,0]));c['q2'].append(float(q[end,1]));c['target1'].append(float(tgt[end,0]));c['target2'].append(float(tgt[end,1]))
 frames={}
 for image in receipt['images']:
  step=image['step'];e=image['env'];idx=(step//720)*r['profiles_per_cycle']+r['cases'][e]['profile_slot'];key=(step,e)
  frame=frames.setdefault(key,dict(run=run,env=e,method=r['cases'][e].get('method'),profile_index=idx,time_s=image['state_time_s'],step=step,images=[]));frame['images'].append({**image,'file':run+'/'+image['file']})
 gallery.extend(frames.values())
 display_reg={k:r[k] for k in ['profiles','cases','os_random_seeds','steps','physics_dt_s','scope']};display_reg['os_random_seeds']={k:str(v) for k,v in r['os_random_seeds'].items()}
 data[run]=dict(registration=display_reg,summary={k:v for k,v in res.items() if k!='rows'},rows=res['rows'],curves=list(curves.values()),normal_metric='vector_resultant' if run=='development' else 'sum_point_normal_magnitudes',results_sha256=sha(p/resultname),registration_sha256=sha(p/regname))
admission=json.loads((p/'ADMISSION_AUDIT.json').read_text());videos=json.loads((p/'VIDEO_RECEIPT.json').read_text())['videos'];v=data['validation']['summary'];status=v['u_paths_qualified']==v['u_hand_paths'] and admission['admitted_requests']==admission['qualified_requests'] and admission['abort_count']==0 and data['paired']['summary']['status']=='PASS_MATCHED_HAND_ADMISSION_COUNTERFACTUAL'
verdict=('本轮精确空手路径的手部候选验收通过；完整机器人与真实任务安全尚未验收。' if status else '新参考验证仍有失败或中止，候选完整采用被拒绝；没有完整机器人安全结论。')
links=[('科研验收矩阵与尚未完成的安全门槛','ACCEPTANCE_MATRIX.json'),('计数字段说明：120档/118独特组合','COUNT_FIELD_NOTE.json'),('相同初态直接执行/拒绝对照，含逐点法向核验','PAIRED_FULL_RESULTS.json'),('操作范围与未覆盖自由度','COVERAGE_REPORT.json'),('开发完整结果','DEVELOPMENT_RESULTS.json'),('新参考验证完整结果（含逐点法向总量）','VALIDATION_FULL_RESULTS.json'),('新参考原始冻结的向量评分','VALIDATION_RESULTS.json'),('标量法向审计器','audit_scalar_contacts.py'),('逐点测量补强记录','SCALAR_NORMAL_CONTRACT.md'),('原生准入审核','ADMISSION_AUDIT.json'),('开发注册与源/资产SHA','REGISTRATION_DEVELOPMENT.json'),('验证注册与未知随机银行','REGISTRATION_VALIDATION.json'),('护照：精确目标与路径范围','PASSPORT.json'),('开发逐步状态与图像索引','development/recording_receipt.json'),('验证逐步状态与图像索引','validation/recording_receipt.json'),('全部原始准入申请','validation/admission_records.json'),('原生生成器','native_probe.py'),('原生验证与近景记录器','native_validation.py'),('独立CPU评分器','analyze.py'),('手部准入候选源代码','hand_admission.py'),('复现步骤与限制','REPRODUCE.md'),('15项软件合同测试与边界修复','ADMISSION_SOFTWARE_TEST.log'),('全部媒体/数据SHA清单','MEDIA_MANIFEST.json')]
payload=dict(**data,coverage=json.loads((p/'COVERAGE_REPORT.json').read_text()),count_note=json.loads((p/'COUNT_FIELD_NOTE.json').read_text()),admission=admission,gallery=gallery,videos=videos,verdict=verdict,links=[dict(label=l,file=f) for l,f in links],video_note='前四路：validation/env0/profile0。后两路：paired/env0原旧目标直接执行、env1准入器拒绝同一目标，参考均为2s。每路原生状态0.008333–5.908097s，60帧10fps。拒绝闭合没有完成握持；没有System0臂安全对照。',curve_window_note='每12步法向最大值放在该窗口结束时间；关节角与目标为窗口最后一个状态，所有数值保留原生精度。')
(p/'DISPLAY_PAYLOAD.json.gz').write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))
summary=dict(status='PASS_EXACT_EMPTY_HAND_CANDIDATE' if status else 'REJECT_CANDIDATE',verdict=verdict,development=data['development']['summary'],validation=data['validation']['summary'],paired=data['paired']['summary'],admission=admission,new_task_trials=0,new_system0_trials=0,constructor_contacts_qualified=False,comprehensive_physical_safety_qualified=False,original_images=len([x for g in gallery for x in g['images']]),native_env_states=data['development']['summary']['env_states']+v['env_states']+data['paired']['summary']['env_states'],payload_sha256=sha(p/'DISPLAY_PAYLOAD.json.gz'))
(p/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['development','validation']},indent=2))
