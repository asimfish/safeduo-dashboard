from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=json.loads((p/'SUMMARY.json').read_text());v=s['validation'];c=s['paired'];a=s['admission']
items=[
 dict(id='A0',criterion='每物理步原生状态、完整性、有限性、渲染前后状态一致与原图SHA',status='PASS_RECORDING',evidence=['DEVELOPMENT_RESULTS.json','VALIDATION_RESULTS.json','PAIRED_RESULTS.json']),
 dict(id='A1',criterion='真实手部完整伙伴身份、自碰开启、直接资产与源封存',status='PASS_MEASURED_SCOPE',evidence=['development/native_metadata.json','validation/contact_identities.json','SOURCE_SNAPSHOT.json']),
 dict(id='A2',criterion='全范围手部构型开发，整段路径/追踪/回开验收，失败不可删除',status='PASS_EXECUTION_WITH_UNSAFE_PROFILES_REJECTED',profile_slots=120,unique_specs=118,all_six_paths_qualified=s['development']['profile_count_qualified'],evidence=['COVERAGE_REPORT.json','DEVELOPMENT_RESULTS.json']),
 dict(id='A3',criterion='冻结精确目标准入、三个新参考的原生验证',status='PASS' if v['u_paths_qualified']==v['u_hand_paths'] and a['admitted_requests']==a['qualified_requests'] and not a['abort_count'] else 'REJECT_CANDIDATE',evidence=['PASSPORT.json','VALIDATION_SEAL.json','ADMISSION_AUDIT.json','VALIDATION_SCALAR_AUDIT.json']),
 dict(id='A4',criterion='同初态、同旧目标直接执行 vs 准入拒绝的真实法向',status='PASS' if c['status']=='PASS_MATCHED_HAND_ADMISSION_COUNTERFACTUAL' else 'REJECT_CANDIDATE',evidence=['PAIRED_CONTRACT.md','PAIRED_FULL_RESULTS.json','PAIRED_SCALAR_AUDIT.json']),
 dict(id='A5',criterion='状态陈旧/非有限/超限/自碰/速度中止语义',status='PASS_SOFTWARE_CONTRACT_ONLY_NATIVE_STOP_EFFECT_UNVERIFIED',evidence=['ADMISSION_SOFTWARE_RED.log','ADMISSION_SOFTWARE_TEST.log','hand_admission.py'],note='锁存后维持最后目标，不保证真实运动或夹压立即停止。配对拒绝避免接触，不是接触后恢复证明。'),
 dict(id='A6',criterion='首次构造内部动力学之前初始化安全与原生接触力',status='UNVERIFIED',evidence=['../safety_hand_initialization_20261007/SUMMARY.json']),
 dict(id='A7',criterion='真实双手共同物体抓持与搬运阶段持续抬升/承重/姿态',status='NOT_EXECUTED_NEW_TASK_TRIALS_0',note='本轮物体移远；旧晚期最大抬升不可替代搬运阶段持稳，旧U终点夹具承重约69%事实保留。'),
 dict(id='A8',criterion='有效放置支撑、完全回开与安全撤离',status='NOT_EXECUTED_PAYLOAD_RELEASE_TRIALS_0',note='空手回开合格不代表带载放置与释放合格。'),
 dict(id='A9',criterion='六臂对、26臂与48手自由度、感知误差与全操作空间随机任务',status='NOT_COVERED',evidence=['COVERAGE_REPORT.json'],note='只有3维手部控制；其余手关节耦合，6个相关静态臂参考，无世界空间体积覆盖估计。'),
 dict(id='A10',criterion='纯随机独立物理留出任务与统计安全性能',status='NOT_ESTABLISHED',note='20开发IID均匀目标、20分层开发目标；32未知目标只有请求拒绝，0未知目标物理执行。验证已合格目标在新参考重测，不是新IID任务。'),
 dict(id='A11',criterion='完整媒体与公开页面精确字节、桌面手机可视效果、视频真实播放',status='RECORDED_SEPARATELY_IN_DELIVERY_RECEIPTS',evidence=['MEDIA_MANIFEST.json','ARCHIVE_DOWNLOAD.json','PUBLIC_DELIVERY.json'])]
(p/'ACCEPTANCE_MATRIX.json').write_text(json.dumps(dict(status='COMPREHENSIVE_ROBOT_SAFETY_NOT_QUALIFIED',items=items),indent=2)+'\n');print('MATRIX_ITEMS',len(items))
