"""Generate the actual closed report before final independent report review."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from candidate_decision import decide

HERE=Path(__file__).resolve().parent
def load(n):return json.loads((HERE/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,text):
    with (HERE/n).open('x') as f:f.write(text)
def main():
    for n in ['ALL_ANALYSIS_EXECUTION.json','ASTRA_ZERO_FINAL_SCORE.json','ZERO_INTENT_AUDIT.json']:
        assert load(n)['status'].startswith('PASS'),n
    r=load('holdout_results.json');v=load('visual_verification.json');b=load('bank_audit.json');c=load('command_audit.json')
    z=load('ZERO_INTENT_AUDIT.json');decision=decide(r)
    assert r['completed_method_windows']==576 and r['invalid_method_windows']==0
    t={x['mode']:x for x in r['totals']}
    lines=['六步延迟下的零输入目标保持实验','',
        decision['text']+'。该判定依据提前登记的规则；软件不变式通过不等于物理安全。','',
        '登记192个新资格初态×三条件=576个方法窗口，各960帧；恢复后的最终有效窗口576、未完成窗口0。首尝试实际448完成、128因远程USD资产初始化失败而无效，未进入样本执行。原失败记录完整保留，只重试未开始的两组候选；不丢弃或重跑已经完成的448条轨迹。所有严格/深度计分包含最初60帧零输入。','',
        '| 条件 | 严格失败/192 | 深度失败/192 | 严格环境帧 | 深度环境帧 | 最小距离mm | 关节L2路径均值rad | 四臂同时运动比例 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in r['totals']:
        lines.append(f"| {x['mode']} | {x['violations']} | {x['deep']} | {x['strict_env_steps']} | {x['deep_env_steps']} | {x['min_nonexempt_mm']:.6f} | {x['joint_path_l2_mean_rad']:.9f} | {x['four_arms_moving_fraction']:.9f} |")
    lines+=['','严格：原始全部非豁免表示距离float32 <0；深度：<−float32(0.005)m，无计分epsilon。环境帧相关，不能作为IID可靠性样本。','',
        '| 块 | 对照→候选 | 救回 | 新失败 | 均失败 | 均无失败 |','|---|---|---:|---:|---:|---:|']
    for p in r['paired']:lines.append(f"| {p['seed']} | {p['a']}→{p['b']} | {p['rescued']} | {p['new_failures']} | {p['both']} | {p['neither']} |")
    lines+=['',
        '机制：候选只扩展原0.010rad参考区间，使保持当前已发目标的零增量处于原速度盒/软限位内。若原可达区间本身排除零，候选报错而不是扩大原限位。原参考0.050rad与原收紧0.010rad是对照。原演员32行观测、有限30次原投影、R19、速度盒、阈值/豁免、全9021表示几何及六个不可撤销待执行目标均保留。','',
        '零输入保持仅针对参考预限幅。原安全投影仍可返回非零指令，实际目标与测量位置也可能继续变化。保留原投影返回值；不强制执行零，也不清空FIFO。完整雅可比每帧有限检查属于冻结生产程序；仅固定/失败时刻保存选中J，不声称独立复算每帧全J。','',
        '| 条件 | 零前缀失败/192 | 严格环境帧 | 零被边界排除 | 参考预限幅注入非零 | 原投影返回非零 | 目标增量非零 | 相对初始测量q的最大已发目标位移rad |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in z['summary']:
        lines.append(f"| {x['mode']} | {x['zero_prefix_strict_windows']} | {x['zero_prefix_strict_env_steps']} | {x['bounds_excluding_zero_env_steps']} | {x['reference_prelimit_injection_env_steps']} | {x['original_projector_nonzero_return_env_steps']} | {x['nonzero_effective_target_env_steps']} | {x['maximum_target_drift_rad']:.9f} |")
    lines+=['','该零输入审计及不变式在新结果前注册。同一保存前态的CPU区间回放不是另一条物理轨迹，也不证明唯一原因；第t帧新目标不能越过前六个待执行目标直接影响同帧失败。','',
        f"随机性：六个OS熵种子、三块各64初态。192初态与枚举的{b['prior_compared']}个历史初态无精确重复，最小关节L2距离{b['minimum_l2_distance_rad']:.12f}rad；实际192输入与{c['prior_unique_tapes']}个枚举旧输入无重复。每块六臂对各8个、全局16个；资格仅依赖原几何及原始零输入60步稳定性。所有候选与拒绝存档。条件采样不是IID可靠性或完整26维覆盖证明。",'',
        '实际运动、各关节初始/访问覆盖及六臂对风险暴露见完整结果。运动减少或暴露不足不被称为安全改善。','',
        '保存所有首次失败前态与原选中安全/alpha/界矩阵，及0/60/62/66/71/75/180/480/959九点全部5184状态/10368机器人集合的HiGHS诊断。归一化binary64求解容差1e-9不是物理计分epsilon；固定采样不证明每帧非线性可行性。','',
        f"真实相机：{v['groups']}组、{v['images']}张1280×720原始PNG，每组九视角，各自state/目标/FIFO与全64native q/qd/root/rootvelocity渲染前后绑定。固定组覆盖六臂对与全局槽75/480/959，另含各分层最早实际相机负几何组。128个独立相机执行复用首块银行和输入，不增加192初态/576数值分母。"]
    for x in v['runs']:
        lines.append(f"相机{x['mode']}对数值轨迹精确重放：{x['binding_status']}，最大q差{x['q_max_difference']}rad。图像绑定自己的相机执行状态，不冒充数值轨迹截图。")
    lines+=['','实际USD相机变换、内参、六个视锥平面和目标组表示球检查不等于全mesh轮廓、无遮挡或硬件安全。独立查看原图子集的数量和边界见最终Astra相机回执；未查看图片不作像素审查声称。','',
        '流程偏差必须保留：初始GPU空闲内存门禁实际失败，父级编排错误仍启动了三组基线。早期三组基线与其包装进程被精确SIGSTOP暂停；无完整960窗口、收紧/保持目标候选尚未启动。同一进程在实际容量满足恢复规则后SIGCONT继续，没有重启、重置状态/FIFO或丢弃原片段。基线独有的墙钟暂停和初始内存门禁违反属于实验流程偏差，限制“干净预注册执行”及效应因果归因；原失败不改为PASS。固定仿真dt、源文件、银行和指令未改。全部实测窗口、源与FIFO仍须逐项验收。','',
        '父级8项CPU不变式测试；专门GPT-6 Astra xhigh独立20项CPU契约测试及5种错误变体检出、32项原始计分合成测试在新结果前通过，独立计分源码在新结果前冻结。最终源码/实际报告/完整原始计分/原图子集的独立复核与发布验收分别留存。所有结果仅限仿真研究，未批准真机或生产推广。','']
    assert load('RECOVERY_EXECUTION.json')['status']=='PASS_COMBINED_RECOVERY_WITH_RETAINED_FAILURES'
    lines+=['补跑流程修订：原上层driver只看退出码，将两组场景初始化失败但Kit退出0误记为PASS。首轮448/128和原失败日志/协议均保留，恢复只覆盖未开始的128窗口，原策略、样本、输入和计分源码未改。相机首次USD初始化失败没有图像；曾登记改用CUDA1，但实际usdrt不支持GPU1，停止前仍无任何保存产物。仅终止确切自有相机PID，SIGTERM超时后SIGKILL及该失败中未持久化的child.wait码UNRECORDED均保留，不猜测成0。最终以原CUDA0计划、原九视角、银行、输入、960帧与native/FIFO契约重新采集；新驱动先持久化真实退出码再验证产物。上述属于结果已见后的操作恢复，不声称干净一次预注册执行。远程USD两次HTTP字节回读证明恢复时可读，未对首尝与补跑所有传递USD依赖做逐字节等价证明。完整动作及几何仍由实际状态数据验收。','']
    write('REPORT.md','\n'.join(lines))
    write('REPRODUCE.md','从RANDOM_EXPERIMENT_DESIGN、NUMERIC_REGISTRATION与plans读取真实初态/输入种子、原演员和源哈希；先用原几何及原始零输入资格筛选，核对bank_audit，再使用不可变计划运行三个条件。独立scorer/math来源见ASTRA_ZERO_RAW_SOURCE_REGISTRATION。必须保留初始内存门禁失败和暂停/恢复链，不能声称无流程偏差。所有原始逐帧、选中J/首失败/相机native数据保存在RAW双次完整回读档案；公开附属证据在asset_manifest所绑定固定提交，PNG无像素转换。复现实验应另建namespace、新资格银行；封存目录只读。\n')
    write('NEXT.md','下一轮先依据完整配对结果区分参考注入修复、投影残余和实际六步跟踪债务。若候选仍有失败，围绕实际先行目标/速度及提前可行性和非线性不确定性开展开发反事实，再用全新预注册银行验证。先检查任何新增失败与运动/暴露差异；不在本192初态上调参后宣称新盲测。运行编排必须把资源检查与启动置于同一有条件控制路径，保持其他任务不受影响。继续实际足够九视角和自身state绑定、专门Astra独立计分，保留所有不利结果。\n')
    write('CANDIDATE_DECISION.json',json.dumps(dict(status='DESCRIPTIVE_REGISTERED_DECISION_APPLIED',**decision,physical_safety_certified=False,hardware_approved=False,production_promoted=False),indent=2,ensure_ascii=False)+'\n')
    write('REPORT_BUILD.json',json.dumps(dict(status='READY_FOR_FINAL_INDEPENDENT_REPORT_REVIEW',report_sha256=sha(HERE/'REPORT.md'),decision=decision,utc=datetime.now(timezone.utc).isoformat()),indent=2,ensure_ascii=False)+'\n')
    print('REPORT_READY',decision['code'],flush=True)
if __name__=='__main__':main()
