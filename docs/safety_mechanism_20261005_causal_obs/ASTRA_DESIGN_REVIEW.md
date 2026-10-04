# ASTRA：具体 queue-envelope 候选合同审查

2026-10-05，Asia/Shanghai。唯一写入为 safety_mechanism_20261005_causal_obs/ASTRA_DESIGN_REVIEW.md。本文已按主线最新指示转为审查 MECHANISM_CONTRACT.md、mechanism_runner.py、development_plan.json 和 causal_analysis.json；此前针对 backlog_aware=true 的候选建议不再是本轮审查对象。未运行仿真、修改生产/配置/冻结证据或接触其他父进程的 PLAN/进程。CPU验证仅提取两段纯helper，不导入runner/Isaac。

## 1. Verdict：Conditional pass（源码/实验合同）；物理安全未验收

具体 helper 与“扩充准入、保留原projector/backlog_aware=false/FIFO6”的合同主要一致。CPU集合/容量不变量通过；开发因果基线的八个原字段、39/64窗口计数及FIFO/时间join已独立核对。候选开发与holdout效果本审查尚未验收，不作策略、硬件或执行器侧架构批准。

Dimensions：功能 Pass仅限静态接线与helper合同；正确性 Risk（线性forecast、有限值与逐row作用时间仍需证据）；架构 Pass仅限严格FIFO不变；接口 Risk（新增row目前只留汇总，候选因果join不足）；可维护性 Risk（继承plan描述有残留）；性能 Unable to determine（全量Jacobian读取未作时间/硬件期限验证）。

Blocking findings：尚无候选完整结果可验，不能给物理安全或推广PASS。没有发现helper删除原行、修改projection、改变FIFO或开启backlog debit的静态证据。以下风险不是已经发生的新物理失败；如需宣称“具体row机制修复”，逐row提前纳入/真实目标作用证据是必要门。

Non-blocking improvements，仅三项：补明确的row8974/7310准入时间证据；单列有限值与1024容量失效；用额外receipt澄清冻结plan旧文字，不改inflight文件。Minimum required validation见§6；不要求另起参数矩阵或更换候选。

## 2. 基线与实际因果证据

最新stable-risk三个保护NPZ全量official_margins独立复算：60317411为39/64，80692357为41/64，109441003为42/64；合计122/192。class为cross8/self_F21/self_U50/table99，可重叠；table与self_U同现32窗口。各组前66个post-step无违例，实际actuator目标均满足FIFO6 exact。三组raw/System0 input_recipe的q_initial、初始flag和整条962帧外部tape均独立exact核对，960帧执行。条件化风险窗口不是IID样本。

实际配置已有COM、actor32、0.4s优先行选择、quota_cross8、soft-target投影、authority clamp和出口箱；backstop horizon为0.16/self0.4/table0.3，predict_backlog=true/pending6/retain_conditional_rows=true，但backlog_aware=false。原补充规则为d≤d_min+10mm，不是d<10mm。旧 [VALIDATION.md](/home/liyufeng/safeduo/artifacts/safety_perturbation_20261003/VALIDATION.md) 的FIFO失败、未推广及backlog_aware未执行勘误保留。

本次又直接读取 causal_analysis.path 指向的已完成开发诊断（只读主线自己的 development_baseline 原数据），与旧seed60317411保护cell比较八字段：
q_initial/cmd/q/ee/exec/official_margins/controller_target/actuator_target全部exact。
独立计数39/64；pre_q[0]=q_initial、pre_q[1:]=q[:-1]和FIFO6 exact；causal_metadata标明com。没有重复计算旧sealed v4或搬运192任务。

对env47/table row8974，直接核 selected_id/valid、投影/实发增量、Jqd、residual，并从压缩NPY按索引读取raw几何和exemption：

| trace index（zero-based post-step） | row纳入 | pre→post raw几何 m | 新issued target的J投影 m | 实际q运动的J投影 m | residual F/U |
|---|---|---|---:|---:|---|
| 66 | false | — | — | — | — |
| 67 | true | +0.06066138297319412→+0.056502945721149445 | +0.006552795879542828 | −0.004168274812400341 | 0 / 4.656612873077393e−10 |
| 72 | true | +0.006011143326759338→−0.01381663978099823 | +0.02208543010056019 | −0.01998724229633808 | 0 / 0 |

67帧实际几何闭合4.158437mm，与issued投影打开6.552796mm是不同变量，不应叫“求解器错误地预测同帧实际运动”。67 issued target按FIFO在73才首次作用，已经晚于72首报；当前运动受旧排队目标和动态状态影响。67/72 pre/post exemption均false。JSON时间线的body rate与COM Jqd接近，67为−0.0896315/−0.0895788m/s，72为−1.0580077/−1.0576859m/s；这一row不支持把首报仅归为COM参考错或豁免撤回。

env35/self_U row7310的分析首报为76，前六帧均已纳入，max_prior_residual约2.382209e−4；该例不能只用“前六帧漏行”解释。全诊断最大reported residual为0.029051568359m，也不能将env47近零residual推广到全部39窗口。这两项来自causal_analysis，本文未对row7310全量几何重建。

这些是候选的机制动机，尚不是“提前准入一定能修复”的反事实结果，更不是122个窗口的唯一根因。

## 3. 具体合同与源码的对应

| 合同要求 | 实际源码核对 | 边界 |
|---|---|---|
| 保留原selected与raw-critical | union_mask对有效selected索引scatter去重，OR d≤full_dmin+10mm及forecast条件。select按原row ID排序取所有mask行，不top-k丢行。 | “original priority”是已有约束选择，不是队列抢占；队列优先权不变。 |
| 真实pending/current/proposal | safety闭包读取actual delay.queue.pending的六目标，加env._targets，再加T_current+clip(cmd,±vmax·dt)且soft-limit裁剪的proposal。 | proposal是未保护的外部命令提案，不是预测actor/project之后的实际最终target。当前target可能与pending尾部重复，min不受重复影响。 |
| 同物理Jacobian与full distances | 用原COM provider对全部row生成J；forecast从full.dists开始，逐目标取min(d+J_F delta_F+J_U delta_U)。 | 线性endpoint envelope，未验证非线性位形/中途扫掠/PD时序；不是完整有序queue rollout，也不等于全量velocity危险行准入。 |
| 不改solver/actor/FIFO | 只覆写env.safety_dist_out。没有替换backstop.project、改backlog_aware、actor参数、targets/history/queue.push，或调用source.sample。原RiskTrace/输入路径保留。 | 多读Jacobian和scene_state不等于已证明全运行无副作用；需完整raw/sourcehash门。真实闭环状态变化后actor输出可分岔。 |
| 动态1024，abort/no drop | width=max(1,max_env mask count)，>1024明确ValueError并记capacity_failure；否则保留全部union。未固定padding1024、未调新的top-k。 | 原baseline仍512。预算确实改变，必须披露；对于同一mask且required≤512，容量参数仅是无作用的检查。>512时结果适用域需单列，不能说512资源下也成立。 |
| 区分baseline/candidate | baseline沿用super的dynamic512；queue_envelope使用新closure；manifest记录mechanism、forecast、helperSHA、actor32、strictFIFO6和容量。 | 候选log保存forecast_added_count/forecast_only_min/required_rows，但没有新增row ID或哪个target触发准入，不能据此证明row8974提前被纳入。 |

helper自身不覆盖官方distance/violation/min_margin，也不更换effective d_min或exemption。原始row metadata保持；新选中row的raw distance和closing送入原projector，原预测/限权逻辑仍作用。

新增判据只对纳入集起作用，但集合增加会改变有限迭代下的排序、交集、padding宽度以及project计算；这是准入策略的闭环后果，不应同时声称所有project数值或alpha/p轨迹不变。“solver源码/参数不变”和“solver输出不变”须区分。

## 4. 五个仍需区分的解释

1. **H1 准入滞后**：低当前closing且raw slack>10mm的row，被pending/proposal目标预测为危险却未在原bounded selector中留下。具体预言是候选在危险target发入队列前就纳入它，并改变对应issued target；若只在67后纳入，不能据此解释避免72首次负值。
2. **H2 已承诺队列及动态状态**：新issued增量打开、同帧实际运动关闭，与FIFO延迟相符；即便row已纳入，旧目标仍可穿越。反例是保留同旧队列的安全continuation；仅负endpoint forecast或线性无解不能证明物理不可避免。
3. **H3 限权/交集/有限迭代**：部分row的未放宽需求无解，或Dykstra/出口归箱未满足可行原集合；reported residual只针对最终h。需要原h、authority后h及实际输出的独立residual/feasibility对照。env47近零只排除该时点“最终约束大残差”，不排除全模型差异。
4. **H4 PD/非受控运动/线性forecast不充分**：同COM row的当前Jqd一致，不证明未来target大位移forecast精确或非受控手关节无影响。若ordered target/sweep本就危险，不能只归给PD；若surrogate安全但实际越界，才结合完整body twist/关节运动进一步区分。
5. **H5 exemption转换**：对某些table失败可能改变官方首报或有效lock line；env47实读帧不支持它，但不能排除其余99窗口。冻结同几何只换mask可区分“几何转负”和“旧负值被计入”，不能用放宽mask当安全成功。

原始临界补充不捕获所有stored/proposal target危险行，源码上存在H1空间；新helper针对它，但target线性预测仍不是可行性/连续安全证明。此轮**不改backlog_aware**；其可能的debit机制是另一个未测因素，不能混入当前结果归因。

## 5. CPU检查与实际风险

只AST提取union_mask/select，以torch CPU合成输入验证，未导入risk/Isaac、未写测试文件：
原selected+raw+forecast集合的独立预期相等；原row0有效而−1 padding不污染；重复ID去重；不同env行数/padding及metadata正确；≤比较边界正确；1024接受、1025明确abort/no-drop。六组helper不变量PASS；它们不是实际physics安全测试。

有限值风险：合成NaN forecast被比较为false且不会抛错，可能静默不纳入仅forecast危险的row。未观察到本轮真实NaN，不据此宣判当前run无效；但应把full d/J/target/forecast finite作为明确验证门。仅原dense有限和forecast_only_min有限不能证明所有未纳入row的forecast有限。没有详细forecast捕获时须披露此证据缺口，不能称全量预测已验收。

plan元数据风险：development_plan继承candidate文字仍说budget512、registered_utc仍是旧risk日期，并有旧prior_campaign_path/continuation_reason；其新的job/design/source绑定和实际helper/manifest才是当前候选。建议主线附加说明receipt列明继承描述字段不作为本轮执行事实，不修改已启动冻结协议。以实际protocol/manifest和签名helper1024为准。

## 6. 最小且有意义的后续验证门

A. **冻结和配对**：按development_plan核源码、actor、risk银行、输入tape及phys/effective配置；候选唯一控制因素为forecast准入，backlog_aware=false不变。baseline512与candidate1024预算显式并列。每cell都检查真实protocol complete和全部64×960，不把exit0、局部完成或slot0当整组成功。

B. **实际作用与提前期**：开发组另列added rows>0的env/step、最大required、超过512的env/step及任何1024中止。对row8974、7310至少保留一次只读机制receipt：t60..78稳定row ID、baseline/candidate是否纳入、trigger target issued index、forecast值、对应实际issued/delivered target。当前mechanism.npz只有汇总，不足以追溯这些ID。无需修改inflight source；后处理能证明的范围先交付，不能由汇总倒推出特定row。一个因果归因需target差先于有效FIFO作用时点，而不只是结果下降。

C. **所有门与原值**：所有64槽重算原official<0窗口/步数、四class、−5mm深度、首事件、reset、实际target增量速度箱及soft-limit；保留每个负值，不用容差清除小负值。全960帧A[t]=T[t−6]exact与兩substep约定保留，prefix零输入仍包含在主分母。若基线strict复现有差，保存exact FAIL及差异字段，不以浮点阈值声称一致。

D. **模型/可行性判别**：同pre snapshot固定实际row/J/alpha/p/bounds，分别检查未放宽h和authority后h；固定实际ordered队列比较目标几何shadow与真实row运动；exemption只在同几何shadow中换mask。有限线性feasibility只说明该模型，不证明非线性不可避免；端点安全也不证明中间无碰撞。候选汇总改善不能自动在H1/H2/H3/H4之间唯一归因。

E. **安全与进展并列**：违例减少与停滞/过度恢复须一起报告，至少实测运动、exec/command量、目标限位和压力到达后≥3帧80mm暴露。更少真实压力不能被当作同压力下安全提升。全部失败、capacity中止和不足额都保留，不重选bank、seed、幅度或候选。

## 7. 已冻结command-only holdout范围

已读holdout_registration和holdout_plan，核六个job的三组配对、bank实际SHA、system0方法、FIFO6及mechanism值：

| 新command seed | 复用bank seed |
|---|---|
| 1930442907 | 60317411 |
| 1940677742 | 80692357 |
| 1931983758 | 109441003 |

planned为192条新external command windows、384个baseline/candidate配对窗口，每条件独立新进程。三个旧bank q不是新初态覆盖；conditioning和相关性不消失，不能给pooled IID CI。开发seed60317411的旧tape不算新留出。计划保留全部结果且holdout后不换候选；本审查不提前读取或宣称holdout成功。

同q_initial与同外部tape支持闭环总效应比较；保护零输入恢复会让两方法早期q分岔，不能宣称首次随机到达时same-state。snapshot shadow才固定alpha/p；完整闭环仍用同冻结actor，其输出改变是中介，不是自动多因素调参。

## 8. 绑定与交付边界

| 当前实读证据 | SHA256 |
|---|---|
| MECHANISM_CONTRACT.md | 4557b45b4112a6469fff93fcfd471f336255f344f7b93c3989f9b7b35e2f0d6d |
| mechanism_runner.py | 174e297cdd9c8e48c00894e37426d7afbc48245f717de69fe2d3d4c6d173ce12 |
| development_plan.json | 4fcd5ce95c3a42c03a5a37b97e862141268994857a520943276db59f02c4117a |
| causal_analysis.json | e27bfd57259f1d9b70a481b34a484c3198c9bd88b3cd7b5bddac7d2fe1ac3ffc |
| causal_runner.py | 224ef9eaa11d381733086b21540e542fe620111188ef4ce40e5b80ee80840156 |
| analyze_causal.py | c3a79ba42bb1051c0bb14d7c5724724b8f3f0d8b5f4b07f2eedf241c33a721c7 |
| holdout_registration.json | 54e6cd57508162491434e6fe0d4922b8335b079774ac75a898ce9d41103de8fe |
| holdout_plan.json | d50be622b15f4c85712735f6d11a9029612f8f9dc97fb01fe8462789ebe189ad |
| latest risk results.json | eaca601b4d61463ad00405068a87565742c8d381bb7af42f1558d1c1387b6264 |
| latest risk analyze_risk.py | 7f5a741337af4db93f753804583681d9e5ffd1e030c0b5b9bea73c37fa74645d |
| latest risk risk_runner.py | 1363631d6f331816a9d73f0e62606668a2f51230885139580b675cdc0ea47135 |
| old perturbation VALIDATION.md | 012f9f3a684f3eb4fc39c3a39d27755cf80210ddd7662386e9ef46852a8e7659 |

contract/helper/causal_analysis/holdout_registration实际SHA均与development_plan中的冻结research绑定一致。causal_analysis.source_sha与实读analyzer一致。最新COM/backstop/环境源码此前也与risk campaign冻结hash核对相同。

独立检查的原三个保护NPZ：
60317411 SHA 6e7ea9438e2ee9d7e78080d76cd4dff0e8f66998005f5b5f2eac359dcaf82176；
80692357 SHA c10917ed6f2e4362e3e682ad130b27f36fa775008682b852ffe355afab1f6ba1；
109441003 SHA 6af54b6a100b01981b710ade1393d527bdcac691447491f3fa1aca4b70d9b954。
均在 /mnt/nas/data/lyf/double_hand/safety_risk_strata_20261004/cells/risk_{seed}_system0/。

开发因果数据路径为causal_analysis记录的 /mnt/nas/data/lyf/double_hand/safety_mechanism_20261005/diagnostic/development_baseline；此次只读取该已完成诊断，不修改并行父命名空间任何文件。新候选raw不在已验收结果中。

本审查到这里完成具体设计/源码sidecar。候选物理效果、全量sensor/连续碰撞/硬件性能和策略验收均未获PASS；待实际结果独立审查。未批准priority抢占、队列清空或执行器侧保护架构。
