# ASTRA：完整fresh四格独立证据终审

2026-10-05T05:40:12.151086+00:00；**PASS_EVIDENCE_REVIEW**。**candidate_physical_status=BLOCKED；safety_strategy_approved=false；hardware_approved=false；production_promoted=false。**

实际读取12个closed cell、input/银行/标签、768个runtime episode及全部原始official_margins。独立NumPy评分覆盖每格960×64×4：任意非豁免margin<0记违规，<−.005m记deep，不加容差，不抹去近零负值。全部12格端点、deep、类计数、最小margin、首个失败、失败env和768个method-case记录与最终v4结果逐项一致。invalid=0，合计768方法窗口来自192个配对初态/tape案例，不能当768个IID案例。未模拟、未改冻结源或旧报告。

## 独立重评分

| mode | 违规/192 | deep | cross/selfF/selfU/table | 最小margin mm |
|---|---:|---:|---|---:|
| baseline_guard | 126/192 | 111 | 6/30/52/101 | -70.524089038 |
| admission_guard | 42/192 | 30 | 0/1/4/39 | -39.435066283 |
| envelope_guard | 31/192 | 22 | 2/1/5/27 | -47.780960798 |
| joint_guard | 24/192 | 16 | 1/0/1/23 | -18.908962607 |

类别可重叠，不能相加当独立窗口数。deep是球模型的深负裕度，不是实测损伤或双物搬运QA。全部方法零输入前60步都没有违规；整个960帧含所有prefix保留。

| 对比 | 案例数 | 救回 | 新增失败 | 都失败 | 都无违规 |
|---|---:|---:|---:|---:|---:|
| baseline_guard→admission_guard | 192 | 85 | 1 | 41 | 65 |
| baseline_guard→envelope_guard | 192 | 101 | 6 | 25 | 60 |
| baseline_guard→joint_guard | 192 | 108 | 6 | 18 | 60 |
| admission_guard→joint_guard | 192 | 33 | 15 | 9 | 135 |
| envelope_guard→joint_guard | 192 | 10 | 3 | 21 | 158 |

组合虽减少总违规，仍24/192、deep16、最差−18.908963mm，且相对baseline新增6例，相对admission新增15例。它不是逐case单调改进，也没有支持物理策略准入。

### 全部baseline新增失败

| seed/env | mode | 首次负裕度step | deep | 四类最小mm |
|---|---|---:|---|---|
| 1701627244/21 | envelope_guard | 632 | True | 14.050796509/23.082361221/25.730550766/-9.074045181 |
| 1701627244/34 | envelope_guard | 286 | True | 44.493949890/23.082464218/16.327619553/-14.880105972 |
| 1701627244/49 | envelope_guard | 486 | True | 287.755554199/23.081377029/16.598165512/-12.645244598 |
| 1701627244/21 | joint_guard | 893 | False | 14.050796509/23.082361221/25.730550766/-1.092532635 |
| 1701627244/49 | joint_guard | 488 | False | 287.755554199/23.081377029/16.598165512/-3.660142422 |
| 1651261960/34 | envelope_guard | 417 | True | 18.450216293/23.082643509/-18.107355118/-6.501356125 |
| 1651261960/34 | joint_guard | 916 | True | 18.450216293/23.082643509/3.305778027/-5.559841156 |
| 1651261960/40 | joint_guard | 300 | True | 174.456420898/18.219202042/14.934807777/-11.811062813 |
| 779659058/3 | admission_guard | 279 | True | 312.126770020/17.224401474/13.635173798/-10.812081337 |
| 779659058/9 | envelope_guard | 547 | True | 27.360483170/16.770347595/12.480243683/-13.964756966 |
| 779659058/41 | envelope_guard | 159 | True | 446.433715820/13.060465813/-8.746236801/-21.878711700 |
| 779659058/9 | joint_guard | 522 | False | 27.360483170/16.707689285/15.050352097/-2.147242546 |
| 779659058/41 | joint_guard | 179 | True | 421.020904541/13.059184074/8.223474503/-10.501437187 |

13条是method-case记录，涉及8个不同匹配初态/tape案例，不是13个独立新案例。最终13条regression_context的集合/首次失败/sourceSHA及已消费输入SHA均核对；nearest-row、previous selected/exempt等几何上下文为父线程源绑定派生值，本次未重复读取其全部局部forecast chunk。

父上下文13条首行都标previous selected，因此不能把所有新增失败解释为漏选。env34/1701627244的E在step286、table行8974此前已选，返回安全残差0、实际target安全残差约9.28e−11m仍发生负几何，说明目标层残差不保证六步FIFO后的非线性PD运动。joint env21/1701627244在step893的最近table行8925则标previous exempt=true；规则/配置未变不等于逐帧豁免状态不变。本审查不由这些posthoc描述认定唯一根因。

## 实际暴露、覆盖和随机性

独立从12格pair_margin[66:]重算指定pair至少3帧<.080m，24条暴露记录及每case帧数与父结果完全一致。每方法每pair分母24，六pair共144目标风险案例，另48广域稳定条件案例；不足曝光全部保留。

| mode | 六pair exposed/24 | 六pair underexposed |
|---|---|---|
| baseline_guard | 24 / 21 / 23 / 23 / 21 / 24 | 0 / 3 / 1 / 1 / 3 / 0 |
| admission_guard | 24 / 20 / 23 / 23 / 21 / 24 | 0 / 4 / 1 / 1 / 3 / 0 |
| envelope_guard | 24 / 22 / 23 / 23 / 21 / 24 | 0 / 2 / 1 / 1 / 3 / 0 |
| joint_guard | 24 / 22 / 23 / 23 / 21 / 24 | 0 / 2 / 1 / 1 / 3 / 0 |

实际银行accepted_q/标签与native初态/recipe逐位一致；三银行192行互不重复，配额各六pair×8+16。四方法初态、标签和完整962步tape逐位匹配，执行cmd等于前960步。独立按每case实际960帧输入SHA确认192段互异，且与command_audit本轮SHA列表一致。与指定旧namespace的256条tape无重复是父command audit已列范围的证据，本次未重读旧raw；不升级为所有历史输入唯一或IID。

实际26关节10格边际占用、EE10cm体素与joint path已读最终coverage并保存摘要于同名JSON；这些网格/体素本次未重算，且没有完整26D联合可达体积分母。银行资格几何/零输入静置及旧384重复性也未重复审。它们是经稳定资格筛选的风险条件分布，禁止pooled可靠性CI或全空间安全覆盖声明。

## 队列、正残差和版本

独立重建全部12×960帧六个oldest→newest pending位置、original project history和实际actuator target，逐位FIFO6一致。dt=.016666s，名义延迟.099996s；actual controller_target增量、issued target−pre q debt以及project返回↔exec↔外部raw tape的绑定均一致。上述是实际目标/记录层证据，不是PD运动兑现保证。

分别读内部solver、R19 returned、最终target的保存安全/alpha/bound残差，独立归约maximum、positive和>原1e−6诊断计数，与v4cache吻合；保留个体不可行下界。这里没有重新算每帧J/半空间，1e−6不用于碰撞端点。

| cell | returned安全max mm | target安全max mm | F/U正单行不可行环境帧 | max selected/budget |
|---|---:|---:|---|---|
| baseline_guard_1701627244 | 14.852094464 | 14.852044173 | 0/0 | 60/512 |
| admission_guard_1701627244 | 10.724270716 | 10.724299587 | 0/0 | 126/1024 |
| envelope_guard_1701627244 | 15.685178339 | 15.685150400 | 17/3095 | 65/512 |
| joint_guard_1701627244 | 15.685178339 | 15.685150400 | 9/2894 | 69/1024 |
| baseline_guard_1651261960 | 17.566338181 | 17.566272989 | 0/0 | 158/512 |
| admission_guard_1651261960 | 15.850340948 | 15.850350261 | 0/0 | 226/1024 |
| envelope_guard_1651261960 | 9.307773784 | 9.307783097 | 8/2170 | 67/512 |
| joint_guard_1651261960 | 9.307773784 | 9.307783097 | 7/2728 | 119/1024 |
| baseline_guard_779659058 | 6.716739386 | 6.716726348 | 0/0 | 69/512 |
| admission_guard_779659058 | 15.257289633 | 15.257236548 | 0/0 | 149/1024 |
| envelope_guard_779659058 | 13.079131022 | 13.079145923 | 8/4258 | 56/512 |
| joint_guard_779659058 | 13.079131022 | 13.079145923 | 9/4073 | 128/1024 |

所有实际selected数未超注册预算；本批最大仅226，不能把CPU620行seam当实际>512物理样本。原A因素仍是raw proposal预测集合，不能说其对reference-limited proposal保守；shadow漏项是被动诊断，不是控制改动。E=.050rad是可达reference envelope，不是backlog_aware债务debit、抢占或rebasing。E0下“reference unreachable”应理解为诊断，不代表已施加E因素。

逐cell核对实际v4cache的schema/path/cell/executed SHA和37项规范输入集合；本次直接消费的protocol/dense/guard_metadata/project_diagnostics SHA与cache一致，且cache内容与results内嵌audit全等。父v4完成全部30chunk/全9021 forecast/union重审并绑定全输入字节，本次不重复读62GB、也不独立重建全9021 J。全J有限检查属于producer；此前selected J的18×64/4564项独立算术报告只绑定SHA复用，未重跑，U12/11单行不可行环境快照与正残差保留。

v4实际analysis SHA与offline freeze一致；BR04 shared cached/fresh dense门、actualbytes compile/exec已由独立最终绑定报告确认。12格物理源码/actor/resolved config/backstop/coordinator元数据一致，实际helper SHA核对冻结且未改。生产279源码全树没有在本次再次hash，属于注册及父终审来源，不把metadata比较冒称新全树验证。

## 相机与正式报告

实际native_visual_verification PASS42组/378图，照片绑定own视觉state。数值baseline46/joint6与视觉47/7均FAIL_EXACT_REPLAY，终审完整保留。native before/after全64严格相等只证明每组瞬时状态保持，不证明未来轨迹；fiveplanes不是实际near/far认证、完整link silhouette或无遮挡证明。

本次仅绑定父全量相机receipt和既有独立相机观察报告（3state、27PNG SHA/27view算术、6张实际图），没有重看378图。原3终态样本不是全部失败/峰值或连续接触图。

已读实际REPORT.md及build_report.py：全768/invalid0、类重叠、全部新增失败、非IID/条件分布、真实FIFO时长、正残差与相机forward FAIL均明确。四臂moving比例及exec/input L2是父最终分析的描述值，本次未另重算；它们不是搬运成功率或物理安全认证。有限值/fault/软件测试沿用已绑定历史报告，不重复测试。候选/硬件未批准，“证据完整PASS”不覆盖仍有负裕度的物理BLOCKED。

## 交付发现与字节绑定

未发现本次独立端点/配对/暴露/FIFO/记录残差归约或v4版本绑定的交付阻断。物理24/192与新增反例是明确保留的科学结果，不能据总数改善批准安全策略。

| 文件 | SHA256 |
|---|---|
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/results.json | 9f0212ec54aca3a5b1fe7e37be5acc5e1669a973227d81c0b3f4102a4fb31a56 |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/REPORT.md | dc28ed8d83a70c62499d956e43fe50081c2e822539e571e8107dbfb727063032 |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/holdout/campaign.json | 150266b3420c9f8b549850ac2749ad78f37f482df27e2415e4ac5be566d5539f |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/holdout_plan_v2.json | 0cfc248bb5d65d6a628a30122c5bf95ff00a4da3f46a4220f595ab2675a0bf0a |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/offline_tool_freeze_v4.json | 5a8133e916d34a4a20e3a6a37a048bf125bb0511d51b996a73c3de794a5696b2 |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/analyze_v4.py | 76b04971b40d9a439ba4faf30da012bf5eb457db6ba71984cd5c6b5206c81255 |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/native_visual_verification.json | 17c540c5ae8d66012ca8372427dfe5d399d566a7bc13f5ac6a8da987b533446a |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/command_audit.json | 57fbb28c4a9abfb9f25b089427daf571eec28ad3ae5bd1825a50a8eb90413a1a |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/regression_context.json | aba4f41c69a86de31c3c0ecfaf18856fab7f1eb71172c6bde43f953c5f576aed |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/failure_context.py | 705fe5f15b8550ac5a0c293671f1c4620e1619d4ec1a57af12bc0475c672e9e9 |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/ASTRA_FRESH_SNAPSHOT_REVIEW.json | 71bc8844cd60f8939e2d229b75626132679c58f076b4c3d1912bb3cda1a47930 |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/ASTRA_BINDING_FINAL_REVIEW.md | efe65abaa92888fcc53845c11c74987ceb01e038e1aad937aa4a3cff8ffefcbb |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/ASTRA_CAMERA_OBSERVATION_REVIEW.md | fe016adf39c3c5ff7783b3b9daf557181fec670d29f52a79b20fcc43dec0c636 |

所有12个dense、input/protocol/银行、diag/v4cache及逐case明细SHA见同名JSON。原有设计/边界/快照/相机审查字节未修改；未改生产、候选、报告或主分析。

终态报告绑定补记：已读当前REPORT的previous_exempt列及“13条method记录＝8个配对案例”澄清，与既有终审上下文一致；仅更新报告字节绑定，原端点重评分、证据PASS与物理BLOCKED不变。终审Oracle为inline Python执行，没有对应独立源码文件，未补造posthoc源码或执行身份；既有快照Oracle文件保持原样。
