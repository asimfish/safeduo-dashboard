# 8 个新增 paired failures：独立只读诊断

候选的既有父决策为 REJECTED。本报告只解释已闭合的实际轨迹与观测时序，不生成新的 policy outcome，不修改 H/RAW、冻结源、评分、FIFO 或旧审查文件。没有运行 GPU、重放、反事实实验，也没有制造图像。

8 个 producer actualwait0、48 个 audit actualwait0 已核对；两几何 auditor 的 strict/deep IDs 与 float32 minima，以及两接触 auditor 的 RAW samehand-inclusive maxima 完全一致。独立最终决策状态在本报告关闭时为：CLOSED_ACTUAL_WAIT0。
独立文件：/home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/ASTRA_ACCEPTANCE_REAL.json；actualwait0，COMPLETE_INDEPENDENT_FINITE_DECISION，candidate REJECTED，parent_decision_used=false。独立关闭时间 2026-10-08T02:15:55.549786+00:00；SHA256 c3cac9ff15000ec6bab9beea4d42b8797bbd1584c9d97896f732920f1e5fabe7。

**直接观测**：8 案首次失败均为已选中、已启用、非豁免的 U-table 行；5 案涉及 table_U、3 案涉及 table_F（以逐案 identity 为准）。每个失败行在当前最终线性投影中的行残差均为极小值/非正值，5/8 案整个当前最终 residual ≤1e-6，3/8 尚有其它当前 residual >1e-6。7/8 案失败当步扩宽；b0/env47 当步 LP0、未扩宽，但仍有 U residual。
所有8案实际 applied target 精确等于未扩宽的 issued[t-6]；最近几步扩宽目标尚未施加。冻结 J 下当前 issued 增量、以及 pre 到 delayed applied target 的位移都指向增大失败行间隙，而实测关节运动指向减小间隙。此符号与时序是观测，不能将“旧目标正在朝障碍关闭”或“延迟/惯性已经被证明是根因”写作事实。
8 案全窗口 RAW 手接触只有 b0/env47、b1/env15 超过 .1N，首次超限都晚于首次球失败；其余6案该 raw 手接触通道全窗口为0。无手接触记录不等于所有刚体/mesh无碰撞；.1N不等于全球伤害安全定义。

| case | 风险 stratum（初始化标签） | 首次 post step / time s | 失败球 / 桌 / row | pre→post m | 首个 RAW 手>.1N | 全窗峰 N | 深失败首 step | 精确首失败图 |
|---|---|---|---|---|---|---|---|---|
| b0/env9 | 1 F_L/U_L | 553 / 9.232964 | U_L/hand/wrist_3_link / table_U / 8901 | 0.000765748322 → -0.00175517052 | 无（全窗0） | 0 | None | 缺失：该env无现有图 |
| b0/env13 | 1 F_L/U_L | 438 / 7.316374 | U_L/hand/left_little_2 / table_F / 8916 | 0.00160150975 → -0.00226605684 | 无（全窗0） | 0 | None | 9+12张 |
| b0/env18 | 2 F_L/U_R | 222 / 3.716518 | U_R/hand/right_index_2 / table_F / 8987 | 0.000346779823 → -0.0039690733 | 无（全窗0） | 0 | 223 | 缺失：该env无现有图 |
| b0/env23 | 2 F_L/U_R | 191 / 3.199872 | U_R/wrist_2_link / table_F / 8973 | 0.00185485184 → -0.00198088586 | 无（全窗0） | 0 | 193 | 9+12张 |
| b0/env47 | 5 U_L/U_R | 868 / 14.482754 | U_L/hand/left_little_2 / table_U / 8917 | 0.0048615858 → -0.00131029636 | 869/1 @ 14.499420s | 12.6884003 | 869 | 9+12张 |
| b1/env15 | 1 F_L/U_L | 282 / 4.716478 | U_L/hand/left_index_2 / table_U / 8909 | 0.00394946337 → -0.00183814764 | 284/0 @ 4.741477s | 181.923981 | 283 | 9+12张 |
| b1/env34 | 4 F_R/U_R | 851 / 14.199432 | U_R/hand/wrist_3_link / table_U / 8980 | 0.00202304125 → -0.000491142273 | 无（全窗0） | 0 | None | 9+12张 |
| b1/env40 | 5 U_L/U_R | 817 / 13.632788 | U_L/hand/left_thumb_4 / table_U / 8941 | 0.0034549702 → -0.000870240852 | 无（全窗0） | 0 | 847 | 9+12张 |

step 使用零起点；macro post time=(t+1)×.016666s；sub0=t×.016666+.008333s，sub1=(t+1)×.016666s。stratum 是银行初始化的风险配对标签，不能把它当作实际失败几何类别。table 行 pair_sphere_idx 的第二元素0/1是桌索引，绝不是另一机器人球索引。strict 原评分使用 float32<0；deep 使用 float32<-.005，不引入 epsilon。

**全窗口当前机制及运动（独立 raw 重算）**

| block | LP checked robot-envframes | LP actual status2 | widened envframes | final residual>1e-6 envframes | checked UNKNOWN | batch secondcall frames（推导） |
|---|---|---|---|---|---|---|
| 0 | 8891 | 884 | 791 | 7841 | 0 | 396 |
| 1 | 10743 | 1187 | 1149 | 9535 | 0 | 637 |

LP infeasible 合计2071次（robot-envframe），widened1940 envframes，final remaining17376 envframes（分母122880）。checked raw solver statuses仅0/2；所有 queued_future flags 都为 UNKNOWN，physical_safety_certified=false。status2属于第一 narrow/current 问题；最终 bounds/h/alpha/gates/returned residual 来自重新实际 project，不能将 status2或0移称为 post-widen 或物理未来判定。
actual path ratio=1.17205901138；fourarm ratio=1.04832668597；all7strata ratios={"-1": 1.0127583450885997, "0": 1.0007061499954966, "1": 1.3258214520727405, "2": 1.4876453145047188, "3": 1.1253524328126792, "4": 1.2781732286449674, "5": 1.1489861713112464}。运动量没有整体抑制至注册阈值以下，也不能抵销 strict/deep/new_fail/initial/contact 拒绝。

**逐案最后16/6步、FIFO与接触**

每表包含首次失败前16步，再附失败当步；最后6步是表中 t-6..t-1。F/U 的 C 是 LP checked，S 是 raw status（-2未检查、0当前数值可行、2当前实际 infeasible），W 是该env扩宽，B2是any-env宽化引起的wholebatch第二次project（从flags+冻结源码推导），R是FINAL returned safety/alpha/actualbounds三者最大残差。dI/dA/dQ分别为issued增量、applied目标增量、实测关节增量L2，单位rad。所有26维目标/增量、六个待发目标、最终 bounds、alpha/p 和 baseline 对照保存在 JSON。

**b0/env9**：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint

首失败 snapshot：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/first_failures/step_0553.npz；SHA256 7061f038ad36a1374fd7f41b5e43a68e3a0a29caada3922f8565f82164f08384。失败行8901在最终 selected position=[10]，valid/gate=true，struct/contact exemption=false。U final h=-0.00447005033493，project行残差=0，effective issued行残差=5.12227416039e-09。
实际 raw距离变化=-0.00252091884613m；同一冻结J×实测dq=-0.00252122362144m，J×preqd=-0.159397289157m/s；J×issued增量=0.00447004521266m；J×(delayed applied target−preq)=0.0386866591871m。这些线性量未被当作非线性/动力学保证。
last6 status2次数=5，widen steps=[548, 549, 550, 551, 552]；last16 widen=[548, 549, 550, 551, 552]。当前 applied source step=547、source widened=false；最近wide首次进入物理的steps=[554, 555, 556, 557, 558]。表内17步两次微步的native position targets均已独立逐值核对为cell applied target，FIFO6/pre issue-history精确一致。

| step | C F/U | S F/U | W/B2 | FINAL R F/U | passes F/U | issued source→applied | dI/dA/dQ | RAW手max sub0/sub1 N | post table m（candidate / zero） |
|---|---|---|---|---|---|---|---|---|---|
| 537 | 0/0 | -2/-2 | 0/0 | 0/1.844e-09 | 19/30 | 531 → 537 | 0.00287549/0.00255405/0.00278821 | 0/0 | 0.001457423 / 0.001457423 |
| 538 | 0/0 | -2/-2 | 0/0 | 0/1.843e-09 | 20/30 | 532 → 538 | 0.0259986/0.00324082/0.00274183 | 0/0 | 0.0014574826 / 0.0014574826 |
| 539 | 0/0 | -2/-2 | 0/0 | 0/0 | 20/30 | 533 → 539 | 0.0259967/0.00294822/0.00272302 | 0/0 | 0.001457423 / 0.001457423 |
| 540 | 0/0 | -2/-2 | 0/0 | 0/2.328e-10 | 20/30 | 534 → 540 | 0.0060738/0.0036831/0.00271343 | 0/0 | 0.0014574826 / 0.0014574826 |
| 541 | 0/0 | -2/-2 | 0/0 | 0/1.84e-09 | 20/30 | 535 → 541 | 0.00496643/0.00342976/0.00271101 | 0/0 | 0.001457423 / 0.0014572442 |
| 542 | 0/0 | -2/-2 | 0/0 | 0/1.456e-09 | 20/30 | 536 → 542 | 0.00296907/0.00314323/0.0027221 | 0/0 | 0.0014573634 / 0.0014572442 |
| 543 | 0/0 | -2/-2 | 0/0 | 0/1.746e-10 | 20/30 | 537 → 543 | 0.00294877/0.00287549/0.00274613 | 0/0 | 0.001457423 / 0.0014571846 |
| 544 | 0/0 | -2/-2 | 0/0 | 0/0 | 21/30 | 538 → 544 | 0.00294362/0.0259986/0.0030537 | 0/0 | 0.001457423 / 0.0014573634 |
| 545 | 0/0 | -2/-2 | 0/0 | 0/0 | 21/30 | 539 → 545 | 0.00288186/0.0259967/0.00488183 | 0/0 | 0.0014573634 / 0.0014573634 |
| 546 | 0/0 | -2/-2 | 0/0 | 0/0 | 21/30 | 540 → 546 | 0.00464129/0.0060738/0.00680458 | 0/0 | 0.0014573634 / 0.0014574826 |
| 547 | 0/0 | -2/-2 | 0/0 | 0/5.844e-10 | 21/30 | 541 → 547 | 0.00672119/0.00496643/0.0079811 | 0/0 | 0.001457423 / 0.0014573634 |
| 548 | 0/1 | -2/2 | 1/1 | 0/0 | 21/30 | 542 → 548 | 0.0702466/0.00296907/0.00832466 | 0/0 | 0.0014575422 / 0.001457423 |
| 549 | 0/1 | -2/2 | 1/1 | 0/1.398e-05 | 21/30 | 543 → 549 | 0.0593275/0.00294877/0.00805402 | 0/0 | 0.0014574826 / 0.0014573634 |
| 550 | 0/1 | -2/2 | 1/1 | 0/1.326e-05 | 22/30 | 544 → 550 | 0.059309/0.00294362/0.00739501 | 0/0 | 0.0014573634 / 0.0014573634 |
| 551 | 0/1 | -2/2 | 1/1 | 0/1.3e-05 | 22/30 | 545 → 551 | 0.0705737/0.00288186/0.00650413 | 0/0 | 0.0014556944 / 0.0014573634 |
| 552 | 0/1 | -2/2 | 1/1 | 1.746e-10/1.319e-05 | 22/30 | 546 → 552 | 0.0673081/0.00464129/0.00538125 | 0/0 | 0.00076574832 / 0.0014573634 |
| 553 首失败 | 0/1 | -2/2 | 1/1 | 1.164e-10/0 | 30/30 | 547 → 553 | 0.067004/0.00672119/0.0040766 | 0/0 | -0.0017551705 / 0.0014573634 |

接触：全960×2微步×4arm的RAW hand-to-allpartner maximum全部0；首次球失败前16步及当步两微步也全部0。相同baseline全窗peak=0N。
现有图像组：[]。缺失：此精确env无相机/hand receipts；同stratum其它env图不替代，不重放/不制造。
paired zero该case全部960 post strict frames=0；其现有精确case图像组=[]。

**b0/env13**：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint

首失败 snapshot：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/first_failures/step_0438.npz；SHA256 c62e3c877c7506d33e96e8fb3a5561b53cc40828bb352bea913348f384701460。失败行8916在最终 selected position=[30]，valid/gate=true，struct/contact exemption=false。U final h=-0.0061099762097，project行残差=0，effective issued行残差=-8.38190317154e-09。
实际 raw距离变化=-0.00386756658554m；同一冻结J×实测dq=-0.00375587400049m，J×preqd=-0.24418272078m/s；J×issued增量=0.0061099845916m；J×(delayed applied target−preq)=0.0548080727458m。这些线性量未被当作非线性/动力学保证。
last6 status2次数=5，widen steps=[433, 434, 435, 436, 437]；last16 widen=[433, 434, 435, 436, 437]。当前 applied source step=432、source widened=false；最近wide首次进入物理的steps=[439, 440, 441, 442, 443]。表内17步两次微步的native position targets均已独立逐值核对为cell applied target，FIFO6/pre issue-history精确一致。

| step | C F/U | S F/U | W/B2 | FINAL R F/U | passes F/U | issued source→applied | dI/dA/dQ | RAW手max sub0/sub1 N | post table m（candidate / zero） |
|---|---|---|---|---|---|---|---|---|---|
| 422 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 416 → 422 | 0.0287361/0.030111/0.027688 | 0/0 | 0.0011810958 / 0.0011869371 |
| 423 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 417 → 423 | 0.0243989/0.0298741/0.0257268 | 0/0 | 0.0011847317 / 0.0011877716 |
| 424 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 418 → 424 | 0.0218572/0.0298541/0.0240048 | 0/0 | 0.0011879504 / 0.001188606 |
| 425 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 419 → 425 | 0.00822196/0.0298495/0.0225397 | 0/0 | 0.0011906326 / 0.0011892617 |
| 426 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 420 → 426 | 0.00738088/0.0216957/0.0209684 | 0/0 | 0.0011926591 / 0.0011897981 |
| 427 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 421 → 427 | 0.00583803/0.0172685/0.018998 | 0/0 | 0.0011940897 / 0.001190275 |
| 428 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 422 → 428 | 0.00411931/0.0287361/0.0174236 | 0/0 | 0.0011936724 / 0.0011903942 |
| 429 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 423 → 429 | 0.00686843/0.0243989/0.0161882 | 0/0 | 0.0011907518 / 0.0011906326 |
| 430 | 1/0 | 0/-2 | 0/1 | 6.116e-06/0 | 30/30 | 424 → 430 | 0.00888201/0.0218572/0.0153509 | 0/0 | 0.0011850297 / 0.0011906326 |
| 431 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 425 → 431 | 0.00854333/0.00822196/0.0146218 | 0/0 | 0.0011775196 / 0.001190573 |
| 432 | 0/0 | -2/-2 | 0/1 | 0/4.657e-10 | 30/30 | 426 → 432 | 0.00961271/0.00738088/0.0139547 | 0/0 | 0.0011690557 / 0.0011905134 |
| 433 | 0/1 | -2/2 | 1/1 | 0/0 | 30/30 | 427 → 433 | 0.0527249/0.00583803/0.0131853 | 0/0 | 0.0011604726 / 0.0011903942 |
| 434 | 0/1 | -2/2 | 1/1 | 0/0 | 30/30 | 428 → 434 | 0.0526751/0.00411931/0.012223 | 0/0 | 0.0011520684 / 0.001190275 |
| 435 | 0/1 | -2/2 | 1/1 | 0/2.874e-10 | 30/30 | 429 → 435 | 0.0420445/0.00686843/0.0110236 | 0/0 | 0.001144439 / 0.0011901557 |
| 436 | 0/1 | -2/2 | 1/1 | 0/0 | 30/30 | 430 → 436 | 0.0419354/0.00888201/0.00965948 | 0/0 | 0.0011377633 / 0.0011900365 |
| 437 | 0/1 | -2/2 | 1/1 | 0/2.092e-09 | 30/30 | 431 → 437 | 0.0417402/0.00854333/0.00812845 | 0/0 | 0.0011325181 / 0.0011899769 |
| 438 首失败 | 0/1 | -2/2 | 1/1 | 0/1.961e-09 | 30/30 | 432 → 438 | 0.0414713/0.00961271/0.0064963 | 0/0 | -0.0022660568 / 0.0011897981 |

接触：全960×2微步×4arm的RAW hand-to-allpartner maximum全部0；首次球失败前16步及当步两微步也全部0。相同baseline全窗peak=0N。
现有图像组：[(438, 'first_failure')]。每组9 parent原图+12手图，state和每张PNG的实际path/SHA见JSON；首次failure group来自相同post cell state，不能把微步接触峰当作图像时刻。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/multiview/first_failure/env_013/step_0438_state.json SHA256 6ea59316915aec4a0426f0467dc58bb2700dd7e355e4aad24287c37ff129d319；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/hand_views/first_failure/step_0438/env_013/state.json SHA256 a5708f0d352d1371fdcec25ba178fe25f7ddf095c0bf102f76b6dc5bfe3dc623。
paired zero该case全部960 post strict frames=0；其现有精确case图像组=[]。

**b0/env18**：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint

首失败 snapshot：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/first_failures/step_0222.npz；SHA256 7a308499a65b1a03353986317eece9b3f5be1eed3adc87240b5147a3ae4cccb6。失败行8987在最终 selected position=[10]，valid/gate=true，struct/contact exemption=false。U final h=-0.00685079814866，project行残差=4.65661287308e-10，effective issued行残差=-1.16415321827e-08。
实际 raw距离变化=-0.0043158531189m；同一冻结J×实测dq=-0.00431860378012m，J×preqd=-0.277042865753m/s；J×issued增量=0.00685080979019m；J×(delayed applied target−preq)=0.0460796654224m。这些线性量未被当作非线性/动力学保证。
last6 status2次数=4，widen steps=[218, 219, 220, 221]；last16 widen=[218, 219, 220, 221]。当前 applied source step=216、source widened=false；最近wide首次进入物理的steps=[224, 225, 226, 227]。表内17步两次微步的native position targets均已独立逐值核对为cell applied target，FIFO6/pre issue-history精确一致。

| step | C F/U | S F/U | W/B2 | FINAL R F/U | passes F/U | issued source→applied | dI/dA/dQ | RAW手max sub0/sub1 N | post table m（candidate / zero） |
|---|---|---|---|---|---|---|---|---|---|
| 206 | 0/0 | -2/-2 | 0/1 | 0/1.3e-07 | 30/30 | 200 → 206 | 0.00619819/0.0160598/0.019712 | 0/0 | 0.0014904439 / 0.0014527738 |
| 207 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 201 → 207 | 0.0269616/0.0249843/0.0160427 | 0/0 | 0.0014954507 / 0.0014528334 |
| 208 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 202 → 208 | 0.0209462/0.0133031/0.0129699 | 0/0 | 0.001499325 / 0.001452893 |
| 209 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 203 → 209 | 0.0206202/0.00954816/0.0104847 | 0/0 | 0.0015022457 / 0.0014528334 |
| 210 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 204 → 210 | 0.0134825/0.00484443/0.00853322 | 0/0 | 0.0015045702 / 0.0014528334 |
| 211 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 205 → 211 | 0.00671118/0.00560638/0.00700352 | 0/0 | 0.00150612 / 0.0014528334 |
| 212 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 206 → 212 | 0.00690504/0.00619819/0.00600715 | 0/0 | 0.0015073717 / 0.001452893 |
| 213 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 207 → 213 | 0.00413655/0.0269616/0.0057387 | 0/0 | 0.0015080273 / 0.0014527738 |
| 214 | 0/0 | -2/-2 | 0/1 | 0/1.164e-10 | 30/30 | 208 → 214 | 0.0039906/0.0209462/0.00581482 | 0/0 | 0.001508683 / 0.0014527738 |
| 215 | 0/0 | -2/-2 | 0/1 | 0/4.657e-10 | 30/30 | 209 → 215 | 0.00452793/0.0206202/0.00771154 | 0/0 | 0.0015090406 / 0.0014527738 |
| 216 | 0/0 | -2/-2 | 0/1 | 0/9.313e-10 | 30/30 | 210 → 216 | 0.00697987/0.0134825/0.00950364 | 0/0 | 0.0015054643 / 0.0014528334 |
| 217 | 0/0 | -2/-2 | 0/1 | 0/2.449e-07 | 30/30 | 211 → 217 | 0.0106764/0.00671118/0.0101835 | 0/0 | 0.0014986694 / 0.0014527738 |
| 218 | 0/1 | -2/2 | 1/1 | 0/0 | 30/30 | 212 → 218 | 0.0564312/0.00690504/0.0100714 | 0/0 | 0.0014917552 / 0.0014528334 |
| 219 | 0/1 | -2/2 | 1/1 | 0/0 | 30/30 | 213 → 219 | 0.0564312/0.00413655/0.0093724 | 0/0 | 0.001485616 / 0.0014527738 |
| 220 | 0/1 | -2/2 | 1/1 | 0/2.201e-06 | 30/30 | 214 → 220 | 0.0477237/0.0039906/0.00829737 | 0/0 | 0.0014803112 / 0.0014527738 |
| 221 | 0/1 | -2/2 | 1/1 | 0/1.102e-06 | 30/30 | 215 → 221 | 0.0474313/0.00452793/0.00698269 | 0/0 | 0.00034677982 / 0.0014527738 |
| 222 首失败 | 0/1 | -2/2 | 1/1 | 0/1.003e-06 | 30/30 | 216 → 222 | 0.0470259/0.00697987/0.00542888 | 0/0 | -0.0039690733 / 0.0014529526 |

接触：全960×2微步×4arm的RAW hand-to-allpartner maximum全部0；首次球失败前16步及当步两微步也全部0。相同baseline全窗peak=0N。
现有图像组：[]。缺失：此精确env无相机/hand receipts；同stratum其它env图不替代，不重放/不制造。
paired zero该case全部960 post strict frames=0；其现有精确case图像组=[]。

**b0/env23**：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint

首失败 snapshot：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/first_failures/step_0191.npz；SHA256 6398e77d7bbe26175d167224fb188938f597d9d35e7d12a47d2044153df233a9。失败行8973在最终 selected position=[2]，valid/gate=true，struct/contact exemption=false。U final h=-0.00569272460416，project行残差=0，effective issued行残差=7.91624188423e-09。
实际 raw距离变化=-0.00383573770523m；同一冻结J×实测dq=-0.00383655028418m，J×preqd=-0.224163755774m/s；J×issued增量=0.00569271668792m；J×(delayed applied target−preq)=0.039781358093m。这些线性量未被当作非线性/动力学保证。
last6 status2次数=5，widen steps=[186, 187, 188, 189, 190]；last16 widen=[186, 187, 188, 189, 190]。当前 applied source step=185、source widened=false；最近wide首次进入物理的steps=[192, 193, 194, 195, 196]。表内17步两次微步的native position targets均已独立逐值核对为cell applied target，FIFO6/pre issue-history精确一致。

| step | C F/U | S F/U | W/B2 | FINAL R F/U | passes F/U | issued source→applied | dI/dA/dQ | RAW手max sub0/sub1 N | post table m（candidate / zero） |
|---|---|---|---|---|---|---|---|---|---|
| 175 | 0/0 | -2/-2 | 0/0 | 0/1.281e-09 | 30/30 | 169 → 175 | 0.00412583/0.0313073/0.0143536 | 0/0 | 0.0011929572 / 0.0012335479 |
| 176 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 170 → 176 | 0.00472752/0.0270384/0.012819 | 0/0 | 0.0011935532 / 0.0012336075 |
| 177 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 171 → 177 | 0.0254041/0.0166983/0.0115277 | 0/0 | 0.0011940897 / 0.0012336075 |
| 178 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 172 → 178 | 0.0247085/0.00310405/0.010295 | 0/0 | 0.0011942089 / 0.0012336075 |
| 179 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 173 → 179 | 0.00414847/0.0129384/0.00926653 | 0/0 | 0.0011940897 / 0.0012334883 |
| 180 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 174 → 180 | 0.0382668/0.00458399/0.00842825 | 0/0 | 0.001193732 / 0.0012334287 |
| 181 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 175 → 181 | 0.0309955/0.00412583/0.00776453 | 0/0 | 0.0011931956 / 0.0012333095 |
| 182 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 176 → 182 | 0.0215307/0.00472752/0.007266 | 0/0 | 0.0011923015 / 0.0012333095 |
| 183 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 177 → 183 | 0.0145893/0.0254041/0.00730254 | 0/0 | 0.0011899173 / 0.0012331307 |
| 184 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 178 → 184 | 0.00343469/0.0247085/0.00829886 | 0/0 | 0.0011850297 / 0.0012330711 |
| 185 | 0/1 | -2/0 | 0/1 | 0/1.528e-06 | 30/30 | 179 → 185 | 0.00520664/0.00414847/0.00897531 | 0/0 | 0.0011787117 / 0.0012329519 |
| 186 | 0/1 | -2/2 | 1/1 | 0/4.922e-07 | 30/30 | 180 → 186 | 0.0700573/0.0382668/0.00855005 | 0/0 | 0.0011714995 / 0.0012329519 |
| 187 | 0/1 | -2/2 | 1/1 | 0/2.595e-06 | 30/30 | 181 → 187 | 0.0586653/0.0309955/0.00837998 | 0/0 | 0.0011642873 / 0.0012329519 |
| 188 | 0/1 | -2/2 | 1/1 | 0/2.644e-06 | 30/30 | 182 → 188 | 0.0587344/0.0215307/0.00821739 | 0/0 | 0.0011573732 / 0.0012328923 |
| 189 | 0/1 | -2/2 | 1/1 | 0/2.697e-06 | 30/30 | 183 → 189 | 0.0587256/0.0145893/0.00780261 | 0/0 | 0.0011511743 / 0.0012328923 |
| 190 | 0/1 | -2/2 | 1/1 | 0/2.757e-06 | 30/30 | 184 → 190 | 0.058665/0.00343469/0.00697188 | 0/0 | 0.0011458099 / 0.0012328923 |
| 191 首失败 | 0/1 | -2/2 | 1/1 | 0/5.18e-09 | 30/30 | 185 → 191 | 0.0585664/0.00520664/0.00590969 | 0/0 | -0.0019808859 / 0.0012329519 |

接触：全960×2微步×4arm的RAW hand-to-allpartner maximum全部0；首次球失败前16步及当步两微步也全部0。相同baseline全窗peak=0N。
现有图像组：[(191, 'first_failure')]。每组9 parent原图+12手图，state和每张PNG的实际path/SHA见JSON；首次failure group来自相同post cell state，不能把微步接触峰当作图像时刻。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/multiview/first_failure/env_023/step_0191_state.json SHA256 cf755f1849d20e0d880b458324e2b237e78c90d5a49c7ed57371b86d03db8953；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/hand_views/first_failure/step_0191/env_023/state.json SHA256 2a1a2888af2467dc39433a3ee25790b6f2fb29ba22c32412e61ff5054131a155。
paired zero该case全部960 post strict frames=0；其现有精确case图像组=[]。

**b0/env47**：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint

首失败 snapshot：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/first_failures/step_0868.npz；SHA256 bf38775f0f5d1a9e3ebb417219e94d85b63fa3e9fd3af00fc5485a89bfc5f8b4。失败行8917在最终 selected position=[12]，valid/gate=true，struct/contact exemption=false。U final h=-0.00242891302332，project行残差=-0.000252099474892，effective issued行残差=-0.000252101337537。
实际 raw距离变化=-0.00617188215256m；同一冻结J×实测dq=-0.00613509863615m，J×preqd=-0.354514330626m/s；J×issued增量=0.00268101436086m；J×(delayed applied target−preq)=0.044934373349m。这些线性量未被当作非线性/动力学保证。
last6 status2次数=3，widen steps=[864, 866, 867]；last16 widen=[864, 866, 867]。当前 applied source step=862、source widened=false；最近wide首次进入物理的steps=[870, 872, 873]。表内17步两次微步的native position targets均已独立逐值核对为cell applied target，FIFO6/pre issue-history精确一致。

| step | C F/U | S F/U | W/B2 | FINAL R F/U | passes F/U | issued source→applied | dI/dA/dQ | RAW手max sub0/sub1 N | post table m（candidate / zero） |
|---|---|---|---|---|---|---|---|---|---|
| 852 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 846 → 852 | 0.00366756/0.00361716/0.00365877 | 0/0 | 0.0011704862 / 0.0010486543 |
| 853 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 847 → 853 | 0.00365859/0.00361841/0.00365163 | 0/0 | 0.0011704862 / 0.0010485947 |
| 854 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 848 → 854 | 0.0383373/0.0036332/0.00364971 | 0/0 | 0.0011704862 / 0.0010485947 |
| 855 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 849 → 855 | 0.025656/0.00365133/0.00365225 | 0/0 | 0.0011704862 / 0.0010485351 |
| 856 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 850 → 856 | 0.0107113/0.00366502/0.00365716 | 0/0 | 0.0011704862 / 0.0010485351 |
| 857 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 851 → 857 | 0.00327963/0.00367003/0.00366222 | 0/0 | 0.0011704862 / 0.0010485351 |
| 858 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 852 → 858 | 0.00328073/0.00366756/0.00366544 | 0/0 | 0.0011705458 / 0.0010484755 |
| 859 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 853 → 859 | 0.0032817/0.00365859/0.00366643 | 0/0 | 0.0011704862 / 0.0010484755 |
| 860 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 854 → 860 | 0.00328281/0.0383373/0.00629182 | 0/0 | 0.0011689961 / 0.0010484755 |
| 861 | 0/0 | -2/-2 | 0/0 | 0/0 | 30/30 | 855 → 861 | 0.00364896/0.025656/0.00738541 | 0/0 | 0.0011651814 / 0.0010484159 |
| 862 | 0/0 | -2/-2 | 0/0 | 0/2.346e-08 | 30/30 | 856 → 862 | 0.00351994/0.0107113/0.00799237 | 0/0 | 0.0011595786 / 0.0010484159 |
| 863 | 0/1 | -2/0 | 0/0 | 0/2.701e-05 | 30/30 | 857 → 863 | 0.0327617/0.00327963/0.00836684 | 0/0 | 0.0011526048 / 0.0010484159 |
| 864 | 0/1 | -2/2 | 1/1 | 0/0 | 30/30 | 858 → 864 | 0.0851962/0.00328073/0.00857669 | 0/0 | 0.0011453331 / 0.0010484159 |
| 865 | 0/0 | -2/-2 | 0/1 | 0/0 | 30/30 | 859 → 865 | 0.00518704/0.0032817/0.00849484 | 0/0 | 0.0011381209 / 0.0010484159 |
| 866 | 0/1 | -2/2 | 1/1 | 3.171e-09/9.313e-10 | 30/30 | 860 → 866 | 0.0778693/0.00328281/0.0081039 | 0/0 | 0.0011311471 / 0.0010483563 |
| 867 | 0/1 | -2/2 | 1/1 | 3.171e-09/0 | 30/30 | 861 → 867 | 0.0778585/0.00364896/0.00753029 | 0/0 | 0.0011247098 / 0.0010482967 |
| 868 首失败 | 0/1 | -2/0 | 0/0 | 0/0.0003059 | 30/30 | 862 → 868 | 0.00321533/0.00351994/0.00677964 | 0/0 | -0.0013102964 / 0.0010483563 |

接触：首次球失败当步两微步max=0；首个RAW>.1N是step 869 sub1 @14.499420s，比球首次失败晚0.016666s。/World/envs/env_47/U_L/left_little_2 → /World/envs/env_47/TableU/geometry/mesh，0.483471483N。全窗口peak=12.6884003N在step 871 sub0 @14.524419s；/World/envs/env_47/U_L/left_little_2 → /World/envs/env_47/TableU/geometry/mesh。独立raw matrix norm与已审metric EXACT，完整非零微步时间线在JSON；后期峰不可移称为首失败时接触。
首接触raw：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/native_contacts/physics_events_1728_1760.npz；SHA256 f5246235120fd4b3717184eb6f92a132d621c5af798c76d9016ce6dd47408c80。峰raw：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/native_contacts/physics_events_1728_1760.npz；SHA256 f5246235120fd4b3717184eb6f92a132d621c5af798c76d9016ce6dd47408c80。baseline全窗peak=0N。
现有图像组：[(868, 'first_failure'), (869, 'first_native_contact')]。每组9 parent原图+12手图，state和每张PNG的实际path/SHA见JSON；首次failure group来自相同post cell state，不能把微步接触峰当作图像时刻。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/multiview/first_failure/env_047/step_0868_state.json SHA256 23d0cbb3b07224a62682013e5909bebe67cc0383c982ce5bd765d4e947ffec63；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/hand_views/first_failure/step_0868/env_047/state.json SHA256 1b042a83b20d013bf8b7ef5cdf8064c64b3ea53e1e2dfc0fa8c85a905290d12f。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/multiview/first_native_contact/env_047/step_0869_state.json SHA256 1411ae08cbd610ccd63a568137015ab6cf22d3398fc3b50fce173542fd6299d7；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/hand_views/first_native_contact/step_0869/env_047/state.json SHA256 40f2a5153f47fc804e04ee76a7634ba800500cffd8f36508a861e76c81f9e3e7。
paired zero该case全部960 post strict frames=0；其现有精确case图像组=[]。

**b1/env15**：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint

首失败 snapshot：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/first_failures/step_0282.npz；SHA256 01bd5a7b9efe5744399df1bc309ae886af4ad089ad86810ae7288e76b2491b58。失败行8909在最终 selected position=[63]，valid/gate=true，struct/contact exemption=false。U final h=-0.00781744811684，project行残差=-1.86264514923e-09，effective issued行残差=3.16649675369e-08。
实际 raw距离变化=-0.00578761100769m；同一冻结J×实测dq=-0.00577700138092m，J×preqd=-0.337386220694m/s；J×issued增量=0.00781741645187m；J×(delayed applied target−preq)=0.0235766563565m。这些线性量未被当作非线性/动力学保证。
last6 status2次数=3，widen steps=[279, 280, 281]；last16 widen=[279, 280, 281]。当前 applied source step=276、source widened=false；最近wide首次进入物理的steps=[285, 286, 287]。表内17步两次微步的native position targets均已独立逐值核对为cell applied target，FIFO6/pre issue-history精确一致。

| step | C F/U | S F/U | W/B2 | FINAL R F/U | passes F/U | issued source→applied | dI/dA/dQ | RAW手max sub0/sub1 N | post table m（candidate / zero） |
|---|---|---|---|---|---|---|---|---|---|
| 266 | 0/1 | -2/0 | 0/0 | 0/2.26e-06 | 3/30 | 260 → 266 | 0.0131483/0.0257087/0.0136632 | 0/0 | 0.0010564029 / 0.0011328161 |
| 267 | 0/0 | -2/-2 | 0/0 | 0/0 | 9/30 | 261 → 267 | 0.00365125/0.0196642/0.0118502 | 0/0 | 0.0010618269 / 0.001131922 |
| 268 | 0/0 | -2/-2 | 0/0 | 0/0 | 9/30 | 262 → 268 | 0.0217353/0.0146363/0.0106137 | 0/0 | 0.0010668933 / 0.0011312068 |
| 269 | 0/0 | -2/-2 | 0/0 | 0/0 | 12/30 | 263 → 269 | 0.0218145/0.0112841/0.00871289 | 0/0 | 0.0010713041 / 0.0011305511 |
| 270 | 0/0 | -2/-2 | 0/0 | 0/0 | 15/30 | 264 → 270 | 0.0217451/0.0114726/0.00726563 | 0/0 | 0.0010750592 / 0.0011300147 |
| 271 | 0/0 | -2/-2 | 0/1 | 0/0 | 7/30 | 265 → 271 | 0.00454971/0.0112469/0.00680586 | 0/0 | 0.001078099 / 0.0011295974 |
| 272 | 0/0 | -2/-2 | 0/1 | 0/0 | 11/30 | 266 → 272 | 0.00422414/0.0131483/0.00734534 | 0/0 | 0.0010803044 / 0.0011292994 |
| 273 | 0/0 | -2/-2 | 0/1 | 0/0 | 6/30 | 267 → 273 | 0.00425871/0.00365125/0.00641012 | 0/0 | 0.0010816753 / 0.0011290014 |
| 274 | 0/0 | -2/-2 | 0/1 | 0/0 | 10/30 | 268 → 274 | 0.00476671/0.0217353/0.00571094 | 0/0 | 0.0010813177 / 0.0011288822 |
| 275 | 0/0 | -2/-2 | 0/1 | 0/0 | 11/30 | 269 → 275 | 0.00495411/0.0218145/0.00600325 | 0/0 | 0.0010785758 / 0.0011288226 |
| 276 | 0/0 | -2/-2 | 0/1 | 0/0 | 11/30 | 270 → 276 | 0.00507874/0.0217451/0.00764346 | 0/0 | 0.0010727942 / 0.0011287034 |
| 277 | 0/0 | -2/-2 | 0/1 | 0/0 | 10/30 | 271 → 277 | 0.00705254/0.00454971/0.00910008 | 0/0 | 0.001064986 / 0.0011287034 |
| 278 | 0/1 | -2/0 | 0/1 | 0/6.069e-06 | 20/30 | 272 → 278 | 0.0107602/0.00422414/0.0100474 | 0/0 | 0.0010563433 / 0.0011286438 |
| 279 | 0/1 | -2/2 | 1/1 | 0/1.397e-09 | 19/30 | 273 → 279 | 0.0696459/0.00425871/0.0103773 | 0/0 | 0.0010474026 / 0.001128763 |
| 280 | 0/1 | -2/2 | 1/1 | 0/0 | 17/30 | 274 → 280 | 0.0697319/0.00476671/0.010168 | 0/0 | 0.0010388792 / 0.0011286438 |
| 281 | 0/1 | -2/2 | 1/1 | 0/0 | 16/30 | 275 → 281 | 0.0694438/0.00495411/0.00957197 | 0/0 | 0.0010310113 / 0.0011287034 |
| 282 首失败 | 0/1 | -2/2 | 1/1 | 0/0 | 25/30 | 276 → 282 | 0.0693626/0.00507874/0.0086326 | 0/0 | -0.0018381476 / 0.001128763 |

接触：首次球失败当步两微步max=0；首个RAW>.1N是step 284 sub0 @4.741477s，比球首次失败晚0.024999s。/World/envs/env_15/U_L/left_index_2 → /World/envs/env_15/TableU/geometry/mesh，37.5286179N。全窗口peak=181.923981N在step 723 sub1 @12.066184s；/World/envs/env_15/U_L/left_middle_2 → /World/envs/env_15/TableF/geometry/mesh。独立raw matrix norm与已审metric EXACT，完整非零微步时间线在JSON；后期峰不可移称为首失败时接触。
首接触raw：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/native_contacts/physics_events_0544_0576.npz；SHA256 66b7516adba4c61c8c759047f192bd734ac1d3679c24279b0533bb0f00fd61da。峰raw：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/native_contacts/physics_events_1440_1472.npz；SHA256 09c4b648904118f55daf560205c58af6331a56ac5cd88949cf35c4ed1d50d578。baseline全窗peak=0N。
现有图像组：[(282, 'first_failure'), (284, 'first_native_contact')]。每组9 parent原图+12手图，state和每张PNG的实际path/SHA见JSON；首次failure group来自相同post cell state，不能把微步接触峰当作图像时刻。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/multiview/first_failure/env_015/step_0282_state.json SHA256 74b0062694ca4ec71552ea5933aebfa074b646010cd6ffce6bcf8276f7c33523；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/hand_views/first_failure/step_0282/env_015/state.json SHA256 11e90895fbb10bee5a0a8e06c34409b3297319a65262c4994f81d51fa0c241d3。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/multiview/first_native_contact/env_015/step_0284_state.json SHA256 6539f963d6c83782521fff7c2df940d801b8fc955a2d437ce19fcc89653be7af；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/hand_views/first_native_contact/step_0284/env_015/state.json SHA256 effd3fadaba0ff2f4b749a7b850496951d8dfe3ecf323a6ca6d194c1975b3287。
paired zero该case全部960 post strict frames=0；其现有精确case图像组=[]。

**b1/env34**：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint

首失败 snapshot：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/first_failures/step_0851.npz；SHA256 af11c92d6d8fd3dbe1efa5aa02ec02b744757ab652bad3148dc67c41161afc5a。失败行8980在最终 selected position=[10]，valid/gate=true，struct/contact exemption=false。U final h=-0.00453274138272，project行残差=0，effective issued行残差=5.30853867531e-08。
实际 raw距离变化=-0.00251418352127m；同一冻结J×实测dq=-0.00251473905519m，J×preqd=-0.166722908616m/s；J×issued增量=0.00453268829733m；J×(delayed applied target−preq)=0.0410257875919m。这些线性量未被当作非线性/动力学保证。
last6 status2次数=4，widen steps=[846, 847, 849, 850]；last16 widen=[846, 847, 849, 850]。当前 applied source step=845、source widened=false；最近wide首次进入物理的steps=[852, 853, 855, 856]。表内17步两次微步的native position targets均已独立逐值核对为cell applied target，FIFO6/pre issue-history精确一致。

| step | C F/U | S F/U | W/B2 | FINAL R F/U | passes F/U | issued source→applied | dI/dA/dQ | RAW手max sub0/sub1 N | post table m（candidate / zero） |
|---|---|---|---|---|---|---|---|---|---|
| 835 | 0/0 | -2/-2 | 0/1 | 0/4.657e-10 | 17/30 | 829 → 835 | 0.00520168/0.00115037/0.0133092 | 0/0 | 0.0012753308 / 0.0012334287 |
| 836 | 0/0 | -2/-2 | 0/1 | 0/0 | 17/30 | 830 → 836 | 0.0260697/0.000410318/0.0118532 | 0/0 | 0.0012786686 / 0.0012334287 |
| 837 | 0/0 | -2/-2 | 0/1 | 0/0 | 16/30 | 831 → 837 | 0.0241413/0/0.0105878 | 0/0 | 0.0012809932 / 0.0012333691 |
| 838 | 0/0 | -2/-2 | 0/1 | 0/0 | 16/30 | 832 → 838 | 0.00319545/0.0257762/0.010283 | 0/0 | 0.0012829006 / 0.0012334287 |
| 839 | 0/0 | -2/-2 | 0/1 | 0/0 | 15/30 | 833 → 839 | 0.00176849/0.00809674/0.00990441 | 0/0 | 0.0012842715 / 0.0012334287 |
| 840 | 0/0 | -2/-2 | 0/1 | 0/0 | 15/30 | 834 → 840 | 0.00201199/0.00734907/0.00907062 | 0/0 | 0.0012850463 / 0.0012334287 |
| 841 | 0/0 | -2/-2 | 0/1 | 0/0 | 10/30 | 835 → 841 | 0.00296844/0.00520168/0.00813127 | 0/0 | 0.0012851655 / 0.0012334287 |
| 842 | 0/0 | -2/-2 | 0/1 | 0/0 | 10/30 | 836 → 842 | 0.00298594/0.0260697/0.00774587 | 0/0 | 0.001283735 / 0.0012333691 |
| 843 | 0/0 | -2/-2 | 0/1 | 0/0 | 10/30 | 837 → 843 | 0.00402768/0.0241413/0.00816895 | 0/0 | 0.0012794435 / 0.0012334287 |
| 844 | 0/0 | -2/-2 | 0/1 | 0/0 | 10/30 | 838 → 844 | 0.00521781/0.00319545/0.00931654 | 0/0 | 0.0012731254 / 0.0012334287 |
| 845 | 0/1 | -2/0 | 0/1 | 0/2.765e-06 | 10/30 | 839 → 845 | 0.0104436/0.00176849/0.00997805 | 0/0 | 0.001265794 / 0.0012334883 |
| 846 | 0/1 | -2/2 | 1/1 | 0/4.517e-08 | 10/30 | 840 → 846 | 0.0757017/0.00201199/0.0100257 | 0/0 | 0.0012582242 / 0.0012334287 |
| 847 | 0/1 | -2/2 | 1/1 | 0/4.47e-08 | 10/30 | 841 → 847 | 0.0758063/0.00296844/0.00959745 | 0/0 | 0.0012511909 / 0.0012334883 |
| 848 | 0/1 | -2/0 | 0/1 | 0/0.0005287 | 10/30 | 842 → 848 | 0.0213441/0.00298594/0.00884252 | 0/0 | 0.0012446344 / 0.0012334287 |
| 849 | 0/1 | -2/2 | 1/1 | 0/1.78e-06 | 10/30 | 843 → 849 | 0.0794162/0.00402768/0.00786781 | 0/0 | 0.00123927 / 0.0012334287 |
| 850 | 0/1 | -2/2 | 1/1 | 0/1.6e-06 | 11/30 | 844 → 850 | 0.079257/0.00521781/0.00651868 | 0/0 | 0.0012349784 / 0.0012334287 |
| 851 首失败 | 0/1 | -2/2 | 1/1 | 0/1.407e-06 | 17/30 | 845 → 851 | 0.0790315/0.0104436/0.00503388 | 0/0 | -0.00049114227 / 0.0012334287 |

接触：全960×2微步×4arm的RAW hand-to-allpartner maximum全部0；首次球失败前16步及当步两微步也全部0。相同baseline全窗peak=0N。
现有图像组：[(851, 'first_failure')]。每组9 parent原图+12手图，state和每张PNG的实际path/SHA见JSON；首次failure group来自相同post cell state，不能把微步接触峰当作图像时刻。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/multiview/first_failure/env_034/step_0851_state.json SHA256 ed0a278501d33a89b3c002abddbd8f354a1ad09601873fe403465e307cc98a75；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/hand_views/first_failure/step_0851/env_034/state.json SHA256 d487317318bed6ae693e7ef133e96e6f9e54ed2bf176afc87c5d5420e12679a2。
paired zero该case全部960 post strict frames=0；其现有精确case图像组=[]。

**b1/env40**：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint

首失败 snapshot：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/first_failures/step_0817.npz；SHA256 70bef381a99a0ecf41939e8fed13dd0e58d6b7762e442d3eaf479e040e800289。失败行8941在最终 selected position=[29]，valid/gate=true，struct/contact exemption=false。U final h=-0.00612673908472，project行残差=0，effective issued行残差=-3.77185642719e-08。
实际 raw距离变化=-0.00432521104813m；同一冻结J×实测dq=-0.00432283291593m，J×preqd=-0.251199066639m/s；J×issued增量=0.00612677680328m；J×(delayed applied target−preq)=0.0427902266383m。这些线性量未被当作非线性/动力学保证。
last6 status2次数=4，widen steps=[813, 814, 815, 816]；last16 widen=[813, 814, 815, 816]。当前 applied source step=811、source widened=false；最近wide首次进入物理的steps=[819, 820, 821, 822]。表内17步两次微步的native position targets均已独立逐值核对为cell applied target，FIFO6/pre issue-history精确一致。

| step | C F/U | S F/U | W/B2 | FINAL R F/U | passes F/U | issued source→applied | dI/dA/dQ | RAW手max sub0/sub1 N | post table m（candidate / zero） |
|---|---|---|---|---|---|---|---|---|---|
| 801 | 0/0 | -2/-2 | 0/1 | 0/2.328e-09 | 30/30 | 795 → 801 | 0.00558573/0.00353712/0.0128519 | 0/0 | 0.00099441409 / 0.00098749995 |
| 802 | 0/0 | -2/-2 | 0/1 | 0/2.328e-09 | 30/30 | 796 → 802 | 0.00475506/0.00316325/0.0108778 | 0/0 | 0.00099512935 / 0.00098749995 |
| 803 | 0/0 | -2/-2 | 0/1 | 0/2.727e-10 | 16/30 | 797 → 803 | 0.0255456/0.00259034/0.00933226 | 0/0 | 0.00099572539 / 0.00098749995 |
| 804 | 0/0 | -2/-2 | 0/1 | 0/2.741e-10 | 22/30 | 798 → 804 | 0.0201191/0.00416601/0.00819967 | 0/0 | 0.00099644065 / 0.00098749995 |
| 805 | 0/0 | -2/-2 | 0/1 | 0/2.701e-10 | 30/30 | 799 → 805 | 0.00293099/0.00385619/0.00725285 | 0/0 | 0.0009970963 / 0.00098749995 |
| 806 | 0/0 | -2/-2 | 0/1 | 0/2.715e-10 | 15/30 | 800 → 806 | 0.00325899/0.00382546/0.00652479 | 0/0 | 0.00099757314 / 0.00098749995 |
| 807 | 0/0 | -2/-2 | 0/1 | 0/1.198e-07 | 13/30 | 801 → 807 | 0.00312791/0.00558573/0.00587371 | 0/0 | 0.00099804997 / 0.00098755956 |
| 808 | 0/0 | -2/-2 | 0/1 | 0/3.002e-10 | 13/30 | 802 → 808 | 0.00290024/0.00475506/0.00548206 | 0/0 | 0.00099828839 / 0.00098749995 |
| 809 | 0/0 | -2/-2 | 0/1 | 0/3.039e-10 | 10/30 | 803 → 809 | 0.002562/0.0255456/0.00619794 | 0/0 | 0.00099846721 / 0.00098749995 |
| 810 | 0/0 | -2/-2 | 0/1 | 0/3.034e-10 | 10/30 | 804 → 810 | 0.00279705/0.0201191/0.00736089 | 0/0 | 0.00099846721 / 0.00098749995 |
| 811 | 0/0 | -2/-2 | 0/1 | 0/2.019e-07 | 10/30 | 805 → 811 | 0.00457554/0.00293099/0.00800691 | 0/0 | 0.00099846721 / 0.00098749995 |
| 812 | 0/1 | -2/0 | 0/1 | 0/0.001831 | 10/30 | 806 → 812 | 0.00507476/0.00325899/0.00834114 | 0/0 | 0.000998348 / 0.00098744035 |
| 813 | 0/1 | -2/2 | 1/1 | 0/0 | 10/30 | 807 → 813 | 0.0678854/0.00312791/0.00834633 | 0/0 | 0.00099816918 / 0.00098749995 |
| 814 | 0/1 | -2/2 | 1/1 | 0/0 | 10/30 | 808 → 814 | 0.0662149/0.00290024/0.0080214 | 0/0 | 0.00099799037 / 0.00098749995 |
| 815 | 0/1 | -2/2 | 1/1 | 0/3.979e-13 | 10/30 | 809 → 815 | 0.0659172/0.002562/0.0074566 | 0/0 | 0.00099787116 / 0.00098749995 |
| 816 | 0/1 | -2/2 | 1/1 | 0/1.397e-09 | 10/30 | 810 → 816 | 0.0657355/0.00279705/0.00668875 | 0/0 | 0.00099757314 / 0.00098749995 |
| 817 首失败 | 0/1 | -2/2 | 1/1 | 0/1.279e-12 | 10/30 | 811 → 817 | 0.0656747/0.00457554/0.00566171 | 0/0 | -0.00087024085 / 0.00098749995 |

接触：全960×2微步×4arm的RAW hand-to-allpartner maximum全部0；首次球失败前16步及当步两微步也全部0。相同baseline全窗peak=0N。
现有图像组：[(75, 'scheduled'), (480, 'scheduled'), (817, 'first_failure'), (959, 'scheduled')]。每组9 parent原图+12手图，state和每张PNG的实际path/SHA见JSON；首次failure group来自相同post cell state，不能把微步接触峰当作图像时刻。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/multiview/scheduled/env_040/step_0075_state.json SHA256 4d66eaf88fe711399b4992aa86b1ba2236b55e2afa064a7ef7f2568e568e7de0；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/hand_views/scheduled/step_0075/env_040/state.json SHA256 7128d6b76d46a258351cfb6af054714e82fbb338b126a61df56a4752a9f42e1f。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/multiview/scheduled/env_040/step_0480_state.json SHA256 08f283656dfb15e53d5bf39824e975d87c3f2ec349fbeb1f463384a82a4585ff；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/hand_views/scheduled/step_0480/env_040/state.json SHA256 d7438a017cd01bff6ece222f0aff6ba5abec01a781514835230e785375276371。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/multiview/first_failure/env_040/step_0817_state.json SHA256 34f2782ababb023b3fcc7da9629afc341876d3a41fa8b6a9e0f80d57178908ef；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/hand_views/first_failure/step_0817/env_040/state.json SHA256 2e2e8344d5b4b6b2b004f94e9543184c865088ce29cd4ddde33118d2035472ba。
图库state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/multiview/scheduled/env_040/step_0959_state.json SHA256 3fb84f275d84b433ede3b06706d91f662620ead2c3e77330cb5b09c975f94dc2；hand state /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/hand_views/scheduled/step_0959/env_040/state.json SHA256 0c62f4e7765b9fe76e24000be44d38b1a1413fce33147289ef5f97cd7f933632。
paired zero该case全部960 post strict frames=0；其现有精确case图像组=[(75, 'scheduled'), (480, 'scheduled'), (959, 'scheduled')]。

**共同 actual 初态 negative：b0/env58，risk label -1 / general**

b0_joint_reference：actual第一forecastpre t0 bad IDs=[58]；raw路径 /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/joint_reference/full_forecast/steps_0000_0032.npz；SHA256 7277037abe64bbf48cdb3ae7fddc00dfcdd7a50f1a471f2514f01904e6f09d68；v2 receipt /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b0_joint_reference_geometry.json。
b0_zero_inclusive：actual第一forecastpre t0 bad IDs=[58]；raw路径 /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/zero_inclusive/full_forecast/steps_0000_0032.npz；SHA256 c477d790ac4094c4f980359b8308b487520caa518b9eb6b99641f6398715ee9a；v2 receipt /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b0_zero_inclusive_geometry.json。
b0_box_admission：actual第一forecastpre t0 bad IDs=[58]；raw路径 /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/box_admission/full_forecast/steps_0000_0032.npz；SHA256 e4ec11086332303300e1aadcc83a2b62fb6381e1bec18661c48c4f2077507a03；v2 receipt /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b0_box_admission_geometry.json。
b0_adaptive_joint：actual第一forecastpre t0 bad IDs=[58]；raw路径 /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/full_forecast/steps_0000_0032.npz；SHA256 5af1dde4db06a0b28b30789ec606a1f45f405ef7dce6ab49d9d7affcd89839a7；v2 receipt /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b0_adaptive_joint_geometry.json。
b1_joint_reference：actual第一forecastpre t0 bad IDs=[]；raw路径 /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/joint_reference/full_forecast/steps_0000_0032.npz；SHA256 8add4ad0a608d89b7c7f5118e5ddc315b5069e1f6623f882f82dcdfd3a0defc3；v2 receipt /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b1_joint_reference_geometry.json。
b1_zero_inclusive：actual第一forecastpre t0 bad IDs=[]；raw路径 /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/zero_inclusive/full_forecast/steps_0000_0032.npz；SHA256 7ceaa7d3080990d6201a148cf06a57fbc5b012f4dcfb3d5b59226dee677c4375；v2 receipt /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b1_zero_inclusive_geometry.json。
b1_box_admission：actual第一forecastpre t0 bad IDs=[]；raw路径 /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/box_admission/full_forecast/steps_0000_0032.npz；SHA256 987f88acc33d8a5fd2c3eb15a5dfbc4a654c45a0b3553d0b56f67f6a27515364；v2 receipt /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b1_box_admission_geometry.json。
b1_adaptive_joint：actual第一forecastpre t0 bad IDs=[]；raw路径 /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/full_forecast/steps_0000_0032.npz；SHA256 48426e52d87e301f9cbde2d8e728a615c55983dbf7d28d90983f53e4484dda05；v2 receipt /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b1_adaptive_joint_geometry.json。

actual初态四类margins=[0.5076265335083008, 0.02308325469493866, -0.004501596093177795, 0.0017354786396026611]m；非豁免负行=[{"global_row": 8585, "class_id": 1, "scoring_class": "self_U", "sphere_id": 107, "sphere": "U_R/upper_arm_link", "arm": "U_R", "partner": "U_R/hand/right_thumb_4", "partner_index": 141, "pair_sphere_idx_second_field_is_table_index": false, "raw_distance_m": -0.004501596093177795, "exempt": false}]。共同U thumb1实际native q=.3499999940395355rad、qd=0，controlled q和原bank accepted_q完全一致，所有8job的四arm full-native注册字段（q/qd/root/velocity/issued/applied/两种六步queue）对同bankzero逐值完全相等，无epsilon。该初态bad四mode共同保留，不能用它解释其它8case的晚期新增table failures。
旧qualification raw映射：bank env58 → all_geometry batch831/env57，identity [1101342169,-1,10,0]。原四类margins=[0.5076262950897217, 0.023083209991455078, 0.01821773499250412, 0.0017354786396026611]m，rawtable min=0.0017354786396026611m；zero60 valid={'eligible': True, 'valid': True, 'bad': False, 'drift': 0.022122859954833984, 'velocity': 0.07971352338790894, 'min_table': 0.0017245709896087646}。旧raw q与本次受控arm requestedq一样，但该bank没有旧全native hand状态/手接触通道，不能转称为新common hand原生资格。当前raw t0直接否定新初始化geometry资格；没有old/new full-native反事实，不能认定thumb .35是已验证根因。
旧raw来源：/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/banks/all_geometry.npz SHA256 723ba6d7736da4e73c3cf0c02f476ff8d442d748eac9e1f962f43da35d49fc56；/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/banks/all_settling.npz SHA256 325864cf281e7ba0c7dae5c3728821fcd21ba34fafa7d7fa461f638b1de19e67。

**后续机制修复方向：全部待验证，不执行本轮调参或重跑**

H1 — Use dynamic tracking and irreversible FIFO state in the safety mechanism, rather than treating a feasible current selected-row increment as a physical barrier.
证据：All8 first failures are admitted/gated U-table rows. Their final same-row linear constraints are satisfied within tiny float32 residuals.；All8 measured moves close the failing row while the same frozen J projects both the current issued increment and pre-to-delayed-applied-target displacement in the opening direction.；Recent widened outputs have not reached PhysX at the first failure, and the applied target is exactly the non-widened issued[t-6].
下一机制方向：Before a new preregistered attempt, develop a conservative dynamics/target-debt response bound that explicitly accounts for measured q/qd, frozen FIFO6 and reachable actuator/PD response. Define failure/UNKNOWN behavior when no bound exists; preserve the original scoring.
所需反事实/证伪：Matched full-native state, identical queue and proposals, separately vary the anticipatory tracking envelope in a new registered attempt; log substep motion and row distances. Current observed timing does not prove delay or inertia alone caused failure.

H2 — Separate detection of a jointly infeasible narrow current problem from proof that a widened returned command is an adequate future response.
证据：2071 robot-env-frame LP status2 checks trigger1940 widened env frames;17376 env frames still have final current residual >1e-6.；5/8 new first failures have final current residual <=1e-6; the other3 have residual >1e-6 on some current constraint. All8 actual failing-row residuals are tiny/nonpositive.；b0/env47 first failure has current LP status0 and no env widening, while final U residual is0.0003059299197047949.
下一机制方向：Keep explicit numerical infeasibility/solverUNKNOWN/current-residual accounting and distinguish the row actually violated from the worst remaining row. A future design could pair a conservative dynamic feasible set with an explicitly registered failure response; widening alone is not a future safety certificate.
所需反事实/证伪：Compare residual-aware dynamic response and unchanged controller from the same actual full-native/FIFO state in a separately frozen attempt; isolate final solver error, authority/R19 effects and dynamics. Do not change p/alpha/30pass/R19 in the current frozen attempt.

H3 — Audit nonlinear rollout/linearization limits and batch sensitivity with explicit counterfactuals; adding these observed failure rows again is not enough.
证据：All8 failing rows were selected and engaged. Pre measured raw distances are positive, post raw distances become negative.；The original production second project is whole-batch whenever any env widens; batch-global early stopping can affect unchanged env iterates.；b0/env47 first failure has neither env widening nor a whole-batch second project. Earlier whole-batch calls and their unchanged-env effects remain a separate source boundary, not an observed cause of this first failure.
下一机制方向：For a future attempt, expose batch-call attribution and reproducible per-env dynamic prediction error; test row-distance response and unchanged-env equivalence under matched batch composition without weakening admission or the full9021 score.
所需反事实/证伪：Matched-state batch composition/per-env replay in a new registration is needed to isolate the batch effect. This report ran no replay and establishes no batch-induced causal failure.

H4 — Qualify actual common initialized hand geometry and raw contact independently of the prior arm-only bank.
证据：b0/env58 has a nonexempt negative U-self row in actual raw t0 for all4 modes; previous bank qualification has positive stored margins and valid zero60.；The bank channels store controlled q/margins/settle summaries; they do not preserve old full-native hand states or contacts.；Two new sphere-failure cases later produce raw hand-table contacts; six do not. Contact .1N and sphere strict0 describe different observations.
下一机制方向：Before another bank/attempt, preregister full-native common initialization qualification and substep hand contact observation in the intended hand context, retaining rejected samples/attempts and unchanged contact exemptions/strict scoring. Do not filter this completed bank by policy outcomes.
所需反事实/证伪：Full-native matched old/new hand initialization with actual t0 geometry and contacts is required to attribute the case58 change. Current raw evidence cannot isolate thumb opening as its cause.

**边界与绑定**

wholebatch第二次original project的global early-stop可能影响未扩宽env；已在冻结adaptive源码中记录。b0/env47首失败当步没有第二次project；该case不支持首失败当步的batch-secondcall归因。这里仅按any widened flag归属调用，并未测量其因果贡献。authority/h/alpha/gates在最终实际project重算；数值first LP无post-widen保证，R19/30pass/p/alpha/硬评分/FIFO6没有修改。
sphere score、原生手normal force、图库绑定是不同观测通道。未知queued future永远UNKNOWN；构造器contacts未观测，完整friction/fullmesh/occlusion未认证；不声称 exactrestore、全球安全、硬件批准或生产promotion。
图像覆盖6/8，缺b0/env9与b0/env18；按每stratum首失败取图导致缺失符合注册观测范围，不代表不存在这些失败。现有6组首失败各21PNG的原始文件SHA已核对，不解读遮挡/mesh安全。

| 关键文件 / 原始通道 | SHA256 |
|---|---|
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/ACCEPTANCE_RESULT.json | 05ea4dbddbf41738b23ded34f8799f819d3703fa8bcb11725a098aba7ee817d1 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/ACCEPTANCE_REGISTRATION.json | 55f4dfc4c648f44b6d42218ccbd9f94591017b543f8c9720a2a9651e373490cd |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/ACCEPTANCE_CRITERIA.json | 01a5b45d1ada7635946f15c1b2e0cee141e8128266bdbed423844cf833c054a3 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/PAIRING_REGISTRATION.json | 1247213588c332c32af34d736d7cf3b77ae36a873bb26d384937332a91bbbea7 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/COMMON_HAND_CONTEXT_REGISTRATION.json | 0984ed6d3f131876f27e4182dcd72910fc3bfcff4d14b8c99839512a494e263c |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/INITIAL_STATE_ACCEPTANCE_REGISTRATION.json | e7649098e7b5a054654b22f63208e7dfadf64273cbb799e31eaa689037c91bc9 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/CAMERA_COVERAGE_REGISTRATION.json | 8b9012f50a6ce61476e10af27034f77eb2ce51dd2737a9db61dfb49a36df58d9 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/adaptive_reference.py | 562da73abffb124e8e39908ec973a2de6f2712ad5ee1d1f191ccec7f4754868e |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/projection_diagnostics.py | 1e2c371d7a8d4a1e7c5c943e585225141412e63b0a70ea98989534bc600a1869 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/guard_native.py | 4103d9c05ed64113575c00b6654c47477dac64ca168db16b6781bdad273a34d6 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/native_contact_runner.py | d633fff5c9ecb953ccfce2705d60efdd64e6844f8bb340b27307b5220a21ab7d |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/audit_geometry_v2.py | 98d0c7e20fcb43b8f68f1146ae814dc42454c1bf4e2d68bd0d7d317cc71787a6 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/audit_contacts.py | 3f492f5416ac961c82a8898647dc7ce712fda5415cecc926b28bbf86b42ce3d6 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/astra_native_audit.py | ea94c3e0d46ab1ce862291d20bae6206216de03b5f4fbc07040942705211ac10 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/audit_hand_views.py | 132df924e249642dd19ea534373b8a83c99c2248cdb892ade4aebd8549b8ef2a |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/astra_acceptance_decision.py | fbdd7b234a72b6fbaea5404e182530831643f8997033c7aa829fbfccab977f13 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/ASTRA_ACCEPTANCE_DECISION_REVIEW.json | c9987ffd6922a1da822bfb517fa821226cf3893f061f8a38c569e418a9fdae07 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/ASTRA_ACCEPTANCE_DECISION_TEST_RECEIPT.json | a25203e5f8ebc44c27b178f59de7778d4b70ff4686bef1dd3a0b5a4e230d2690 |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/cell_001.npz | ac3ba212352072b07048768cbc56d9c50939e46f62df26037580bf6c7a66c426 |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/project_diagnostics.npz | 7d1ad538d6099199c3584c07761c45593e36c29ffb4d4c51635df096d0883c2b |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/adaptive_joint/first_failure_receipts.json | 4896c62c275fbba8620ffa3dd09b438d199c65fc20780d8860067a17e95cc2a8 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b0_adaptive_joint_contacts_native_contact_metrics.npz | 2f171bbdb56479bfa3b9ed6371eab3c1b2be6d878e937537a7c0380c1734f883 |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/zero_inclusive/cell_001.npz | a71ea04386e0912b7923499f3260490d2453cfa6344574e42bf3c9536f3b7bc3 |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_0/zero_inclusive/project_diagnostics.npz | 73a7a3bff4b3c6d3e37f580004684f2ebe1eef01eebb7921ac5a78ae6e6f3f30 |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b0_zero_inclusive_contacts_native_contact_metrics.npz | c645aee5f06adc082ba9b291db46d8b79eff692e5587ed47ff812ec37caa099d |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/cell_001.npz | 641afe26bae6f1d2518c606cd53081ce915695e481d23af0c351457530fb2477 |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/project_diagnostics.npz | b63d41e0b065f989612cd045f3ea17e152415a602ccaf9b044fd046544381dc3 |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/adaptive_joint/first_failure_receipts.json | 556a3c21b2b3da48391152a16b95b69832ed50dbe3ba4f69686ef5d637652b9a |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b1_adaptive_joint_contacts_native_contact_metrics.npz | a69432ac79b4a92ee6182a67a7a525a3295435bd9f5ded3d4816d94231a4d636 |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/zero_inclusive/cell_001.npz | 10be69baff55652dbe664b8b4100b3bf88a25d3f5b2e61d4c23f7011929afc07 |
| /mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance/block_1/zero_inclusive/project_diagnostics.npz | 9b3a665f0b40ddd557956293811fc25ccd8b7ac0789cd3712f5c357260897acd |
| /home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/b1_zero_inclusive_contacts_native_contact_metrics.npz | e394271582bf163a4c307eebe91530dcc6476730758ff141c1598aa437bc0f0c |

JSON保存完整consumed SHA ledger、48actualwait绑定和冻结source前后同SHA证明。此报告关闭时通过3246条有界一致性assertion；它们不是新的实验验收或反事实因果验证。独立decider已经actualwait0关闭并拒绝candidate；本报告不改该文件/日志。sidecar有界分析进程也已实际exit0。
详表JSON：/home/liyufeng/safeduo/artifacts/safety_native_acceptance_20261007_2307/ADVERSE_CASE_DIAGNOSIS.json；SHA256 41c54e22fdf10bb42e574ff3f686e540aff1eec0b2960e5793edec00cc13d8f5。
