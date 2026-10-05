# ASTRA：18个冻结快照的联合可行性 oracle

本次状态 **PASS_BOUNDED_JOINT_FEASIBILITY_ORACLE**，仅表示所选线性约束集的独立 CPU 核验完成。18例中13例可行、5例不可行；3例虽可行，原返回安全残差仍超过原求解容差 `1e−6`。没有物理运行、策略准入或硬件安全批准。

## 范围、冻结与复现

只消费旧封存 `holdout/joint_guard_1701627244` 的注册 selected-J 快照、对应 dense/diagnostics、protocol/helper 源及 `regression_context.json`。新父候选不在本 oracle 的评价范围。没有导入 torch、Isaac、runner 或原 projection helper。

选择在 **2026-10-05T13:15:15.685701+00:00** 落盘，当时 LP 调用数为0；实际求值于 **13:15:35.596270+00:00** 完成。选择依据仅为已有残差／回归上下文：4个回归病例相邻注册帧，6个跨阶段最大返回安全残差，6个严格零残差对照，2个剩余最大单行不可行 lower bound。没有按 LP 结果换样；这是有目的抽样，不能估计全体失败频率。

选择 SHA256：`6221504ba7101c60a6ab13e8345a04830f31bb758c451483f034c0a0381a156c`。

oracle 源 SHA256：`f5fa617bc3d4d500c27154a98b66c82db99df3a9acbfb3b3f641497ed83e8610`。

结果 JSON SHA256：`75ec9604c058f191ec072091c5e5cd8fd215897b4a0932ce2cfd08c06245f7b7`。JSON保存全部输入 SHA、冻结选择、18例 bounds/返回命令、可行 witness 或不可行对偶证书、重建检查和5个实际 CPU 测试结果。读前、读后 SHA一致；旧 helper 与实际 backstop SHA均符合原 protocol。SciPy **1.15.3**，NumPy **1.26.0**。

已有冻结结果可直接复算；`--freeze` 刻意拒绝覆盖已有文件。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/liyufeng/miniforge3/envs/safeduo/bin/python -B \
 /home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/astra_joint_oracle.py --evaluate
```

## 联合约束与实际执行的检查

26维增量同时包含两机器人原有的独立安全半空间和4臂 alpha 行。固定原 `p` 分配，没有改为重新分配跨机器人预算的全局约束：

`G_r u_r ≤ h_r`，`G_r = −J_r`；正 cross cap按原 `(1±p)/2` 分配，负 cap不分配。保留原 gate、valid、arm involvement、structural/contact masks，以及实际 `h = max(raw_h, 0.9 minGu)`。`backlog_aware=false` 的 debit 为0；预测 d_eff仍包含原 Jqd horizon及6项 pending target history。

bounds从原 softlimits减 issued target、`vmax×dt` 速度箱和 `q±.050rad` reference 重建；reference不可达时保留原最近端点单点恢复。alpha为 **沿 limited command 方向的进展上界**，不是下界：`dir_a·u_a ≤ alpha_a ||c_a||`。实际 gate、bounds、limited input、FIFO历史／队列、返回命令与 dense对应字段严格相同。

执行 **3006项** selected-snapshot 重建／绑定检查，无失败；最大跨精度算术差 **6.2995631489e−8**，小于预先固定的 `5e−7` 算术尺度。LP使用经重建验证的**实际保存 float32 G/h/alpha系数**转 binary64，不用重建舍入值替换实际约束。

每例分别运行完整集合和 safety+bounds集合。HiGHS primal/dual容差 `1e−9`；等价 L2行归一化后检查 witness，以及不可行 phase-I 的原始解、对偶下界、符号和 stationarity。13个可行 witness 的未归一化约束／bounds最大正残差均为0。5个不可行证书最大 stationarity误差 **1.3878e−17**，最大对偶间隙 **4.3368e−19**。`1e−8`仅为 witness/certificate 数值核验尺度；归一化 slack是增量空间的诊断量，不是物理距离门或安全许可。

5个有独立解析答案的 CPU fixture实际通过：可行区间、逐行可行但相互矛盾的安全行、安全与 alpha冲突、强制单点不可行、返回残差为正但集合仍可行。测试与真实快照结果分开记录。

## 18例实际结果

返回安全残差为 `max_F,U max(G u−h,0)`，单位m；不是 post物理几何违规计数。所有正数均保留。

| step/env | 联合集合 | 原返回安全残差 m | 说明 |
|---|---|---:|---|
| 480/21 | 可行 | 9.70893e−9 | 相邻回归帧；小于原 solver tol |
| 959/21 | 可行 | 8.42776e−11 | 相邻回归帧；小于原 solver tol |
| 480/49 | 可行 | 0 | 相邻回归帧 |
| 959/49 | 可行 | 0 | 相邻回归帧 |
| 66/23 | 不可行 | 8.46282e−4 | 逐行可行，但安全行联合冲突 |
| 71/3 | 可行 | 8.21191e−6 | 可行集上的原返回投影失败 |
| 75/40 | 可行 | 9.47844e−4 | 可行集上的原返回投影失败 |
| 180/18 | 不可行 | 1.51466e−3 | 有单行与bounds冲突；同时也有多行冲突证书 |
| 480/31 | 可行 | 7.77392e−5 | 可行集上的原返回投影失败 |
| 959/50 | 不可行 | 2.31539e−4 | 单行不可行gap约1.60402e−8m |
| 0/0 | 可行 | 0 | 零残差对照 |
| 60/1 | 可行 | 0 | 零残差对照 |
| 62/2 | 可行 | 0 | 零残差对照 |
| 66/0 | 可行 | 0 | 零残差对照 |
| 71/2 | 可行 | 0 | 零残差对照 |
| 75/0 | 可行 | 0 | 零残差对照 |
| 959/24 | 不可行 | 2.16409e−7 | 单行不可行gap约2.15786e−7m |
| 480/20 | 不可行 | 1.29224e−7 | 单行不可行gap约1.29224e−7m |

3个可行但返回失败的病例均没有 R19 bypass；原 solver报告与独立返回残差吻合。F/U passes是该 batch 调用的全局值，不能说这些 env各自独立耗尽30次。LP witness只证明存在解：零目标 LP产生的任意顶点不是建议交付的执行命令。

这里的5个不可行集合在去掉 alpha后仍不可行；**不能归因为 alpha独有冲突**。180/18虽有正的单行lower bound，其对偶证书还使用多条安全行和一个alpha行，不能把全部残差归给该单行。959/24、480/20的不可行 native安全gap本身小于原求解tol，依然记录数学不可行，不冒称显著物理风险幅度。

## 真实 R2 反例：单行 minGu 不保证联合可行

step66/env23，U 的原 selected cross行 **2296、2376** 均实际 relevant，两个 structural/contact mask均为false。每行各自在实际 reference/speed/soft bounds内有解：

| row | h m | minGu−h m |
|---|---:|---:|
| 2296 | −0.000514243089128 | −0.000057138147132 |
| 2376 | −0.001294897287153 | −0.000143877448032 |

将两行分别除以其 G范数，用非负权重 **0.9023776980744769、0.09762230192552313** 合成。合成不等式要求 `ḡ·u ≤ −0.001168712240280`，但实际 bounds内 `min(ḡ·u)=−0.000879895703608`。两者矛盾gap **0.000288816536671**；无需alpha即可证明这两行与bounds联合无解。JSON对偶证书保存原row身份、权重和bound multipliers；可从 SHA绑定的该帧 G/h复现。它不表明真实机械碰撞必然不可避免，只表明当前这组线性控制约束无共同增量。

另有单行机制：实际bounds使minGu为正时，`0.9 minGu < minGu`，原authority公式不会保证该行可行。这里保留事实，不自行改h或authority规则。

## 最小隔离建议与限制

最小可检验的 solver改进是：沿用全部原行／alpha／bounds／固定p／gate，对原返回不满足约束的情况增加**同时约束的可行性检查及最近 limited-command投影**；可行时要求输出 primal残差满足原门，失败必须显式记录，不能只凭逐行minGu或迭代变化小宣称成功。71/3、75/40、480/31是可用于软件回归的3个实际可行反例。

66/23用于验证不可行语义：仅增加迭代或改求解器不能创造共同解。若父候选使用不可避免 slack，须显式分开 alpha(rad)与安全(m)并保存正slack、hard bounds和失败状态；它改变了不可行时的执行选择，不等于保持硬安全可行。不能把最小slack输出、参考目标可达、零solver残差升级为物理安全成功。没有建议更改阈值、exemption、FIFO6、priority或actuator结构。

R1仍成立：raw proposal forecast相对reference-limited proposal没有保守包络保证。R2已由真实快照证实。集合只含原 selected且实际 relevant的行，不是9021全行或未来运动安全证明。

joint新失败 env21@893、env49@488 的失败当步没有保存J；所选480/959帧**不能解释或排除**失败当步不可行。env21@893最接近的row8925虽然 `selected_previous=true`，同时 `exempt_previous=true`；不得将“selected”写成“前步非豁免受约束”。env49@488的row8987前步selected且非exempt，但没有该步联合集合的LP结论。PD跟踪、pending target执行、几何线性化误差及动态mask仍是独立限制。

返回命令和真正 target增量分开重算：例如75/40 actual-target bound正残差 **1.4156103134e−7rad**、60/1 **2.2351741791e−8rad**，没有清零；18例均无R19 bypass，本任务未覆盖 repaired bypass执行语义。

本任务仅新增 `astra_joint_oracle.py`、`ASTRA_JOINT_ORACLE.json`、`ASTRA_JOINT_REVIEW.md`。旧namespace和父源均未写入。
