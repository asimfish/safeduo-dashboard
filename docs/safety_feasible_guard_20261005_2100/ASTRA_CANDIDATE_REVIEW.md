# ASTRA：joint_repair 的有界源与 CPU 快照审查

**Verdict: Conditional pass（限定软件诊断范围）；physical strategy approval仍 BLOCKED。** 没有发现阻断本次隔离实验的已验证软件缺陷。以下条件必须保留在分析／交付语义中：正slack／正残差不称安全；fallback不称最近命令解；post-repair的 `original_*` 字段不称原 Dykstra出口。没有评价新development或holdout物理效果。

## Dimensions

| 维度 | 状态与依据 |
|---|---|
| 功能 | Pass（有界）：实际4个旧快照经过冻结 candidate纯函数及 recorded-exit wrapper seam；3个可行返回失败被修复，1个联合不可行明确保留正slack。 |
| 正确性／可靠性 | Pass（有界）：独立 residual/bounds重算、不可行对偶下界、float32 wrapper出口、原输入未变、QP失败fallback均实际核验；没有真实新轨迹结论。 |
| 架构 | Pass（限定evaluation）：在原project后、observer前局部包装；reference在外层；没有queue／target／physics写入或preemption。 |
| API／证据语义 | Risk：R19mask和 `original_*` 的历史含义会改变，详见下文；已有字段不能支持未保存的原solver内部向量重建。 |
| 可维护性 | Pass（范围内）：复用SHA一致的 projection_record构造实际行，未复制gamma/gate/authority公式。 |
| 性能 | Unable to determine：没有实测实时deadline或最坏wall latency；这次CPU检查不能批准100ms实时执行架构。 |

## Blocking findings

**无本次已验证的软件blocker。** full物理效果、全窗口finite/FIFO及original strict margins尚不属于本报告已执行证据。原几何门、FIFO6、candidate/hardware准入未获批准。

## 实际执行与绑定

仅运行自己的 `astra_candidate_tests.py`，没有重跑父6项测试作为验收。读取父tests确认其断言范围，独立补核实际旧快照及不同边界。进程 `CUDA_VISIBLE_DEVICES=''`、`torch.cuda.is_available()==False`；没有AppLauncher、simulate或物理任务。Torch **2.7.0+cu128**仅用CPU，OSQP **0.6.7.post3**，SciPy见独立oracle。

候选源 `joint_repair.py` SHA256：`375a7f42fb37e27e95dd31fed88d19cb63ce4b70ce4c6e11df4a18a3c6efd299`；runner：`3168d12d6225fa6f5ee82fb9e0ebac6d9f69cd5e7d845569651484e441cd2aa5`。development及3个holdout计划逐项绑定这两个SHA及reference/diagnostics/fullfinite源；读后SHA未变。计划时间／SHA写入测试log；未借此宣称新物理任务已完成或重新验证银行。

自己的实际执行源 SHA256：`f5db40968ae072218824e95477aabe66130377c208b2b15e9f5ee1cbf0175761`。

完整执行结果 `astra_candidate_tests.log` SHA256：`ae3a50fbf2b1622d49171ae5b44d07c2e1f2d78740641d004cd68f4bdb22d0d1`，状态 **PASS_BOUNDED_CANDIDATE_CPU_CHECKS**。log保存全部source/input SHA、4个captured case输出和各边界实际结果。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/liyufeng/miniforge3/envs/safeduo/bin/python -B \
 /home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/astra_candidate_tests.py
```

4个case沿用先前18例的**预LP冻结选择**，没有按候选效果换样；另核同清单内480/49零残差出口。wrapper seam中的原project出口是保存的真实返回张量，由CPU shim重放，不冒称重跑原CUDA Dykstra或 physics。synthetic R19、alpha及QP failure fixtures独立标记。

## LP/QP及返回路径

[joint_repair.py:21](/home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/joint_repair.py:21) 每robot联合解其所有实际relevant安全行与两臂alpha行；固定原p，没有跨机器人重新分配预算。bounds始终hard。先最小化common alpha slack（rad），再固定该alpha层最小化common safety slack（m），最后OSQP最小化 `1/2||u−limited_cmd||²`。单位没有相加。

这不是“优先最小化安全违约”的字典序。实际合成例 `x≤0`（alpha）、`x≥.02`（safety）、`x∈[−.025,.025]`：alpha最小slack0，条件安全最小slack **.01999998m**，尽管 safety+bounds单独有解。强制reference单点 `x=.025` 时，另一个实际fixture得到 unavoidable alpha **.025rad**、safety **.0025m**。这符合注册的alpha优先设计，不能描述成无条件全局最小安全slack。

[joint_repair.py:35](/home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/joint_repair.py:35) 的 `SOLVE_ALLOWANCE=2e−8`分别加入alpha(rad)和safety(m)的LP/QP上界；出口再允许原slack＋`CHECK_TOL=2e−7`。这是数值准则，不是原始半空间严格零残差证明。下表三个可行例都实际保留约2e−8m正残差；没有清零。bounds经float64 clip和float32 wrapper后，本次所测返回出口均严格在实际bounds内；未来真正target增量仍须另审。

OSQP `solved/solved inaccurate`均需有限x且独立残差检查；超出允许残差会切回LP点，LP点仍需检查，否则abort。实际强制QP返回NaN／失败的fixture走到LP fallback，重算安全/alpha/bounds均为0。没有测试每一种OSQP退出状态，也未独立证明所有成功QP的最优性／dual KKT。

## 4个实际快照对照

安全残差为线性 `max(Gu−h,0)`，m；不是物理负几何端点。

| step/env/robot | 原保存系数下返回安全残差 | candidate double残差 | CPU wrapper float32残差 | safety min slack | QP fallback |
|---|---:|---:|---:|---:|---|
| 71/3/F | 8.21190510e−6 | 2.00000000e−8 | 2.02380761e−8 | 0 | false |
| 75/40/U | 9.47844379e−4 | 2.00000000e−8 | 1.95577741e−8 | 0 | false |
| 480/31/U | 7.77392062e−5 | 2.00000000e−8 | 1.98706402e−8 | 0 | false |
| 66/23/U | 8.46282381e−4 | 1.48156133e−4 | 1.48156076e−4 | 1.48156133e−4 | **true** |

前3例先由独立联合oracle证明完整硬集合可行；candidate双精度返回alpha/bounds残差0，且两种slack均0。CPU wrapper实际通过输入／rows/J/alpha/p未改变检查，另一robot未触发修复。

66/23使用独立oracle已有的2296/2376两行对偶证书，将归一化权重转换到native common-metre slack，得下界 **0.00014815613294860895m**；candidate返回slack／primal残差达到该下界（差<1e−12）。这独立证明该快照不能硬满足所有原行。实际QP fallback使返回的是LP可行点；此时只能声称条件minimax层满足，**不能声称最终命令离limited input最近**，其额外运动／PD后果待真实窗口计量。

480/49的保存零残差出口经candidate wrapper仍逐位相同，两个 `repair_applied` 均0。这里的“exact pass-through”只指返回命令，info新增repair字段不可能与旧info字节一致。

## R19、trigger与诊断字段的实际语义

[joint_repair.py:71](/home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/joint_repair.py:71) trigger为原最终返回 safety/alpha/bound残差中任何一个 **严格大于1e−6**，包括R19覆写后的残差。没有触发的微小正残差／不可行gap仍原样保留；不称“所有不可行均被修复”。

实际合成R19检查：原旁通增量 **.009999999776rad**，修复后 **1.999999988e−8rad**；`bypass_arm=True`仍保留，`active=True`因改写而产生。因此mask表示原请求旁通，不再证明实际最终逐位直通。这是候选已明确接受的行为变化，不是R19架构保持的证据；4个真实捕获病例均无bypass，未证明真实旁通场景的效果。

observer位于repair之外。[joint_repair.py:95](/home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/joint_repair.py:95) 只要batch任意robot/env触发，便把整个F/U `info['residual_*']`向量更新为post-repair returned残差。于是 observer的 `original_safety_residual_*`在该路径是post-repair，`original_active`也包含repair的active改写；passes仍是原Dykstra的batch-global passes。**不能把这些字段一并当原solver状态。** CPU seam已核post `original_safety_residual`等于post returned残差。

`repair_pre_safety_residual_m_*`保存应用repair的robot/env之pre-return安全残差；未应用时默认0，不是完整全env原Dykstra残差序列。字段没有保存pre-alpha/pre-bound，也没有保存R19前的原内部solver向量；任何未来按trigger类型或pre-R19解归因都需声明该证据缺口。runner实际把repair字段加入固定宽summary，target-after-minus-before另作target残差，与 returned层分开。

## Non-blocking improvements（最多3项）

1. 最终分析按 `repair_applied` 分组读取pre-safety，解释 `original_*` 的路径依赖；不能用默认0补成未修复前残差。需要区分alpha/bounds trigger时，另一个版本再注册保存pre-alpha/pre-bound，勿改本轮冻结源。
2. 报告QP fallback次数及其运动代价；fallback时撤销“最近命令”结论。本次真实反例已出现fallback，应保留该字段参与结果解释。
3. 只把minimum-slack字段解释为固定bounds、原行、alpha优先且有数值allowance的条件最小量；保留原始／实际target／物理端点的所有正残差与strict负值。

## Minimum required repair／未执行门

本次没有要求修改冻结candidate；上面的报告语义是有界条件通过的条件。实际full窗口仍需独立验证：finite abort／capacity1024不丢行、FIFO6实际序列、原class严格负margin／deep、动作与movement代价、对应输入bank/tape及所有退出。当前CPU/source证据不能替代这些门，也不支持改变原threshold、exemption或priority。

本侧审仅新增 `ASTRA_CANDIDATE_REVIEW.md`、`astra_candidate_tests.py`、`astra_candidate_tests.log`；先前oracle三文件保持原bytes，父源／计划及旧封存空间未写入。R1 rawproposal不保守、R2联合不可行以及动态prior exemption限制继续保留。
