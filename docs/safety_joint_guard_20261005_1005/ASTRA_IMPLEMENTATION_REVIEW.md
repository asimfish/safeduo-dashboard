# Astra：passive projection diagnostics 实施与集成审查

审查快照：2026-10-05 02:37 UTC（10:37 +08）。**HELPER_READY_FOR_FREEZE：API定稿，12项独立CPU测试通过。** projection_diagnostics.py为17233 bytes、SHA `1e2c371d7a8d4a1e7c5c943e585225141412e63b0a70ea98989534bc600a1869`；test_projection_diagnostics.py也停止修改。代码evaluation-only，production/hardware/safety approval均false。这里只新增诊断模块、测试及本报告，没有编辑父runner、reference/full guard、生产代码或旧封存证据，没有启动模拟/GPU。

父runner的API/finite gate集成已匹配；但本次实际读取的 `a7a6de10…` 仍有原512 wrapper在A=1先行abort的预算合同问题，详见第5节。helper可冻结不等于该合同已验收。live fault、16s development及holdout的执行结果不在本报告验收范围。

## 1. 最终接口与固定调用顺序

```python
from functools import partial
from projection_diagnostics import install, checked_project, summary_record

actual = env._backstop.project
env._backstop.project = partial(checked_project, actual)  # 四格共同inner gate
sink = {}
restore = install(env, sink, snapshot=lambda: t in diagnostic_steps)
# E=1才安装 reference；E=0保留原输入语义，不能装box_only。
# reference_envelope.install(env, 'envelope_050')
outer = env._backstop.project
env._backstop.project = partial(checked_project, outer, diagnostic_sink=sink)
```

`install(env, sink, *, snapshot=False)` 支持MutableMapping或 `sink(record)` callback，返回安装前的callable；不创建文件、不触碰queue/state/config。dict每次调用清空并填入本次record，callback每次收到一份record。所有record值均为detached tensor clone，消费者修改它们不能改变cmd、rows、solver result/info。wrapper返回原project的**同一个三元组/对象**，不修改输入、输出、active或info。

`snapshot` 为bool或零参数predicate，每次调用求值一次。默认仅固定宽度汇总；`snapshot_*` 仅在选定帧存在。`summary_record(record)`筛除它们，适合全程stack；完整record适合父线程9个预注册snapshot。sink要在下一次project前保存。callback模式若要补外层字段，父线程须保存callback收到的dict并在外层操作该dict；`diagnostic_sink=`本身接收MutableMapping。

`checked_project(project, cmd, rows, alpha, p, dt, *, diagnostic_sink=None, **kwargs)`是独立共同finite gate，递归检查实际call seam的cmd/geometry/J/alpha/p/dt/kwargs（含reference生成的bounds、limited input），以及返回exec、active、所有numeric info leaves；异常直接上抛，不替换数值、不放松半空间。两层gate分别保护reference内侧真正solver输入/输出和外侧最终结果。全9021行、q/qd/limits、box配置和实际FIFO身份仍由父full guard/receipt负责，不能将selected-seam检查改称full-scene检查。

外层 `checked_project(...diagnostic_sink=sink)` 补记 `external_raw_cmd`、`outer_returned_cmd/outer_active`、真实 `governor_changed`及availability。内层observer是在reference写入该info之前执行，不能自行猜测它：默认availability=false、placeholder=0。缺失/不可见不等于已证明没有governor改写。

## 2. 诊断计算、字段与单位

安全预算复用原 `info['cap']`，不重算cap。rel严格遵循实际cfg的engage、structural/conditional gate、row.valid及arm involvement。为保持严格engage门的浮点比较，不通过 `cap/(gamma*dt)+dmin` 反推d_eff；仅在engage或详细snapshot需要时，按原source运算顺序重复d_eff。这是只读计算，没有新预测策略或admission。

每robot重建G=−J、cross正cap的p预算/负cap全额预算、可选backlog debit、原authority前后h、原 `_alpha_rows` 半空间、实际速度/target bounds。关节bounds/command为rad；G与J为m/rad，安全Gu−h及cap为m，alpha/bound residual为rad。不同性质的residual不能直接当同单位比较。

默认固定宽度字段包括：

- `project_input_cmd`（reference之后真正进solver的cmd）、`returned_cmd`、26维bounds、alpha/p、原active/bypass、governor availability；外层再加raw cmd/真实gov变化。
- 各robot原 `original_safety_residual_*`、原passes、returned安全/alpha/bound残差、alpha逐臂残差、相关行数量、singleton/zero-outside-bounds、individual infeasibility下界和行数、authority变化行数。
- 最坏selected位置、该行G/d/dmin/class/cap/raw_h/debit/authority前后h/minGu。位置是**当前selected矩阵列号，不是global pair ID**；父线程通过同次selected IDs映射。无相关行以位置−1/汇总0表示，不伪造row0证据。

详细 `snapshot_*` 保存selected d/dmin/class/valid/armmask/J/G、cap/d_eff/row gate、各robot rel/raw_h/debit/authority前后h/minGu，以及alpha G/h/rel。还保存project kwargs里的backlog、past_backlogs（[N,K,26]）、qd、struct/contact masks及availability。past_backlogs是solver收到的**target−pre_q**，不是adapter pending绝对target；父线程的两条完整target历史应分别保留、再按pre-q对齐。

`passes_*`是生产solver按robot对整批环境使用的循环次数，扩成[N]便于落盘；`passes_hit_limit_*`不意味着批内每个env都独立不收敛。individual infeasibility>0能说明至少一个相关半空间与bounds不可兼容；等于0不能证明多行/alpha交集共同可行。所有数值残差保留，不用新容差把小残差抹成0；active/governor布尔阈值仍是原source定义。

## 3. project-returned 与最终target增量范围

`returned_*`重算使用原project返回的exec，**包括R19覆写之后**；原info residual生成于覆写之前，两者可以不同。它仍不是后续soft-limit clamp后的真实controller target增量，更不是PD关节位移/加速度/物理避碰保证。

独立函数 `projection_record(backstop, cmd, rows, alpha, p, dt, kwargs, result, snapshot=False)` 可供父线程对保存的同次pre-state context重算。若父线程将result里的cmd换为真正 `target_after−target_before`，必须将产生的残差另命名为target residual，保留原returned字段。父runner目前正按这个方式只提取target_safety/alpha/bound残差；没有将新target重基到q、取消pending命令或改solver。这个target诊断仍是同次pre-J/半空间上的solver意义，不是下一物理帧安全证明。

## 4. 已实际执行的CPU验证

我的独立命令（无GPU、无Kit、禁止pycache写入）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/home/liyufeng/safeduo/src:/home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005 \
/home/liyufeng/miniforge3/envs/safeduo/bin/python -B -m unittest -v test_projection_diagnostics
```

最终12/12 PASS，exit0，测试执行0.045s。测试直接调用冻结生产 `VelocityDamperBackstop` 的CPU实现，验证：

1. 包装前后exec/active/info逐位相同，原inputs不变，sink tensor无alias，以及原返回对象identity不变。
2. 真实R19：原solver residual=0而returned安全residual=0.020m，正确记录覆写差异。
3. 真实reference+inner/outer gates：R2 singleton delta0.025rad、authority后h0.0225m、returned安全残差0.0025m、alpha残差0.025rad及外层governor标记；不修复原不可行集合。
4. strict engage边界、永久/条件结构豁免、struct距离gate、负cap留行和padding屏蔽，以及可独立算出的d_eff。
5. stored/backlog最大debit与authority数值oracle、cross正cap按p分摊/负cap全额预算。
6. Caller提供不同target delta时其残差与原returned残差分开，且原返回值不变。
7. 无相关行的−1位置、snapshot predicate/固定summary schema，以及cmd/J/bounds/alpha/p输入故障在原project调用前被拒绝；坏exec、nested info、active在返回physics前被拒绝。

首轮fixture使用float64碰到生产临时buffer的float32限定；随后仅将测试fixture匹配实际float32。padding行的d_eff oracle也修正为负值并保持其rel=false，未改生产行为、残差算法或验收阈值。最终测试范围不包含GPU性能、跨CUDA bit equality、真实9021 provider fault、960步非干预或holdout结果。

父 `projection_integration_gate.log`另记录同12项PASS（0.047s），SHA `43c1c39d69179312ef4e5345ef7695d2addb67627d7284f1071099cad5b2da08`；父5项full guard日志PASS的SHA为 `b8a2b09f445ac39230a70fb88e4e762cc4710aca854b8954de7f9eb4ed8fadce`。我读取这两份日志，没有再次执行父的整套gate或其模拟。

## 5. 实际父runner集成复核与剩余合同问题

本次读取guard_runner.py（18480 bytes，SHA `a7a6de100a64983c6690bcd1ffba96fa8e2c55e4519c204d9495eed3fcf52f78`）确认：共同inner checked_project → mapping diagnostics/9帧predicate → E=1 reference →共同outer checked_project；有限性覆盖actual limited input/bounds、returned active/info，且在原pre-physics返回前核integrated issued及delivered actuator target。post-step另保存真正target delta残差。E=0不装reference、不改变project输入。父代码9帧snapshot可含全部snapshot_*，写全程NPZ的 `all(k in record and same shape)`避免首帧可选字段导致KeyError；snapshot-only字段不会被该条件当成全960帧summary保存。

全forecast chunks现在同时保存measured d、dmin/exempt、actual/baseline selected IDs和shadow-missed bitset。full_finite_guard已增加positive finite box与exact six pending长度检查，并仍在min之前检查每个displacement/linear delta/prediction。full9021 J只有producer检查；完整selected J/h/rel只在9帧保存，不能宣称独立离线重算全部960帧9021行预测。shadow missed在A=0也是相对假设raw-admission union，不是该baseline实际selection的全部遗漏；A=1才对应实际raw-admission集合。该诊断未加入控制mask，R1非保守局限保留。

**尚存预算合同问题（已即时通知父线程）：** `super().start()`调用的WideTrace在system0下先安装raw-critical union+512 hard cap；guard_runner:34捕获该wrapper，:71在A=1也先调用它。因而原selected+raw-critical>512但总union≤1024时，1024 select之前就abort。测试12项只证明diagnostic/reference seam，不覆盖这个继承容量边界；16s development如果没达到该边界也不能排除它。

若注册A=1意为只按1024总union预算，则需父线程在新version中使用wrapper之前的original选择，再做注册union1024（A=0保留原512）。若有意共同保留512前门，则必须明确冻结该额外abort合同，不能称仅1024预算。当前source与原DESIGN尚未证明这一点；**helper/API freeze-ready，父runner预算验收仍有条件阻断**。本报告未替父线程修源码、改注册或终止进程。

## 6. SHA与待运行证据边界

| 文件 | SHA256 |
|---|---|
| projection_diagnostics.py，17233 bytes，冻结 | `1e2c371d7a8d4a1e7c5c943e585225141412e63b0a70ea98989534bc600a1869` |
| test_projection_diagnostics.py，13759 bytes，冻结 | `dda9bc503b0aed91216e82d7b9003a1bda13814c45b3d0f45a5946cd2f185bcf` |
| guard_runner.py，本次读取快照 | `a7a6de100a64983c6690bcd1ffba96fa8e2c55e4519c204d9495eed3fcf52f78` |
| full_finite_guard.py | `e4644512a380e22d06984401d7ad5081219c3b60b4172f1ea01eb5d3902373d6` |
| reference_envelope.py，旧helper逐字节复制 | `7cf0ccccce2f1a81d0ea657797a71eb1c5cb8b852aa94e84767a2dfca1ab9c47` |
| 生产backstop.py，未修改 | `82a0f0e5ddbbe1b9eecbdbbd249a9354b50d78ab6e121473286e8dbb28f19afa` |
| 继承wide_runner_feasible.py，未修改 | `1b039bf6df0e9b7a4abe3f0461628a42db8c6c60ef2d75fa58c8e00e30662ebb` |

已对这些Python文件执行AST解析。父方报告3banks完成且无旧bank重复，但本任务未独立重算bank去重；live fault、两cell development和12 holdout均须以各自actual source/receipt另验，不能由本报告预填PASS。旧v2 prefix未修改。现在停止源码/测试修改，仅交付本实施审查；无需扩展CPU测试、实验或旧证据审计。
