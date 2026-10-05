# Astra：v4 capacity 路径有界独立复核

2026-10-05 02:43 UTC（10:43 +08）读取快照。**PASS_BOUNDED_CAPACITY_SOURCE_REVIEW：v4源码已解除先前A=1继承512前门的问题。** 这是source层结论；建议的CPU容量seam尚未由我执行，实际fault/development/holdout仍需各自receipt。未运行模拟/GPU，未修改runner、冻结helper/tests、生产或旧报告；唯一新增本文件。

## 实际调用路径

`WideTrace.start()`先将wrapper之前的production选择函数保存在 `self.original_safety`；system0路径另将 `env.safety_dist_out` 包成raw-critical union加512 hard cap。v4:34现在按A因素明确取不同入口：

| 模式 | original_selected入口 | 最终选择预算 |
|---|---|---:|
| baseline_guard / envelope_guard（A=0） | 继承的512 wrapper | 原raw-critical dynamic512 |
| admission_guard / joint_guard（A=1） | wrapper前 `self.original_safety` | 原priority union raw-critical union注册raw-forecast，1024 |

v4:71调用上述入口；:90–91构造baseline/mask；:111仅A=1执行 `admission.select(...capacity=1024)`。原select按mask每env所需行数的批内最大值分配，超过capacity即abort、不取top1024继续。A=0仍先经过原512门，不因reference打开而绕过该门。A=1不再调用它；reference在选择完成后的project阶段，不能重新引入该旧容量入口。

v4:128的 `baseline_ids` 在A=1由同帧production选择+raw-critical union单独生成，使用同一个1024预算，**只用于记录**；控制仍返回:111生成的result。完整finite forecast以current d初始化再取minimum，故baseline mask是actual mask的子集；actual select若已在1024内成功，baseline所需宽度不会更大。这层记录不会重新制造512门，也不改变控制结果。

注意A=1的baseline_ids是同帧基准集合，可能超过512。它不是A=0独立轨迹，也不能证明此状态下A=0成功执行了513行。A=0字段则来自实际512 wrapper。报告需保留这个区别。

## 差异与接口范围

我对旧guard_runner.py和v4做了完整文本diff：只有4处差异——A1入口切换、baseline_ids记录扩大为raw-critical union、schema v3→v4、source manifest绑定v4文件名。共同full finite检查、原measured geometry/exemptions、raw-proposal因子、shadow诊断、reference/inner+outer gates、target/FIFO记录和projection_diagnostics接口没有其他文本改动。

registry_v2脚本实际将argv指向guard_runner_v4.py；development_v2仍是旧60317411两方法/cuda1验证，holdout仍按DESIGN的三对banks/seeds四模式生成12jobs。execute_campaigns_v2先要求development complete及注册字段的strict equivalence PASS，再启动holdout。这里仅审脚本路径/顺序，没有执行注册、独立审计279源或读取新结局；旧v3 partial/暂停及0holdout是父线程报告，不在本任务重计。

## 建议一个实际source seam oracle

使用CPU合成fixture测试**实际v4路由语句与safety选择路径**，随后调用真实 `union_mask/select`。可AST提取v4对应代码，避免导入Kit/启动env；几何/provider、production selection与文件receipt在边界处提供CPU double。不能把v4的if/else手抄进测试，否则入口回归可能不被发现；也不能仅测试 `select(capacity=1024)`。

fixture按source固定维度[N=64,P=9021]，全dmin=0.030m，默认d=1m：

1. 512个raw-critical IDs为0..511，d=0.035m。env0的原production选择另外保留ID7000；其余env选择ID0。原选择中可再重复ID0以检查union去重，最终计数不应增加。
2. env0另有ID8000 forecast-only，预测d=0.035m，但原measured d仍1m。故env0 baseline集合恰为513，actual集合514；其余env为512。
3. 若要连真实full_forecast也经过seam，可设q/issued/6pending=0、box0.025、rawcmd仅F_L第一关节+0.025，full J只在env0/ID8000该列为−38.6，其余为0。它产生约0.035m的finite线性预测，无需模拟。这个J是容量fixture，不声称真实几何样本或物理可达性。

强oracle应同时断言：

- A=1两模式都只调用production入口，继承512 wrapper调用次数为0；保留exact IDs `{0..511,7000,8000}`、count514、forecast_added_count=1，不截断/丢原priority ID。
- 同帧baseline_ids恰保留 `{0..511,7000}`（513）；诊断select前后控制result不变。A=0两模式仍调用真实继承512 gate并在513需求上拒绝，不能通过新1024路径。
- env0 selected宽度514时，其余env的尾部padding是idx−1、mask=false；各有效行measured d/dmin/class/exemption从原full值逐位保留。ID8000不能把其1m原观测改成0.035m预测。
- 边界变体：forecast-only改为IDs8000..8510，512+1+511=1024必须完整保留；再增加ID8511得1025，必须在projection/physics之前抛capacity错误，不能掉1行继续。

这个seam直接区分旧错误路径（env0的513 raw-critical基准先被512门拒绝）与新正确路径（514/1024成功、1025拒绝）。单次16s development若未达到512边界，只证明该轨迹上的非干预，不能替代该容量oracle。父线程实现/执行测试后应绑定其source和日志SHA；本报告没有预填测试PASS。

## 证据绑定与最终边界

已执行：读取调用链、完整v3→v4 diff、CPU AST语法解析及SHA；未运行新的CPU容量测试，也未重复已通过的12项helper测试。

| 文件 | SHA256 |
|---|---|
| guard_runner_v4.py，18620 bytes | `d52598b63dd82575337bbd6b58ec30030718f4753594032e262322d8fe728789` |
| 旧guard_runner.py，保留 | `a7a6de100a64983c6690bcd1ffba96fa8e2c55e4519c204d9495eed3fcf52f78` |
| 冻结projection_diagnostics.py，17233 bytes | `1e2c371d7a8d4a1e7c5c943e585225141412e63b0a70ea98989534bc600a1869` |
| 冻结test_projection_diagnostics.py | `dda9bc503b0aed91216e82d7b9003a1bda13814c45b3d0f45a5946cd2f185bcf` |
| full_finite_guard.py | `e4644512a380e22d06984401d7ad5081219c3b60b4172f1ea01eb5d3902373d6` |
| reference_envelope.py | `7cf0ccccce2f1a81d0ea657797a71eb1c5cb8b852aa94e84767a2dfca1ab9c47` |
| register_campaigns_v2.py | `2e4f98885fc66240e5c5213c235ed554db5f37001481ed88327cd2bea51040cc` |
| execute_campaigns_v2.py | `89359f0c45f24b66f6a39709a320f95f58d5aebba83507a22ddbfb6b462e9e63` |

此前报告中对v3预算的阻断结论保留，不能改写成当时已通过；本文件只解除v4读取快照中该source缺陷。未发现新的容量路径阻断；全finite live fault、development equivalence、完整任务完成和物理安全均未由本有界审查验收，production/hardware/safety approval继续false。
