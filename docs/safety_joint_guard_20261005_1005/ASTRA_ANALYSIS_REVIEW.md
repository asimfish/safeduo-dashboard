# ASTRA：分析与原生相机源码预审

审读日期：2026-10-05。主要文件 SHA 读回时间：03:05:26 UTC（11:05:26 +08）。本报告只针对本轮 `analyze_v2.py`、冻结的 `visual_runner_v4.py` / `visual_registration_v4.json` 和当前 `verify_native_visual.py`；不修改任何控制源、冻结 helper、旧证据或面板。

**结论：源码预审已完成，结果验收仍待执行。** 已确认 FIFO/帧轴、returned command 与实际 target 增量的区分，以及 v4 保存 fresh native before/after 的实现。当前 analyzer/validator 仍有下述离线证据门缺口；不能把它们目前能检查的子集称为完整 admission、history 或 camera coverage 核验。这些发现涉及分析/验证层，无需由本侧线改冻结控制源，也不构成物理策略批准。

## 1. 本次实际执行及范围

实际执行：读取相关源码；对 `analyze_v2.py`、`visual_runner_v2/v3/v4.py`、`verify_native_visual.py`、`guard_runner_v4.py`、冻结 diagnostics/helper/tests 作 AST 解析或对应源码审读；计算下列文件 SHA；逐字比较 camera v2→v3→v4；读取有效安装路径 `safeduo_isaaclab` 的 camera 内参/pose 更新实现和当前 `DuoEnv` camera 配置。没有导入 Isaac/runner，没有运行模拟、GPU、analyzer、validator 或重复 CPU tests，没有读取尚未产生的相机图像来宣称验收。

v2→v3 唯一文本增量是保存 `env._joint_idx`；v3→v4 唯一增量是 state 字典中的 `full_braking_dmin_m`、`full_closing_m_s`、row identity 文件链接、pending actuator targets 和 sphere radii。两次增量均未改变槽、视角、阶段、物理/控制调用或目标。v4 源 SHA 与注册文件一致。这是源码差分结论，不是同次 native invariance 或跨次轨迹一致性的实测结论。

按最新用户状态，baseline development 39/64、十字段 exact PASS 为主线程报告，admission development 尚在运行，fresh numeric/camera 结果尚未产生；本报告不独立评分这些结果。注册量为 3×64×4=768 method windows、192 paired cases；相机计划为两个独立 replay，各 7 槽×3 步×9 视角=189 图，合计42组/378图。注册量不能当作已完成量。

## 2. 帧轴、FIFO6 与配对：源码逻辑成立的范围

- `guard_runner_v4.py:216` 保存 pre-step debt、actual FIFO pending 和另一个 pending project history；`full_forecast` 的 measured geometry/forecast 属于 pre-step。chunk 的 t 对应 initial（t=0）或 dense post[t−1]，不能直接与 post[t] 的 official margins 比较。
- `analyze_v2.py:62–69` 的 controller-before、debt 和 FIFO pre 重建符合该顺序。`pre_pending_actuator_targets` 轴为 **[960,6,64,26]**，队列轴 oldest→newest；第 i 个槽在 t 步为 controller[t−6+i]，负索引用 initial 填充。继承的 `inspect` 另核 delivered actuator target[t]=controller[t−6]，前6步为 initial。生产 `TargetDelayQueue.push` 为 append/popleft；没有 priority、重排或 rebasing。
- 控制增量为 controller[t]−controller[t−1]（t=0 用 initial）；实际关节运动为 q[t]−q[t−1]。首个非零 tape 命令 t=60 经 FIFO6 在 t=66 delivery 后形成该步 post 状态，因此 pressure exposure 切片 `[66:]` 的定义一致。first failure index 是零基索引，post 时间为 (t+1)dt。
- 按 command seed 配四 modes、逐位核 initial/tape/stratum，rescued/new/both/neither 与逐 case first failure 的源码公式正确；四因素交互仅为描述性差分。当前分析明确非 IID、非全26D覆盖、非硬件批准。
- `measured_d<0` 是已扣球半径的几何负裕度；`measured_d−dmin<0` 是低于制动边界。最新 analyzer 已分开这两项，official violation 仍严格 `<0`，deep 仍 `<−.005`，不以数值容差抹去近零负值。

## 3. 待闭合：实际 admission 集合与 cell 身份

**证据门缺口 A。** `forecast_audit` 当前只核 selected IDs 上界、唯一性及有效数量（`analyze_v2.py:46–51`）。没有读取/检查已保存的 `baseline_ids`，没有重建实际 union，也没有直接核每 mode 的512/1024 padding 宽度、`-1` sentinel、required row count 与无漏行。因此缺失一条危险 row、换入另一条合法 ID，只要数量不变，就可能通过当前检查。

最小离线收口：核全部 full-array 形状/identity、ID dtype/范围/唯一性/固定容量及 padding。A1 每 env/frame 用已保存 baseline 集合 B 核 `selected == B ∪ {forecast <= dmin+.010}`，并核 B 含 raw-critical 集合；核 required/count/forecast-added 统计一致。A0 核 selected 与已保存 baseline 一致、容量512。A1 selector 按 row ID 升序取完整集合，超1024 abort，无截断；这里的 ascending ID 是存储顺序，不是改变生产 priority。已保存 baseline 集合仍是 producer 记录，不能因此声称离线重新计算了全部原 actor32选择；原32选择/豁免不变还须绑定原生产源及可用快照。

**证据门缺口 B。** 当前 main 只要求12 jobs及12不同 PID，未要求 job ID 集合恰等于注册的 modes×3 seeds。PID 唯一不能排除 cell 重复/漏项。mode 直接从 job ID 解析，也未核各 cell `guard_metadata.json` 的实际 mode、A/E、capacity、FIFO6及 helper 源 SHA 与注册匹配。应逐 cell 核唯一注册 ID、command/initial bank 身份和 metadata，再计完成窗口。两 cell 协议“彼此相同”不等于都与注册相同。

源码中 v4 的 A1 入口直接捕获 `self.original_safety`，A0 保留原512 wrapper；此次未发现旧 v3 的512前门再次进入 A1。这里是源码路径复核，不是本轮实际曾运行 >512 rows 的证明。

## 4. 待闭合：full fields 与独立 project history

`forecast_audit` 确实设计为核30个连续32-step chunk、SHA、全9021 forecast finite、measured_d/dmin finite、forecast≤measured_d、最小值及 shadow bitset 计数。但当前仅 forecast 显式要求 [T,64,9021]；dmin/measured/exempt/bitset/identity 还需严格形状和类型门，避免 broadcast 或 identity 错位。应同时检查 mechanism 的 required/full-rows/guard-call frame 轴。

exempt 目前参与计数，dmin 目前参与 finite/阈值计算；这不是“独立重建全部原 dmin/豁免”的 oracle。可用 full pre 非豁免几何四桶最小值对齐前一帧 official margins；原 dmin/engage/exemption 来源及未改语义须按冻结生产源与可用 selected snapshots 陈述。不要把 producer 保存值与原门复算混为一项。

`analyze_v2.py:81` 返回 `separate_project_history_saved=True`，但代码从未检查 `pre_pending_project_history`。runner 确实另外保存了它（:225），应核该字段 [960,6,64,26]、finite 和顺序，与原 `PendingTargetHistory.reset/append` 语义重建；不能只凭 actual actuator FIFO 已核就替代 project history 核验。当前 config 两者预期可逐位比较，但它们是不同来源的两条记录。

full J 全9021逐步 finite 是 producer 执行的共同前门；全 J 未全程落盘。本轮只计划9个 selected-J快照，不能写成960步全9021 J/forecast公式均已被 offline 独立复算。

## 5. Returned 与 actual target 残差路径

源码把 diagnostics 安装在 reference 内层；E1 输入 clamp/bounds 在原 solver/R19 之前，外层 `checked_project` 记录最终返回命令。common finite checks 覆盖内外输入/输出/info以及 physics 前 integrated/delivered targets；四 modes 共用前门。有限门的实际 PASS/fault abort 属于随后运行证据，本报告没有重跑。

`guard_runner_v4.py:262–279` 的 target 路径使用真实 controller-after minus saved-before，复用同次原 inside cmd、rows、alpha/p/dt、实际 bounds/active/info，重算 target safety/alpha/bound residual。这正确区分 R19 返回增量、之后 soft-limit target clamp、实际 measured PD 运动；这些残差都不是 PD 物理可行性证明。

当前 analyzer 只聚合 producer 残差，没有验证9个快照的算术。后续有界独立核验可用 snapshot J、d/dmin、rel、authority 前后 h、cap/bounds，分别以真实 returned/target 增量核 `max(Gu−h,0)`、alpha 行及 bounds 残差，并核最坏行/individual minGu infeasibility。individual infeasibility 为零不证明联合约束可行；passes/hit-limit 为机器人批量属性，不能当成每 env 都失败。所有帧有限/统计与9快照算术重建应分别标 scope。

## 6. v4 fresh native / dense binding 的实际源码性质

v4 每 capture step 先 fresh getter clone 全64槽 q、qd、root transforms、root velocities，保存 before NPZ；每槽9图完成后再次 fresh getter，assert 全64相同并保存 after NPZ（:35–43、:131–138）。因此本轮已具备 **after 数组可被独立复算** 的设计，不再只有 producer 的 after assert。cached invariance 和 fresh invariance 是两组检查。

state 保存真实 `env._joint_idx` 和所有 joint names，validator 用这些索引取 native controlled q/qd，而非“前7/6关节”假设。q 对 dense post[t] 全64核；qd 在71/75对 next-pre[72/76] 全64核；959没有下一帧，仅 native before/after与选中 capture qd。源码表达了这一终态限制。建议验证层再核 indices 整数/唯一/非负/范围和对应 names，不能只核长度。

full physical classes 三类映射 eval 四桶的源码正确：cross→0、self 且首球属于F→1、self 且属于U→2、table→3；每桶只取 nonexempt raw d，并对本 replay official margins。半径已经在 raw d 中扣除，不能再次扣。初态和 tape exact 与 q/ee/exec/targets 等各字段跨 replay exact 分项报告；轨迹 FAIL 仍为 FAIL，照片只能绑定自己的 visual replay，不是数值 holdout 同轨迹照片。

## 7. 相机 validator 的剩余证据门及五平面范围

**证据门缺口 C。** verifier 当前只核21条 receipt/每 state 9 views的数量，没有核 `(env,step)` 唯一集合恰等于注册的7槽×3步，也没有核9个 view names 与9张唯一 image paths 的双射。重复记录可能通过数量门；主结果378张也为硬编码。应核注册集合、每 view/image路径唯一映射、实际 image/view/group计数，以及 source/registration/receipt/state/native/image SHA 闭合，再报告全覆盖。producer 的固定循环设计成立，但不能替代独立集合检查。

v4 新增 dmin/closing/queue/radii/row identity link 目前 **已保存，validator尚未验证**。`full_row_identity_file` 是文件名链接，不是逐 state 的 identity SHA。可在最终离线 receipt 绑定 identity SHA并核其 schema/9021 rows、sphere/pair indices与半径；71/75的 post full d/dmin/exempt应对下一 pre chunk72/76，post pending应对 next-pre queue，959可对终态 controller 序列重建 pending。closing 若无全程独立记录，应仅按可用数据 scope；不能笼统宣称全部新增字段已核验。

实际 camera pose 从 USD `ComputeLocalToWorldTransform` 获取；SDK stale pos/quat另存而不改标签。有效 SDK `_update_buffers_impl` 受 `update_latest_camera_pose` 开关控制。内参 K 在 producer 仍来自 SDK缓存，validator另用 USD optics离线计算比对；这需要实际记录通过后才可称 bounded calibration通过。

当前 `DuoEnv` 为1280×720、focal18、默认 aperture、零offset；有效 SDK 在创建时按宽高比补 vertical aperture，并计算 square pixels（fy=fx、principal point在中心）。validator的 USD optics公式在这个零offset/匹配宽高比条件下与之相容，不是任意offset/纵向aperture的通用 renderer标定：安装SDK明确不使用这两项；现有非零offset公式不应推广。

producer与validator的球体检查为 **左右上下四侧面 + `depth−radius>0`** 五平面。它没有验证实际 near/far clipping planes；当前 camera配置 clipping=(.1,60)，第五项仍用0而非实际near .1。故应写“五平面/正深度的代表球体容纳”，不能把 state 的“all view planes”扩称完整实际六面视锥。也不证明完整机器人轮廓、遮挡、像素识别、连续接触或搬运安全。遮挡仍须真实图像抽看；本阶段相机未启动，不给照片有效性 PASS。

## 8. 版本绑定与最终界限

| 文件 | 审读 SHA-256 |
|---|---|
| analyze_v2.py（本次未冻结的审读版本） | bf0809a812d9f0b8652160f583331128822547df395125ebb55ec662cd646055 |
| visual_runner_v4.py | 4df89bb9c9f5b8f3242c6c4a8aa584c5b7451796661c8c4d068915006cf8d174 |
| visual_registration_v4.json | 4f8a73bb4b471756d4fee54f6d16f83863ede013a70c806cff7a2e24ccf5bb89 |
| verify_native_visual.py（当前审读版本） | 5564f6f746bb35fa5ac5a4ff89a10d77ba878d49ec150fed8f62222b07cdb845 |
| guard_runner_v4.py | d52598b63dd82575337bbd6b58ec30030718f4753594032e262322d8fe728789 |
| full_finite_guard.py | e4644512a380e22d06984401d7ad5081219c3b60b4172f1ea01eb5d3902373d6 |
| reference_envelope.py | 7cf0ccccce2f1a81d0ea657797a71eb1c5cb8b852aa94e84767a2dfca1ab9c47 |
| projection_diagnostics.py（冻结17233 bytes） | 1e2c371d7a8d4a1e7c5c943e585225141412e63b0a70ea98989534bc600a1869 |
| test_projection_diagnostics.py（冻结13759 bytes） | dda9bc503b0aed91216e82d7b9003a1bda13814c45b3d0f45a5946cd2f185bcf |
| holdout_plan_v2.json | 0cfc248bb5d65d6a628a30122c5bf95ff00a4da3f46a4220f595ab2675a0bf0a |

冻结 diagnostics/tests SHA 与此前核定版本一致，本次未修改。后续 analyzer/validator若修订，应保存版本和最终执行 SHA；本报告不预先验收未来修订。注册参数、原始豁免/FIFO6/actor32、容量 abort/nodrop 和原严格负裕度门不能由解释或分析纠正替代。

本侧线当前交付是源码发现与有界验收建议；实测完整性、768窗口配对结局、native/image校验和轨迹差异须待实际完成后独立审计。**production_promoted=false；hardware_approved=false；物理安全策略未批准。**
