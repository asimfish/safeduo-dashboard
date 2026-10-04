# Astra bounded observer / empirical evidence review

审查对象：`safety_mechanism_20261005_causal_obs`。本轮仅新增此文件；CPU 读取原始 NPZ/JSON、源码和 PNG，未启动模拟、GPU、Kit，未修改已有设计审查、生产源、冻结 v1、协议或面板。

**有界证据结论：PASS_BOUNDED_OBSERVER_EVIDENCE。策略结论：BLOCKED_SAFETY_STRATEGY；production_candidate_promoted=false，hardware_approved=false。** 该结论接受下面明确限定的观测交付，不接受全 holdout 完成、唯一根因、连续时间安全或硬件安全主张。

## 1. 审计时间与窗口分母

独立计算读取 `results.json` 的时间为 **2026-10-04T17:05:03.292602+00:00**（北京时间 2026-10-05 01:05:03），该快照自身更新时间为 **2026-10-04T17:04:34.510745+00:00**，SHA256 见 §6。计算完成 2026-10-04T17:05:10.941075+00:00。更早读取时只有首对完成；计算时主线程已更新到两对完成：

| 范围 | 本次读取状态 / 独立核验范围 |
| --- | --- |
| holdout 计划 | 3 command seeds × baseline/candidate ×64 =384 方法窗口，对应192配对外部命令窗口 |
| 快照完成状态 |256方法窗口完成、128 pending、0 invalid；尚非384全部完成 |
| 本次直接重算的 holdout |仅 seed1930442907 的 baseline/candidate 各64；第二对只读取完成元数据，未在本次复算 |
| finite-guard diagnostic |120步 ×64环境，执行旧960步 development tape 的前缀；新增完整16s窗口计数为0 |
| visual development v2 |另一次960步 ×64环境 replay，12 camera groups /84 PNG；不是 holdout，也不是 numerical trajectory 的图像替身 |

所有完成数均绑定上述读取快照，不作为此后运行进度的声明。复用既有 stable banks、使用新 command tapes，未增加初态域覆盖；配对/跨环境相关样本不作 IID 泛化置信区间。

## 2. finite guard 与有效前缀

直接审读 `guarded_diagnostic.py`、冻结 `mechanism_runner.py` 和实际 registration/metadata。调用顺序支持以下限定主张：

- safety wrapper 在调用 v1 admission 前检查 q、qd、issued targets、raw proposal command 和实际 queue.pending 中每格 target。
- provider wrapper 对返回的 F/U Jacobian 全张量检查 finite；v1 先请求全部9021行 Jacobian，随后求 forecast。union wrapper 在 select/projection 前检查 full d/dmin/closing/forecast；NaN/±Inf 触发 ValueError，不能靠未选中行隐藏指定数组中的无效值。
- 来源为单独 guarded v2 diagnostic。runtime manifest 与源 SHA 确认 **v1 holdout 仍是原冻结 helper**；本轮不把 finite guard 的防护推广到未改动的 v1。
- 用 AST 只抽取真实 `require_finite` 和现有三项 unittest，在 CPU 独立执行，**3/3 PASS**：forecast NaN/±Inf 拒绝、J Inf 拒绝、有效输入不变。未导入 runner/Isaac、未执行模拟。它验证 helper 的拒绝语义，不是完整故障注入运行。

独立 NPZ 比较：guard `q[120,64,26]` 与旧 numerical candidate 前120步，**q、ee、cmd、exec、official_margins、controller_target、actuator_target 七项逐值全等**，q_initial 全等。已读取的 guard dense / row_receipt 数组均 finite。实际 actuator_target 的六步 FIFO 全等。

全行 finite 检查是源码调用及成功执行前缀支持的结论；row_receipt 只保存两个稳定行、三个槽的 forecast/distance/admission，未保存全部行 J/forecast，不能据此声称本审查从存储数据独立重建了全部9021行预测。

## 3. 实际 queue / 稳定行时序

独立核验 `row_receipt.npz`：

- row_admitted、row_distance、row_forecast 均为[120,3,2]，槽顺序[14,35,47]，行顺序[8974,7310]；
- actual_queue 为[120,3,6,26]，issued 为[120,3,1,26]；
- 全120步 queue[j] 严格等于本次 controller_target[t−6+j]（负索引用 q_initial），issued 严格等于 controller_target[t−1]（t=0用 q_initial）。**零不一致**，不是自行假设的外部队列。
- baseline 时序直接读取旧 causal NPZ 的 selected_id + selected_valid，按真实 detail_env_ids 映射，不靠 added_count 推断行身份。

下表为 pressure t≥60 后的首次 admission；forecast/distance 是 candidate 当步 pre-state 的实际 receipt：

| env / row | baseline | candidate | candidate measured d | candidate forecast d | 此 admission 当步产生的新目标最早可到达 actuator |
| --- | ---: | ---: | ---: | ---: | --- |
|47 /8974 |67 |62 |62.089212mm |22.662718mm |candidate68；baseline73 |
|35 /7310 |68 |63 |135.387704mm |10.427073mm |candidate69；baseline74 |

两个稳定行均提前5步进入 admission，支持“forecast 可在当前 raw 距离尚大时提前加入约束”的窄主张。**Admission 时刻不是执行时刻**：首个 pressure input t60 经 FIFO6 在 t66 到达；t62/63 admission 不能倒写已经入队的 t60目标。这些记录不单独证明投影可行性、PD闭环必然满足约束，或提前 admission 是全部改善的唯一原因。

## 4. Camera state 绑定、实际原图与限制

独立核验 visual v2 自身完整960步 FIFO6；q_initial 与 numerical development 相同，完整 cmd tape 严格相同。跨 cuda0 visual / cuda1 numerical 七项比较结果：

| 字段 | 全轨迹 strict exact |
| --- | --- |
|cmd |TRUE |
|q / ee / exec / official_margins / controller_target / actuator_target |各自FALSE |

差异不是“只差图片”。例如 q 最大绝对差3.541534rad、official margin 最大绝对差0.693593m。不能将 visual 的图当作 numerical baseline/candidate 对照轨迹的现场图片，也不能把差异唯一归因给 CUDA 设备或 render。独立 official<0 计数确认 **visual17/64**（四 class 的负值窗口数[0,0,0,17]），负值未用容差清零。

全部12组 state（slots0/14/35/47 ×steps71/75/959）逐组核验：

- state SHA、env/step、state_time=(t+1)dt、七个 view/image 身份与 receipt 一致；
- 四臂 q、controller_target、actuator_target 与同次 dense 对应 t/env 严格一致；sphere centers 严格一致；
- 9021 full distances 与 full_exempt，按冻结 row_class 重新求四 class 非豁免最小值，与同次 official_margins 严格一致；
- 8个非终态组 qd 与下一步 pre_qd_compact 严格一致；4个 t959组缺少下一步 pre_qd，终态 qd 不标为独立跨记录全等；
- 84/84 PNG 的 SHA 与 receipt 一致，全部实际解码为1280×720，84路径唯一，无空白均匀图；源码对应7次 render 保存，不是重复标签。图像哈希/解码核验不等于逐张视觉检查。

直接用 view_image 实看 **四张**原图：

- [env_047/step_0071_overview.png ](/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/visual_development_v2/multiview/env_047/step_0071_overview.png)
- [env_047/step_0071_u_opposite_high.png ](/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/visual_development_v2/multiview/env_047/step_0071_u_opposite_high.png)
- [env_035/step_0075_u_opposite_low.png ](/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/visual_development_v2/multiview/env_035/step_0075_u_opposite_low.png)
- [env_047/step_0959_u_pair.png ](/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/visual_development_v2/multiview/env_047/step_0959_u_pair.png)

env47/t71 overview 可见四臂与工作台，opposite_high 提供 U 对侧轮廓，邻居环境未挡住主体；env35/t75 low angle 可见 U 前侧大段臂体，但中央多臂/手部相互遮挡；env47/t959 u_pair 可见 U 臂整体终态，前景另侧机器人有裁切。四张提供布局/姿态证据，无法视觉量出毫米级安全球间距、不可见自碰几何或连续子步。其余80图仅作文件/绑定核验。

同次 state 的 env47/t71 table official margin 为+1.166672mm，t75为−9.473845mm，终态为+1.587003mm；终态正值不能抹去该窗口此前违规。t75负值来自独立 state/dense 重建，其对应 PNG 未列入本次四张视觉抽查。

**render invariance 的准确范围：** `visual_runner_v2.py` 对四臂全部64环境的 `.data.joint_pos/joint_vel/root_state_w` 和缓存 sphere centers 作 before/after torch.equal；capture 内只改 USD visibility/camera，并调用 render/camera update，未调用 sim.step。实际 SDK `articulation_data.py:538,754,763` 的这些 data 属性按 simulation timestamp 缓存。capture 未做 before/after fresh native PhysX getters，所以“render_state_unchanged_all_64”应解释为**被检查的 SDK 缓存状态不变**，不是独立 fresh-native 不变证明。早期 qd 的下一帧交叉核验补强同次状态绑定，但不补成完整 native render 反事实。该限制不阻断本节有界图像交付。

## 5. 首对 holdout 的独立经验结果与审查 findings

seed1930442907 两个 runtime protocol 均 complete；各 dense official_margins 为[960,64,4]。两者 q_initial 和全 cmd tape 严格相同，input_recipe 和 bank_assignment 文件 SHA 分别全同，actual actuator FIFO6 严格成立。两者 manifest 的 helper SHA 与冻结 v1 一致，baseline capacity512、candidate1024、actor rows32、projection_unchanged，未替换成 guarded diagnostic。

用全部960步、全部64槽、四个 official class 的 **margin<0，无 epsilon** 独立重算：

| 方法 |违规窗口 |margin<−5mm窗口 |四 class 违规窗口数 |
| --- | ---: | ---: | --- |
|baseline |44/64 |40/64 |[3,10,19,31] |
|queue_envelope |13/64 |12/64 |[0,0,1,12] |

逐槽 violation 与实际 episodes 一致。配对为 baseline-only31、candidate-only0、both13、neither20，与快照首对报告一致。candidate 违规槽为9,11,12,14,19,20,25,28,30,32,33,42,47；不能因改善或0 new宣称安全。这是预注册首对固定命令带上的有限改善，不是192配对最终计数。

**本次窄交付无未披露的计算冲突或阻断缺口；以下边界必须保留：**

1. v2 finite guard 只验证120步 development prefix，未修复/替换原 v1 holdout 的 finite-observation gap；全计划结果完成前不能声明384方法窗口或192配对窗口已审完。
2. 实际 row receipt 支持两个行的较早 admission 和实际 FIFO；不支持从聚合 count 推断所有行身份、重建全预测或唯一 PD/可行性根因。
3. visual/numerical strict exact **FAIL**，render 缓存比较不是 fresh-native 反事实；这些图仅绑定 visual v2 本身。无连续时间/物理接触安全证明。
4. 首对 candidate13/64 与 visual17/64 违规实际保留；生产策略、硬件和执行架构均未批准。原安全门、100ms FIFO6 无放宽，无 priority/actuator-side 架构认可。

## 6. 审计绑定 SHA256

下表本地源/报告路径相对本审查目录；raw 路径相对 `/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs`。baseline_causal 两项来自旧 `safety_mechanism_20261005/diagnostic/development_baseline`，只读。results 为上文读取快照的 bytes SHA，此后正常更新不纳入本次结论。

| 文件 | SHA256 |
| --- | --- |
| `results.json` | `a2c7ef21d56651a304b3d6a6043f7d19a0700c6606d9ad29defe6063012bd5cb` |
| `holdout/holdout_1930442907_baseline/cell_001.npz` | `d5c942271a6465d53da6edc0ea28f89043639f43d8c50eec72af3d605ece0b6c` |
| `holdout/holdout_1930442907_baseline/episodes.json` | `1311194a64664950c0715f46c17558de8514b3de22717ddfb916c97a829652ec` |
| `holdout/holdout_1930442907_baseline/protocol.json` | `efd6ae1380ea692b13056a20d0370fa99ca80141e25525d53e135cfa8e7d9ce4` |
| `holdout/holdout_1930442907_baseline/random_manifest.json` | `592b1e9d0c94a8843cadd6a26f746152ee984f3bb295aca51e74ad74d9a551e2` |
| `holdout/holdout_1930442907_baseline/input_recipe.npz` | `4f26a3a77f0881df86f62d8df141ddf514c73db33939c3372519908361507088` |
| `holdout/holdout_1930442907_baseline/bank_assignment.npz` | `7da501800e927d9aa58124ace9b4e830ce9a5d1f273237b5404709d4b226456e` |
| `holdout/holdout_1930442907_queue_envelope/cell_001.npz` | `e616037dd6d63800436037ec56c44ed9ea60ae7d797a9b0c94917de588fd9827` |
| `holdout/holdout_1930442907_queue_envelope/episodes.json` | `c0a8de2299f9e606f4462060cca6e88609f9a23107253d5fda4da8090c4d3b13` |
| `holdout/holdout_1930442907_queue_envelope/protocol.json` | `1db62912d1493d4eb84e03382209cca9218e36f806cdba38a3e3e2bffe7c9072` |
| `holdout/holdout_1930442907_queue_envelope/random_manifest.json` | `db95dcbc66fadb21a9f4e861e05221a75c184f1b3b4c7ea49676420f51e88b1c` |
| `holdout/holdout_1930442907_queue_envelope/input_recipe.npz` | `4f26a3a77f0881df86f62d8df141ddf514c73db33939c3372519908361507088` |
| `holdout/holdout_1930442907_queue_envelope/bank_assignment.npz` | `7da501800e927d9aa58124ace9b4e830ce9a5d1f273237b5404709d4b226456e` |
| `guarded_diagnostic.py` | `ff76f1071b9619a7dc07764799bc8b496ba6e4e440fdea3c05b07e2ed95fc880` |
| `mechanism_runner.py` | `174e297cdd9c8e48c00894e37426d7afbc48245f717de69fe2d3d4c6d173ce12` |
| `visual_runner_v2.py` | `d8db45fbab7531a33a3b4692753e173d958f4dc40c055fd97afb26b8a2f0b820` |
| `verify_observers.py` | `0dfd457d1c74d1c96b997b9ddb2ed5a881badcd6dc91176e4ab8d1e6df86d55c` |
| `observer_verification.json` | `a7ac2552b15e341ac9a509ad86a7d8e9749bdaf3bb8ac526a000d43e836bfca9` |
| `guard_registration.json` | `2b2ad48fc54fc664383b5210fe6a7446f2fa087693f18294bde0492a9deb2bad` |
| `visual_registration_v2.json` | `b48b660ec55728d0a83947e728ecca8cc19569dbb25694b9a7bae8e70a01ca99` |
| `holdout_registration.json` | `54e6cd57508162491434e6fe0d4922b8335b079774ac75a898ce9d41103de8fe` |
| `holdout_plan.json` | `d50be622b15f4c85712735f6d11a9029612f8f9dc97fb01fe8462789ebe189ad` |
| `test_finite_guard.py` | `2e296ffcf91edb9475ecb890b7603da57d251caca2133c53006945a95a12036c` |
| `finite_guard_tests.log` | `96a3ce9ca9ab54f51eb492ca49c7bf77d339645b47df420ec3cd2312c83c6d0f` |
| `ASTRA_DESIGN_REVIEW.md` | `aef3063b738fac52c9391f4bdf612df5695533e74e3c7b10d6b6fd1fa9d4910f` |
| `guarded_prefix/cell_001.npz` | `69349083dba9342ed0bb9717067d81ff4d099ebe9e581713828dfd8599e2c657` |
| `guarded_prefix/row_receipt.npz` | `4ae1c35b4a29635a1d3bc67963068d87c9778fa7cd7ee11a25f10d64c4b6ae84` |
| `guarded_prefix/guard_metadata.json` | `9c1132c768f441d4640548ef4c15cf3ee0d5cc8cf0442a16001429fd2484cb33` |
| `guarded_prefix/protocol.json` | `496f6f8227312b4fbffa2ebe37067a8ad5e403f3aa4025ca4293f4d9abc4d082` |
| `visual_development_v2/cell_001.npz` | `331290a26725dee799a7035a612f79ba9bb006f01257321f178d446e60aad65f` |
| `visual_development_v2/camera_receipts.json` | `1a5cde9b1427c37f0a218b955de1beec23edf19b4879dc20de42020dd559cd22` |
| `visual_development_v2/visual_protocol.json` | `d318020dd3199a192b39c5539287c5706271c66e4f32725d02318c3731d1b7e5` |
| `development/development_queue_envelope/cell_001.npz` | `4255395dcd281c015d34bfdfd56e2adde2fe6f5a3bf854f9648a604fd7bdfec5` |
| `baseline_causal.npz` | `1920cb097d6b0a3bb899f2a4eafab4d9b0d72490d93f1d347200d8889f65dd70` |
| `baseline_causal_metadata.json` | `aa53b8db24e8e169f29c1988e54459fc017580323790c44aeb389eb814999ab3` |

先前 `ASTRA_DESIGN_REVIEW.md` SHA 仍为原完成版本；本轮未重写它。此文件是 observer/部分经验审查，不是后续完整 holdout 终审。

