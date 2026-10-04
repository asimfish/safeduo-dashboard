# Astra final holdout review / bounded native-render review

**Verdict：Pass，仅针对冻结 v1 六组、384 方法窗口的经验数据交付；安全策略仍 BLOCKED_SAFETY_STRATEGY，v3 camera calibration 仍 BLOCKED。** 本审查不批准候选、生产策略、硬件、priority 或 actuator-side 架构。只新增此文件，使用 CPU 读取源码/JSON/NPZ/PNG；未运行模拟、Kit/GPU或修改既有审查、源、协议、配置和面板。

独立计算时间：2026-10-04T17:30:18.498853+00:00 至 2026-10-04T17:30:30.601012+00:00（北京时间2026-10-05 01:30:18–01:30:30）。读取的最终 results 更新时间 2026-10-04T17:18:07.241202+00:00；所有统计先从原始数据计算，再比较 results，不能以 results 自报替代原始核验。SHA256 绑定见末节。

## 1. 维度结论与计数范围

|维度 |状态 |证据与范围 |
| --- | --- | --- |
|v1 full holdout 的完成性/统计正确性 |PASS_EVIDENCE_REVIEW |六个独立进程均 runtime complete/exit0、64槽×960步；384方法窗口、192配对，0 invalid/pending；逐槽 endpoints/episodes与汇总一致 |
|配对输入、来源与 FIFO |PASS |全部配对962-frame tape、初态/limits/config全等；实际 actuator target 严格 FIFO6；冻结源/actor核验无差异 |
|预算/target边界 |PASS，按既有数值门 |最大 required rows baseline146/candidate217；target soft limits严格通过，实际增量按已存在的+5e−7数值门通过，非无条件实数盒不超界主张 |
|v1策略安全 |FAIL |candidate仍40/192违规，新增2例，deep30/192；原negative-margin门未满足 |
|v2 finite诊断 |仅 bounded PASS |此前独立120步前缀及3个CPU fault tests通过；原文件SHA保持。未作用于v1完整384 |
|v3 native/render与camera接口 |Risk / calibration FAIL |本次只审steps71/75的8组/56图；fresh native读取及源码after断言与缓存区别明确，camera pose元数据全零 |
|性能/硬件/完整生产接口 |Unable to determine |未作实时benchmark、硬件或搬运QA；不据此批准执行架构 |

既有 ASTRA_DESIGN_REVIEW.md、ASTRA_EVIDENCE_REVIEW.md 字节未修改。开发/finite prefix/visual replay 不加入384 holdout分母。3组复用旧stable初态bank，刷新外部command seed；192是配对命令窗口，不是192个新初态或IID独立可靠性试验。

## 2. 六组原始 endpoints 与三组配对

直接读取全部六个 `cell_001.npz`、`input_recipe.npz`、`mechanism.npz`、`bank_assignment.npz`、两个episodes载体及protocol/manifest。official_margins为[960,64,4]；endpoint为任一步任意official class **margin<0，无epsilon**，deep为margin<−.005m。四class顺序[cross,self_F,self_U,table]，豁免规则不变。first failure取每槽首次negative step，无失败用960哨兵。对六组全部槽重算class flags、violation_steps、deep、首次失败及episodes一致性。

|command seed |条件 |违规 |deep |class违规窗口数 |最大required rows |
| --- | --- | ---: | ---: | --- | ---: |
|1930442907 |baseline |44/64 |40/64 |[3,10,19,31] |146 |
|1930442907 |queue_envelope |13/64 |12/64 |[0,0,1,12] |137 |
|1940677742 |baseline |39/64 |33/64 |[3,11,16,27] |78 |
|1940677742 |queue_envelope |13/64 |9/64 |[0,0,0,13] |130 |
|1931983758 |baseline |37/64 |32/64 |[2,7,19,32] |54 |
|1931983758 |queue_envelope |14/64 |9/64 |[0,0,1,13] |217 |
|合计 |baseline |120/192 |105/192 |[8,28,54,90] |146 |
|合计 |queue_envelope |40/192 |30/192 |[0,0,2,38] |217 |

class计数可能重叠，不能相加作为窗口数。candidate主要剩余table违规，但零cross/self_F窗口不是全状态/连续时间安全证明。

|command seed |baseline→candidate |rescued/baseline-only |new/candidate-only |both |neither |
| --- | --- | ---: | ---: | ---: | ---: |
|1930442907 |44→13 |31 |0 |13 |20 |
|1940677742 |39→13 |26 |0 |13 |25 |
|1931983758 |37→14 |25 |2 |12 |25 |
|合计 |120→40 |82 |2 |38 |70 |

“0 new”只对前两对成立，不能推广到三对完成结果。两例新增均属seed1931983758：

|env |首次违规step（零基、post-step） |首次table margin |该窗口最小table margin |
| --- | ---: | ---: | ---: |
|11 |833 |−2.931103mm |−19.352719mm |
|58 |724 |−2.801582mm |−14.777824mm |

两例 baseline 均无official违规，candidate均达到deep门；不是近零负值。此次全部负值原样保留，没有修改容差或排除槽。首次class与四class最小值来自raw class margins，不给出未存储的精确body/row身份，也不据此认定唯一根因。

独立重算六组报告中的首次失败列表、failed_envs、零前缀/首pressure delivery前违规、干预/运动、command extrema、holds/amplitudes、required/added rows，以及三组配对和下节所有coverage/totals；**与最终results的这些字段递归比较零差异**。没有将旧122/192 protected结果、39/64 development baseline或visual窗口混入本表。

## 3. 严格 FIFO、配对、来源、预算与边界

六个 campaign job 的 PID互异，均exit0、runtime complete、completed_cells=1；实际dense [960,64,26]及64个episodes齐备，protocol SHA与campaign子进程receipt一致。源 `random_battery.run_window` 每一步对所有槽term/trunc做检查，发生任意window内reset则抛错；完整写盘与complete支持该运行期无reset检查通过。未存逐步term/reset tensor，不声称额外从该缺失tensor复算reset。

三对均直接验证：

- 全962步recipe tape严格全等，实际执行960步cmd等于tape[:960]；60步零输入前缀保留；q_initial、sampled_initial、initial_violation、joint_soft_limits、ee_initial全等。
- 实际q_initial、limits、strata labels等于对应注册bank原始数组；独立bank SHA与registration和runtime manifest一致。每bank为6pair×8槽+16general；没有重新挑选初态或排除失败槽。
- actual actuator_target[t]严格等于controller_target[t−6]，前6步等于q_initial；不是依据queue长度字段推测。实际dt=.016666s，FIFO6名义100ms（记录值.099996s）。
- resolved_config、effective_backstop、effective_coordinator、source/checkpoint SHA配对全等，且六组一致。保留backlog_aware=false、predict_backlog=true、pending_target_steps6、actor32、原solver/horizons/exemptions，未加入v2 guard。
- 独立核对计划中的 **279个core source与42个research/provenance项**，以及各runtime core source映射，当前bytes无SHA差异；actor SHA `4dc303940d9fa6de2dbdf1e38599e7219ec3e816179a0884920826be46abd0d6`。冻结helper `174e297cdd9c8e48c00894e37426d7afbc48245f717de69fe2d3d4c6d173ce12` 与六manifest相同。
- 全部已存numeric dense以及mechanism数组finite，initial violation全0；recipe、metadata与同次dense绑定成立。

required_rows与同次critical_selected_count全逐值一致，六组extra均[960,64]。baseline最大行数[146,78,54]，candidate[137,130,217]；未超过冻结512/1024预算，**实际没有任何>512的env-step**。源码超过1024即abort、不截断，原优先/raw-critical行与forecast行union保留。候选新增forecast-row的env-step为[12939,9902,13280]，单步最大added为[124,93,193]。本数据中的改善不能说成“使用超过512的容量”造成；实际宽度均低于512。

target soft limits六组严格通过。nominal box=vmax×dt=.024999rad；六组实测max target增量均.024999141693rad。原报告box门为nominal+5e−7，本审查沿用该已存在的数值门，六组over门计数0。**若按数学上的strict >nominal计数，存在浮点级超出**：baseline [22873,21161,20503]、candidate [22158,20830,20242] env-step，最大超出1.41693115e−7rad；不能将published gate PASS写成无限精度增量全不超界。official margin判违从始至终不使用此target数值容差。

冻结元数据仍有已披露的旧描述：holdout_plan.candidate / candidate manifest.allocation提到budget512，而manifest.capacity/capacity_reason、真实helper与MECHANISM_CONTRACT定义candidate1024。按实际源/运行receipt判定；这是文字范围不一致，不是本次实测改变或截断，不能修改冻结文件掩盖它。

## 4. 最终 coverage 与暴露缺口

独立重算同次NPZ的所有six-pair margin exposure、EE相对接近/远离/切向/静止次数、joint occupancy、EE voxel/AABB、关节range/path、command holds/amplitudes与干预/运动比例。未调用主analyze_risk/analyze_results/coverage_metrics的统计函数；按其公开定义用NumPy另行计算，最终coverage和totals递归 **零差异**。

pressure exposure固定取实际delivery后steps66..959，>=3步pair margin<.080m；near<.005m和overlap<0为任一步。方向统计是EE径向变化的描述，不能当作body最小距离梯度或碰撞因果。以下分母仅是被预分配给该风险pair的24槽（3banks×8）：

|指定pair |分母 |>=3步80mm暴露 baseline / candidate |candidate未达到暴露 |near5mm baseline / candidate |overlap baseline / candidate |
| --- | ---: | --- | ---: | --- | --- |
|F_L-F_R |24 |24 / 24 |0 |6 / 0 |4 / 0 |
|F_L-U_L |24 |20 / 20 |4 |0 / 0 |0 / 0 |
|F_L-U_R |24 |22 / 22 |2 |3 / 0 |3 / 0 |
|F_R-U_L |24 |24 / 24 |0 |2 / 0 |2 / 0 |
|F_R-U_R |24 |21 / 21 |3 |2 / 0 |1 / 0 |
|U_L-U_R |24 |24 / 24 |0 |4 / 0 |3 / 0 |

两条件均135/144指定风险槽达到该暴露，9槽未达，另48general槽不并入144分母。全部192槽（含非指定pair）的六pair暴露数baseline[79,20,27,31,21,78]、candidate[72,20,27,28,21,67]；不能用192作为各指定pair已充分测试的分母，也不能从joint/EE覆盖估计可达空间覆盖率。

两条件初始joint十等分bin占用均94.6154%，加上实测轨迹后均95.7692%。local EE的0.1m voxel计数初始[136,134,127,127]；实测+初态baseline[2352,2324,2217,2288]、candidate[2322,2304,2195,2294]。这些是被观察单元，没有全部可达/无碰撞空间分母。

physical q在soft limits外的已采样joint entries baseline44231、candidate46240，不能被target soft-limit PASS替代或当作新增official碰撞窗口。四臂同一步运动>1mrad的比例baseline46.6377%、candidate43.6325%；执行/原command L2总量比.575539/.569234。零外部输入前缀内仍有保护controller输出，三bank各[43,42,40]槽；最大前缀关节漂移.34547rad。零输入不等于静止或无控制，不能把前缀当作纯被动基线。

全部失败保留；同一银行配对、选择后的条件初态、重复关联时间序列均不提供IID pooled置信区间或安全认证。

## 5. finite-gap：v1全384与v2前缀分开

v1六组所有**已保存**q/qd、cmd/exec、margins、targets、pair/table指标及forecast_only_min均finite。该结果不能推出未入选全行的J/forecast均finite：完整9021行forecast/J未在v1 NPZ中保存，且v1 union没有全行finite abort。原观察盲区继续存在；没有把finite最低统计升级为完整防护。

v2 guarded_diagnostic的SHA仍 `ff76f1071b9619a7dc07764799bc8b496ba6e4e440fdea3c05b07e2ed95fc880`，是独立120步development prefix。此前实际CPU审查的七项prefix全等、actual6queue receipt全等、2行提前5步admission和3/3有限值fault helper tests保留，原ASTRA_EVIDENCE_REVIEW.md SHA仍 `32dfcb46c061379f8ddd0decc41cfd37abef5516866186012ed0a0f33ebd4421`。它的源码guard在select/projection前检查全d/dmin/closing/forecast和J，以及q/qd/issued/pending/rawproposal；不计新增完整16s窗口、不改冻结holdout。此次复用已经完成且bytes未变的有界检查，没有重复模拟或把它扩成384窗口故障注入验收。

更早admission、整体违规减少与两个新增失败共同构成经验结果，仍不能唯一判定可行性、stored closing authority、PD误差、非线性forecast误差或豁免机制的责任比例。

## 6. v3独立视觉/native replay：局部状态有效，camera calibration BLOCKED

本次read-only partial检查时间 2026-10-04T17:31:46.005386+00:00。所审范围严格为steps71/75、slots0/14/35/47的 **8组state、56 PNG、2个native_render NPZ**；全部PNG独立SHA/解码1280×720。读取时visual_protocol已变为complete，日志已出现step959采集；**本次未审最终959组、全部960-step visual dense或主线程最终v3 verification receipt**。它们不影响已闭合的v1 holdout核验，不以“仍在跑”隐瞒当前runtime状态。

v3源直接对四臂root_physx_view.get_dof_positions/get_dof_velocities/get_root_transforms/get_root_velocities做fresh getter().clone()；before包含全部64槽，F每臂19DOF、U每臂18DOF。每slot七次render后再次fresh getter、clone，torch.equal与最初before比较，发生变化则assert失败。与v2缓存比较有实际区分力；state生成路径发生在该assert之后。

已审2个before NPZ的16张量全部finite、64槽形状成立；8个state的fresh_native_selected每字段逐值等于同step NPZ对应槽，控制臂q/qd与native正确joint顺序对应、zero mismatch。**after native arrays未持久保存**，因此本次离线独立比较限于before与selected/cached状态的绑定；all64 before/after不变由真实fresh getter源码assert和已完成capture记录支持，不能写成离线对两个已保存完整after/before数组复算。

直接看两张v3/t75/env47原图：

- [env_047/step_0075_u_pair.png](/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/visual_development_v3/multiview/env_047/step_0075_u_pair.png)
- [env_047/step_0075_u_opposite_high.png](/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/visual_development_v3/multiview/env_047/step_0075_u_opposite_high.png)

两侧视图有实际不同构图，可见U臂/桌面且无邻居环境遮挡；中央手部及臂体相互遮挡，不能视觉确认毫米级官方球距离或子步碰撞。图像可读与外参有效性不是同一判据，图像不能充当cuda1 numerical holdout counterpart。

**实测calibration blocker：所审56个view的actual_position_world_m全部[0,0,0]，quat也未随七view改变；与eye+origin不符。** 例如env47 origin[-6,14,0]，overview请求world eye约[-2.3257,9.5260,4.4739]m，保存actual仍零。intrinsic矩阵finite且正焦距不修复无效extrinsic，不能给完整camera-calibration PASS。

独立核对安装SDK定位到准确接口范围：

- `camera_cfg.py:77` 的update_latest_camera_pose默认False，文档明确False返回初始化pose；
- `duo_env.py:1070` 的实际CameraCfg未覆盖该flag；
- `camera.py:336` 的set_world_poses_from_view只调用view.set_world_poses；不更新camera.data pose；
- `camera.py:496` 的_update_buffers_impl只有flag为True才调用_update_poses。`camera.py:607`中get_world_poses是后者内部实现；从“函数内部存在getter”不能推断此次render/update调用过它。

这足以解释当前保存字段可能停留初始化pose；**尚不是native view.get_world_poses坏了或图像相机实际没移动的实验证据**。最小可区分接口验证是从真实camera prim的USD ComputeLocalToWorldTransform读回world transform/实际camera属性，明确OpenGL −Z光轴与world/ROS约定，验证eye、look-at方向、K/分辨率及可见点投影。observer开启update_latest_camera_pose后也可交叉检查，但本审查未修改设置或启动此实验。主线程拟另建v4属于后续prepared/运行证据；本报告不提前验收。

## 7. Findings、最小后续验证与交付边界

本轮v1完整原始经验统计、source/inputs/FIFO/coverage对照无计算差异或未披露的missing job。v1数据交付PASS不覆盖以下阻断：

1. **安全准入阻断**：40/192 candidate官方违规，含2个相对baseline新增deep table案例；硬件、生产候选和执行架构均false。
2. **finite防护不可推广**：v1未实现全行finite abort；v2仅120步，不是完整384修复版本。不能追改冻结v1或将finite-prefix补到v1名下。
3. **camera-calibration阻断**：v3实际pose字段全零；旧v2 cache limitation保持记录。v3/v4是独立replay，不能贴成数值对照原图或挪入holdout分母。

非阻断的描述纠正：说明已存在target+5e−7门与严格nominal差异、旧budget512描述字段与实际1024差异、native after数组未保存的离线范围。最小后续交付验证只需新视觉诊断在自己冻结记录上证实真实pose/projection与state绑定，并明确是否保存after数组；不需要重跑/篡改六组holdout或调整安全阈值。不存在本审查授权的源码修复、模拟或硬件动作。

## 8. SHA256绑定

holdout路径相对RAW/holdout；本地报告/源相对H。原始六组所有主要NPZ、protocol、manifest、两episodes载体均独立读取和hash。v3记录是本次partial读取的bytes；v3/visual_protocol之后有元数据更新不纳入此快照。PNG完整hash绑定由各state.images和state文件SHA支持，实际视觉抽查仅上文两张。

|文件 |SHA256 |
| --- | --- |
| `results.json` | `0c582d2d1dc8bfae397429d8d37fe36ad2f95a5cd357be3c136621abe711fa5d` |
| `holdout_1930442907_baseline/cell_001.npz` | `d5c942271a6465d53da6edc0ea28f89043639f43d8c50eec72af3d605ece0b6c` |
| `holdout_1930442907_baseline/cell_001.json` | `28132b8cda9af78722df68abb223876611ed574b76c2443d6edffe37ac4b6d4e` |
| `holdout_1930442907_baseline/episodes.json` | `1311194a64664950c0715f46c17558de8514b3de22717ddfb916c97a829652ec` |
| `holdout_1930442907_baseline/mechanism.npz` | `34b6420f8e83527a29acf7647ce6ab8e8d3a617f35f7610e4c70440e779e153c` |
| `holdout_1930442907_baseline/input_recipe.npz` | `4f26a3a77f0881df86f62d8df141ddf514c73db33939c3372519908361507088` |
| `holdout_1930442907_baseline/bank_assignment.npz` | `7da501800e927d9aa58124ace9b4e830ce9a5d1f273237b5404709d4b226456e` |
| `holdout_1930442907_baseline/protocol.json` | `efd6ae1380ea692b13056a20d0370fa99ca80141e25525d53e135cfa8e7d9ce4` |
| `holdout_1930442907_baseline/random_manifest.json` | `592b1e9d0c94a8843cadd6a26f746152ee984f3bb295aca51e74ad74d9a551e2` |
| `holdout_1930442907_queue_envelope/cell_001.npz` | `e616037dd6d63800436037ec56c44ed9ea60ae7d797a9b0c94917de588fd9827` |
| `holdout_1930442907_queue_envelope/cell_001.json` | `dad1ecf8078ca39256424166610b35081bcb8182efe4ecab5a7673ee7580c4f9` |
| `holdout_1930442907_queue_envelope/episodes.json` | `c0a8de2299f9e606f4462060cca6e88609f9a23107253d5fda4da8090c4d3b13` |
| `holdout_1930442907_queue_envelope/mechanism.npz` | `4392782117bca3544753c423ce60cfc02652ff6c25f0e1e283e6a1755ff268bb` |
| `holdout_1930442907_queue_envelope/input_recipe.npz` | `4f26a3a77f0881df86f62d8df141ddf514c73db33939c3372519908361507088` |
| `holdout_1930442907_queue_envelope/bank_assignment.npz` | `7da501800e927d9aa58124ace9b4e830ce9a5d1f273237b5404709d4b226456e` |
| `holdout_1930442907_queue_envelope/protocol.json` | `1db62912d1493d4eb84e03382209cca9218e36f806cdba38a3e3e2bffe7c9072` |
| `holdout_1930442907_queue_envelope/random_manifest.json` | `db95dcbc66fadb21a9f4e861e05221a75c184f1b3b4c7ea49676420f51e88b1c` |
| `holdout_1940677742_baseline/cell_001.npz` | `5a7e378e8436c923f791c4f470a862ab9749d9b361535f3a7d5a490c1a34b646` |
| `holdout_1940677742_baseline/cell_001.json` | `9d0e9a25d1467882ab184e4f21a48e126005d37f2ed9093a476ed04b9c317131` |
| `holdout_1940677742_baseline/episodes.json` | `8fb2a8e957247317c144e0f9657a3a4e68dbedf8e9bacf1c7f29443928e8317e` |
| `holdout_1940677742_baseline/mechanism.npz` | `88426148f599974ccca4d3896b040fa02fafefbb139fa377d63be0b2659c08d3` |
| `holdout_1940677742_baseline/input_recipe.npz` | `5838b5540b0ebcf53c19aeabdfd836262a47713c27cf332a3c0708508eff45c8` |
| `holdout_1940677742_baseline/bank_assignment.npz` | `f3ccea1bacdd88a4864b19646a632b51f5aaa63cf961da30a7b95d7a16401f5b` |
| `holdout_1940677742_baseline/protocol.json` | `fb35dc2a7d9574c5931255cd133e53d44d31687b9d01a8d2c1e75490b73d9e2a` |
| `holdout_1940677742_baseline/random_manifest.json` | `91ac9e0785c840cd9bf03675fb099ad19c339e9cebf609ea13b8dbd7b3692920` |
| `holdout_1940677742_queue_envelope/cell_001.npz` | `0e0d77b1bbc08a8d3ebb0f524375497db70c5e8b415190be18ffa92b427201b8` |
| `holdout_1940677742_queue_envelope/cell_001.json` | `7e1fe7d95280d01aa648a40fd39b2043ec45c16887c3428458391216ec0e3388` |
| `holdout_1940677742_queue_envelope/episodes.json` | `3a476b77e57fe82db176d2e8024c86b3d305ae0cde63b160f3e00f53cab669f6` |
| `holdout_1940677742_queue_envelope/mechanism.npz` | `f5e28d3aba3cd24cbf332dacbfcaceb3214d8d6f1245e6793b3e05e2943c686d` |
| `holdout_1940677742_queue_envelope/input_recipe.npz` | `5838b5540b0ebcf53c19aeabdfd836262a47713c27cf332a3c0708508eff45c8` |
| `holdout_1940677742_queue_envelope/bank_assignment.npz` | `f3ccea1bacdd88a4864b19646a632b51f5aaa63cf961da30a7b95d7a16401f5b` |
| `holdout_1940677742_queue_envelope/protocol.json` | `74e5c6cbaec7d0bdd4ecc85fc56021c2bdb3a7f7e2d1fdf1cf2c89fbe1fe378e` |
| `holdout_1940677742_queue_envelope/random_manifest.json` | `585332557dfd881e61a925b18a344d5628af2c24a832a4c7855b7f8b3f264bf7` |
| `holdout_1931983758_baseline/cell_001.npz` | `2eed7e99cb62c03759f423cf96040cfb4eb586686c15aa26ddc1e5d799a31ef0` |
| `holdout_1931983758_baseline/cell_001.json` | `c1ed14670ae44fd3d7586ab6b19b84c7d10a3d5745aca902adb0888fbbe0c12f` |
| `holdout_1931983758_baseline/episodes.json` | `4f20389a0163bfa78b68f14c2ede5ea11e296aa33d718f07c64f6b279ea2cf08` |
| `holdout_1931983758_baseline/mechanism.npz` | `16304f946727ec9fc33d0bdc3d9cb6a651ec25be51e5fa9b7daefc169c3053d1` |
| `holdout_1931983758_baseline/input_recipe.npz` | `d7a013af25e9966926766f7108e2d3d07e65ba95c985e651bd8cc31c722c9106` |
| `holdout_1931983758_baseline/bank_assignment.npz` | `08b4f845109b7503e7c5ca94cf3776e706b15ac8659e691090941e8a9be6e478` |
| `holdout_1931983758_baseline/protocol.json` | `cb47d8a581b4d22471d0abaaee9ac17815d8d891a5ce22b98ad55d7626e5c0c0` |
| `holdout_1931983758_baseline/random_manifest.json` | `ca641b0bad7f759277dc16336f7972f1509092ffbdac9b96bb7cb4a57b12f366` |
| `holdout_1931983758_queue_envelope/cell_001.npz` | `aaff5e1dc6ba2fb3ed663bf08f9c4c2d828c926dd9cf637f813ee1d5934c064f` |
| `holdout_1931983758_queue_envelope/cell_001.json` | `21e33347fba97c29663e5e07dfadf2a0d011b0654c10b4b06217298dff48a433` |
| `holdout_1931983758_queue_envelope/episodes.json` | `cae82dd847a84bb4ea9961eb75a2de5c9c2fde0eaca81972bfa938e3fcf9fc6c` |
| `holdout_1931983758_queue_envelope/mechanism.npz` | `fe71b73b13a80a4bad89b3e31aab9e2ea06644053fa613cbcf87f43d37f82da0` |
| `holdout_1931983758_queue_envelope/input_recipe.npz` | `d7a013af25e9966926766f7108e2d3d07e65ba95c985e651bd8cc31c722c9106` |
| `holdout_1931983758_queue_envelope/bank_assignment.npz` | `08b4f845109b7503e7c5ca94cf3776e706b15ac8659e691090941e8a9be6e478` |
| `holdout_1931983758_queue_envelope/protocol.json` | `b5300af672c888a0a55688fc1250cedb7587d5fb97b6480d03114bd4c8bd5e98` |
| `holdout_1931983758_queue_envelope/random_manifest.json` | `830b6b1ba625d3e2c5c34a6686a40d98229e4f98902d1dbca5c5a85b028c9f1f` |
| `holdout_plan.json` | `d50be622b15f4c85712735f6d11a9029612f8f9dc97fb01fe8462789ebe189ad` |
| `holdout_registration.json` | `54e6cd57508162491434e6fe0d4922b8335b079774ac75a898ce9d41103de8fe` |
| `mechanism_runner.py` | `174e297cdd9c8e48c00894e37426d7afbc48245f717de69fe2d3d4c6d173ce12` |
| `guarded_diagnostic.py` | `ff76f1071b9619a7dc07764799bc8b496ba6e4e440fdea3c05b07e2ed95fc880` |
| `guard_registration.json` | `2b2ad48fc54fc664383b5210fe6a7446f2fa087693f18294bde0492a9deb2bad` |
| `observer_verification.json` | `a7ac2552b15e341ac9a509ad86a7d8e9749bdaf3bb8ac526a000d43e836bfca9` |
| `visual_runner_v3.py` | `af4ad0737c1e7f724821465ed53235beb484498bd10bd1172b0eafb2780592a3` |
| `visual_registration_v3.json` | `eff104379fe638b399216b404ed3bd3a06a766d0b65633e878ea4dfedad7d996` |
| `ASTRA_DESIGN_REVIEW.md` | `aef3063b738fac52c9391f4bdf612df5695533e74e3c7b10d6b6fd1fa9d4910f` |
| `ASTRA_EVIDENCE_REVIEW.md` | `32dfcb46c061379f8ddd0decc41cfd37abef5516866186012ed0a0f33ebd4421` |
| `REVIEW_RESPONSE.md` | `ba99d932a7a4a7cca80c0d9147cf586bfce12ba3752327429936e9d880c38b60` |
| `MECHANISM_CONTRACT.md` | `4557b45b4112a6469fff93fcfd471f336255f344f7b93c3989f9b7b35e2f0d6d` |
| `holdout/campaign.json` | `a4ad6652a2108710b07a37ae56ecef88515b9d6da36a734c7517aec554f64f2a` |
| `v3/native_render_step_0071.npz` | `9887fbcd6d4028a140387e0ffb4466c49219d70f14947c85001ef6c0c175f299` |
| `v3/multiview/env_000/step_0071_state.json` | `1c1f1798c67435f4d97a0380bbcd582476198f6cb5350456839968063d8714f6` |
| `v3/multiview/env_014/step_0071_state.json` | `6d7297bf7983fc5b0a6aff8e257fde84c52513d499d4345fdeea1b35ca2e4a73` |
| `v3/multiview/env_035/step_0071_state.json` | `1a384ccc0491bd3c4dfbf6619e8f84a87d7641c4ddc963bd7d8afa95850396f1` |
| `v3/multiview/env_047/step_0071_state.json` | `7336bcc4cebe8e4dc117aab9a8fa579f68cbe2b3bbc188559af8ec69238e07f9` |
| `v3/native_render_step_0075.npz` | `628c8a49c688dfea92bb77b5b516478aaa7cc0a7c7c7f4d7444651379ef31d32` |
| `v3/multiview/env_000/step_0075_state.json` | `ae240b245b674ed1946d37c85528a3715ba71e3f4d7b77f104928701d25378a2` |
| `v3/multiview/env_014/step_0075_state.json` | `be7b068b229a9ea31f8bda9f928fd3a93b9b7759664a65eb7169494a7d833feb` |
| `v3/multiview/env_035/step_0075_state.json` | `9974756d8b3ab663d55b83b25af0cbe3c03c2ecfeb43565ec6f8d6f87c10d838` |
| `v3/multiview/env_047/step_0075_state.json` | `91f6f9be5177ebebfe09bceb527ce9e3935431892424a43c4b7e556ffe3bf0e6` |
| `v3/visual_protocol.json` | `40655ce1e8b53805c36d2e1aff6c2895ec31dab317ab87f91c373fc0717dd618` |
| `v3/visual_runner_v3.py` | `af4ad0737c1e7f724821465ed53235beb484498bd10bd1172b0eafb2780592a3` |
| `v3/visual_registration_v3.json` | `eff104379fe638b399216b404ed3bd3a06a766d0b65633e878ea4dfedad7d996` |

scope：**v1 full384经验终审 + v2已完成bounded有限值证据的版本界限 + v3早期fresh-native/camera接口局部审查**。没有安全/硬件晋级，没有旧封存审查字节变化。

