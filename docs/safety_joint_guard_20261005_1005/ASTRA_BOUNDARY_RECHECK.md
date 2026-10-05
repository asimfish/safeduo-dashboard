# ASTRA：最终分析工具有界静态复核

审读时间：2026-10-05 03:46–03:56 UTC（11:46–11:56 +08）。范围为当前 `analyze_v2.py` 的行集合、注册/银行和 closed-cell cache，以及 `verify_native_visual.py` 的 state/image 集合与五平面范围。只读源码/少量JSON元数据和内存边界探针；未执行完整 analyzer/validator、未重算旧18帧4490比较、未重新扫描原始NPZ/PNG/NAS SHA，未修改任何候选/生产/冻结源。

## 1. Verdict

**Blocked：当前缓存复用的完整输入/实际执行源码绑定，以及图片到 state 的绑定，还不能作最终工具验收。** 这不是判定已有 baseline 原始数据损坏，也不撤销其已完成物理任务；只限制这些工具目前能独立证明的范围。冻结控制源无需由本侧线修改。

当前第一 fresh pair 的46→16、deep40→11、rescued30/new0，以及 envelope 的9，是用户提供的进展，本次没有从raw重新评分。joint numeric未闭合时不补齐四因素结论。视觉图绑定其自己的 replay；baseline numeric46与visual47的 `FAIL_EXACT_REPLAY` 保留。

## 2. Dimensions

| 维度 | 本次状态 | 证据/范围 |
|---|---|---|
| Functional achievement | Fail | A/B多数补齐；cache manifest与image→state仍存在可通过的错误边界。 |
| Correctness/reliability | Fail | 实际cache函数的内存seam可接受漏消费输入；磁盘源码变化可把旧执行结果标成新SHA。 |
| Architecture | Risk | cache仅包装forecast audit；不得扩称整个cell及9快照已被缓存验收。 |
| API design | Risk | complete-cell前置条件依赖调用者；版本不同的cache使用同文件名，重算后再拒绝覆盖。 |
| Style/maintainability | Pass（静态范围） | 显式分项、shape/集合与跨run结果范围可读；无风格阻断项。 |
| Performance | Unable to determine | 本次未测性能；仅确认stale-cache路径逻辑上会先调用fresh audit再中止。 |

## 3. 已补齐且逻辑一致的门

- `analyze_v2.py:43–80` 现在严格核 full forecast/measured_d/dmin 的float32形状，packbits的uint8形状和尾padding；selected/baseline IDs 为固定capacity的int32、范围/唯一性均核。A0核actual==baseline，A1核 `actual == baseline ∪ forecast-dangerous`；baseline须包含raw-critical，forecast-added/required/count交叉一致。相同数量但危险row被替换的旧缺口已补齐。这是相对于保存的生产baseline集合核membership，不是独立重新执行全部actor32选择。
- `:81–85` full非豁免几何四桶最小值核到上一dense post帧，t0跳过无前一帧的比较，时序正确。`measured_d<0` 与 `measured_d−dmin<0` 仍分开；原official `<0`/deep `<−.005` 未改。
- `:99–105` actual FIFO及独立project history都重建为[960,6,64,26] oldest→newest并严格比较。原target debt/真实target增量与returned command的区别保留。
- `:159–182` 注册cell ID集合、argv、design/args、source/checkpoint、mode/A/E/capacity/FIFO6/helper SHA均有实际断言。实际冻结plan为12个唯一ID；bank NPZ及metadata都在 `research_source_sha256` 中。main先读回这些SHA，再核initial与对应bank.accepted_q，不能把这一银行字节绑定遗漏成未实现。
- `audit_closed_cells.py:11–15` 增量入口只对job complete且inspect row complete的cell调用cache。最终main尚缺同样的row-status显式断言：inspect若返回invalid/data=None，当前后续可能崩溃而不是形成规范invalid结果；未发现它会因此把不完整任务当安全成功。
- 当前cache生成的forecast输入确为6个固定文件加30个chunk。只读检查现存 `baseline_guard_1701627244.json` 的header/receipt清单：36项、集合完整、path正确，analysis SHA与本次审读版本一致。**本次未重新逐字节hash这36个真实输入**，因此不冒称独立完成该cache实际raw SHA复验。
- 相机现核21个注册 `(env,step)` 唯一集合、每组9个view names、189个唯一image路径；actual indices的整数/唯一/范围核已补；全64 fresh before/after、q到post、qd到next-pre、71/75的d/dmin/exempt/queue到next-pre、终态queue到最后6 issued targets均有对应断言。终态qd、closing及terminal dmin来源限制单列。

## 4. Blocking findings

### BR-01：缓存命中必须重新确定完整消费清单

- Severity：Major；Correctness/reliability；Introduced；Fail。
- Evidence：[analyze_v2.py:132](/home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/analyze_v2.py:132)。同SHA只循环 `saved['input_sha256']`，未核该key集合恰为6个必需文件+当前receipt全部chunk；也未校验schema/path等cache身份。
- 内存探针抽取实际 `forecast_audit_with_receipt` AST，使用纯内存Path及audit stub：删掉manifest中的 `project_diagnostics.npz` 并改变该内存文件，命中路径仍返回cached marker，fresh audit调用0次。没有更改任何真实文件。证明的是遗漏manifest输入可逃过现有复核，不是声称现存36项cache已经遗漏。
- Impact：不能保证“每个消费输入都逐字节复核”；诊断变化可不使缓存失效。
- Minimum fix：命中前从当前receipt构造必需消费集合，核manifest集合完全相等、schema/path/cell身份；核当前cell确已complete/960×64，再逐个hash。fresh与cached使用同一输入定义。协议/银行/episodes等由inspect/main每次fresh核的范围，应与forecast缓存范围分开声明。
- Verification：少任一固定输入、漏/加/重复chunk、消费文件改字节、cell未complete、cache path/schema不符均应fail；完整不变输入才可命中。

### BR-02：缓存SHA要绑定实际执行版本，不能只在结尾hash磁盘文件

- Severity：Major；Correctness/reliability；Introduced；Fail。
- Evidence：[analyze_v2.py:138](/home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/analyze_v2.py:138)、[:149](/home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/analyze_v2.py:149)。fresh audit完成后才记录 `sha(__file__)`；Python已加载函数不会随磁盘源码修改而更新。
- 内存实际函数seam：在audit stub执行期间将内存磁盘源码换为含新required gate的版本，receipt保存新磁盘SHA，但audit仍为旧执行marker。初始SHA `5342bba8…`，stamp为新SHA `cbc610b3…`，结果 `receipt_matches_new_disk_code=true`。这仅证明可发生的版本绑定错误，不证明实际baseline审计期间发生过此事。
- Minimum fix：固定进程/模块执行版本的SHA；审计开始和写receipt前确认源未改变，receipt只用固定SHA。长时audit期间不要修改其执行源。消费输入同样在审计前/后确认字节不变；closed raw不变的运行约定须显式保留。若改analysis，另开新版本worker和receipt。
- Verification：审计中修改源或任一输入必须拒绝写有效receipt；旧代码不能产生标新SHA的有效cache。

### BR-03：图片集合唯一还不等于图片绑定本state

- Severity：Major；Functional achievement/Correctness；Exposed；Fail。
- Evidence：[verify_native_visual.py:98](/home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/verify_native_visual.py:98)、[:126](/home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/verify_native_visual.py:126)。只取图片stem的view名，未核image路径中的env/step，也未比较receipt.images与state内部images。
- 内存集合反例将每槽三步的整组9图循环换步：21个注册state、9视角名、189路径唯一的现有membership条件全部通过，189个路径实际step均不匹配。若各PNG和SHA仍真实，文件hash/尺寸/std不会检测这种换组；无须伪造图片内容。此探针仅运行metadata条件，不是全validator/实际图像审计。
- Minimum fix：receipt.images与被SHA绑定的state.images逐项/规范集合相同，再要求每view映射到该env、step的规范图片路径和对应SHA。可接受显式map，不能只看stem后缀。
- Verification：跨step/跨env交换整组图、单图换组、state和receipt image清单不一致，都必须fail；正确固定映射才能PASS。

## 5. Non-blocking improvements（至多三项）

1. 版本不同的cache目前先fresh audit，再因同名cache存在而assert（:147）；内存seam已观察fresh调用1次后抛 `preserve an existing source-version audit rather than overwrite`。这是safe failure，**不是静默复用旧版本**，但新analysis不能在当前目录自然形成新receipt。建议以固定执行SHA作版本目录/文件名，保留旧receipt，避免删除/覆盖旧证据。
2. 补bank_assignment.risk_pair_index与已冻结bank标签的逐位比较。当前main核q_initial，四mode间核labels相同；冻结RiskSource确实从bank复制labels，但分析尚未独立核assignment属于原bank。这是分层/暴露报告的剩余绑定项，不是已经观察到标签错误。
3. cache只涵盖forecast audit；当前工具没有执行9个projection snapshot的独立算术。此前E0独立4490比较已另交付且本次未重跑；新的E1/joint快照须另审，不能把cache命中当作这些算术也已通过。

## 6. 五平面、native瞬时与forward跨run范围

当前validator离线以actual USD matrix/optics与代表sphere centers/radii重算四侧面+正深度平面；最终最小margin门为 `.050−2e−6`，2µm是视觉数值比较余量，不是物理安全阈值。该五平面局限已经在返回结果中明确；未读取USD authored clipping_range。观察深度若全落在1..8m，可与冻结配置(.1,60)作有界一致性说明，**不能称实际USD near/far已校准，也不能称完整六面验证**。不证明轮廓/像素遮挡/连续接触。

SDK stale pose与actual USD pose仍分开；当前零offset、matching aspect/square-pixel条件下K复算有效范围清楚。all64 native before/after exact仅是采样render段两端的瞬时状态不变证明；不证明同一或跨次未来轨迹无扰动、不证明所有中间子步。跨run每字段exact与first difference/max difference分别输出，actual numeric/visual违规数已分列；`FAIL_EXACT_REPLAY`应继续保留。PolicyDriver的deterministic mean不能推出PhysX跨进程轨迹必然bit-exact，也不能把未知torchseed称唯一根因。

## 7. 最小修复、执行证据与绑定

本侧线不实施修复。最小修复仅在离线工具：完整cache manifest/closed身份、固定执行版本+版本化receipt、state→image规范绑定。随后只需CPU边界探针验证上述拒绝路径；最终full analyzer和native validator应在固定最终SHA下实际执行。它们PASS仍不覆盖未闭合物理窗口或安全批准。

实际执行：源码读取、AST解析、plan/cache JSON header检查；三个cache内存seam（stale版本、漏消费输入、执行中源改版）及图片换步membership反例。未运行模拟/GPU、未导入完整analyzer触发写文件、未重做旧快照/全raw/全PNG审核。本次唯一写入此MD。

| 审读文件 | SHA-256 |
|---|---|
| analyze_v2.py | 5342bba80cbff4e2a044f260b9295c53a02315fe936bb65520debd80fa472e41 |
| verify_native_visual.py（03:56最新版，增加numeric违规数；绑定条件未改） | bfeaea21d80c271cddac64bfc26c7ed357c22ece45b8428c75979a523eab8106 |
| audit_closed_cells.py | 0461c7303ed0e07f646f2f66ed24d939b44817c4c5f00d7d6f0b4ea3534623ee |
| register_campaigns_v2.py | f1c335c7dbb6a06a7fa95704555d639756e5e602db7bd2429c497f649fdc2dd8 |
| holdout_plan_v2.json | 0cfc248bb5d65d6a628a30122c5bf95ff00a4da3f46a4220f595ab2675a0bf0a |
| visual_runner_v5.py | dfddc2f1c56b1fb4b428a829f9e602fd0648517f26e28ea187af5d2ce1436523 |
| visual_registration_v5.json | b8310ede26582f0126e6115a82ab3a58877f01e1bd63edb73e72fef797c9986e |

冻结diagnostics/tests SHA仍与前一报告一致，均未修改。后续源码修复和实际E1快照结论另行绑定新执行SHA，不以本报告预填PASS。**production_promoted=false；hardware_approved=false；物理安全策略未批准。**
