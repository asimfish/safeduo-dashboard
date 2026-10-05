# ASTRA：离线证据绑定修复有界复核

审查日期：2026-10-05。**PARTIAL_REPAIR_CONFIRMED；BLOCKED_COMPLETE_STALE_CALLER_GATE**。这不是运行候选的失效判定，也不是最终768窗口验收。

本次实际读取新版源码、冻结清单、15项测试源码与其实际PASS日志、supersession记录和视觉验证receipt/log；未修改任何候选或分析工具，未运行模拟，未重复父线程完整forecast/378图验证。另用实际函数AST执行了一次完全内存化的缓存命中反例，没有写测试fixture或外部文件。

## 已确认的修复

- `audit_input_paths` 要求protocol complete、960步、64环境、completed_cells=1；37项输入是7个固定文件及30个连续32步chunk。chunk起止及相对路径逐项等于规范值，因此缺漏、重复、额外或traversal路径不能通过。
- 缓存以实际执行analysis SHA分目录；命中需schema、resolved cell路径、cell ID、analysis SHA及完整37项键集合/字节SHA一致。未命中路径在审计前后比较全输入字节SHA及receipt，并用独占创建保留旧缓存。
- `frozen_tools.load_source` 对一次读取的完整bytes求SHA并compile/exec同一份bytes；分析与视觉工具入口/出口核对当前源码SHA。没有把执行旧函数后的新磁盘SHA冒充实际执行身份。
- 主分析源码新增bank `accepted_q` 与初始数组、`risk_pair_index` 与标签逐位核对；配对仍要求初态/tape/标签全等。此处是源码路径复核，未声称已独立审完12个实际cell或全部银行。
- 视觉验证先核state字节SHA，再要求 `receipt.images == state.images`，并要求唯一9个完整env/step/view规范路径和规范state路径。原BR03跨帧/跨环境置换缺口已在此函数中封住。
- 六个冻结源/日志实际SHA均与 `offline_tool_freeze_v3.json` 一致。15项AST seam测试的日志为 `Ran 15 tests ... OK`；我审读了用例，未再次执行这15项测试。它们是绑定函数边界测试，forecast重审主体在fixture中是stub，不证明全9021真实数值正确。

## BR04：缓存命中仍未拒绝stale caller dense

`analyze_v3.py:167` 的命中分支在 `return saved['forecast_audit']` 前没有执行fresh dense加载或传入`data`逐位核对；该核对只在未命中分支的169–170行。测试 `test_stale_caller_dense_array_rejected` 从没有缓存的setUp开始，未覆盖命中分支。

独立反例执行的是当前SHA-bound源码中原函数AST。内存fixture具备合法closed协议、完整37项正确输入SHA、正确schema/path/cell/source身份和现成缓存；磁盘对应dense值为1、传入caller值为999。函数正常返回缓存结果，`fresh_load_calls=0`，没有拒绝stale caller。此次实验没有真实raw/source变更，也没有实际缓存写入。

这不说明已有closed raw或视觉receipt错误；缓存forecast仍绑定其磁盘输入。但它不能支持“命中和未命中均拒绝stale caller”的声明，且主分析后续继续使用caller数组。最小补齐是让两条分支共享fresh dense逐位核对，并加一个“先生成合法缓存，再传入不同dense数组”的实际seam用例。需另存修订源码/执行冻结，保留当前v3和旧测试记录；本侧线不实施修改。

`audit_follow_v3.py` 的 `CLOSED_RECEIPT_PRESENT` 快捷路径只核analysis SHA；它应按字面解释为receipt存在。完整复用资格仍由最终main里的37项核对决定，不能把这个中间日志视为全输入重新验收。

## 视觉与快照证据范围

实际读取的新版视觉receipt记录 `PASS_BOUNDED_NATIVE_CAMERA_EVIDENCE`、42组/378图，执行源SHA与冻结v2工具一致。其日志和receipt是父线程实际执行证据；本次没有独立重算所有native数组、K/五平面或图片SHA。

receipt明确保留baseline数值46/视觉47、joint数值6/视觉7，二者均 `FAIL_EXACT_REPLAY`。图像绑定各自瞬时state；native before/after严格相等不能升级为跨运行或后续轨迹一致。五平面及正深度不代表已录制/验证实际USD near/far clipping，也不证明遮挡、完整连杆或连续接触安全。

另行实际完成的第一freshseed E1审计为18快照×64环境、4564项NumPy算术检查，全部通过；结果见 `ASTRA_FRESH_SNAPSHOT_REVIEW.md/.json`。它保留U侧12/11个单行不可行环境快照及所有正残差，不将算术PASS写成策略安全。最终768窗口目前未闭合；本次不重评分第一seed四格全程失败数。

## 字节绑定

| 文件 | 本次实际SHA256 |
|---|---|
| analyze_v3.py | `8f001efcbd99913424ee2170dbb435e28759acba4c364b041d98af343dd80f16` |
| frozen_tools.py | `cb27bb2d346799d9346509c17488f97fc9178cb53e5d1e19623c504f878d3361` |
| verify_native_visual_v2.py | `26a9c3579e570c4ea5b43937f249228a079c8442821a2c7619126815f63f43ba` |
| audit_follow_v3.py | `d48f10564e47025ca9054e90368b12219652bade805089c62a4ca4d7df9486cf` |
| test_evidence_bindings.py | `4846d68fa02ae43eb4694753d47b8445b9b2ae7a0b5570025073fd82aace132e` |
| evidence_binding_tests.log | `9cc51833dcfc7061dc0df84420f81d379e18b8c9a454b2d836d14b26fb9abae5` |
| offline_tool_freeze_v3.json | `6dc754d1c27ba647b308a9659c6e84dfedfbbddd7e81c4af2e69e8300e196ac8` |
| ANALYSIS_SUPERSESSION.json | `d272e2d7e048cec0ce09745c62666217378cd05e8ce1b3837252c2455da7aa23` |
| native_visual_verification.json | `17c540c5ae8d66012ca8372427dfe5d399d566a7bc13f5ac6a8da987b533446a` |
| native_visual_verification_v2.log | `086c10e25852374ede3db8894df48f7080826abfe75d4f3d5049d9422bbac981` |
| ASTRA_FRESH_SNAPSHOT_REVIEW.py | `938218b37a647a05c61993aabef0975ee8c94b5783c6ef3232a59fe4a2354e23` |
| ASTRA_FRESH_SNAPSHOT_REVIEW.json | `71bc8844cd60f8939e2d229b75626132679c58f076b4c3d1912bb3cda1a47930` |

旧BR报告、旧cache和未获准的物理/硬件结论均保留。**production_promoted=false；hardware_approved=false；不批准安全策略。**
