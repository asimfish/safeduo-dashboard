# ASTRA：BR04 v4 最小修复复核

2026-10-05；**PASS_BOUNDED_BINDING_REPAIR**。BR04 的缓存命中stale caller缺口在本次v4修复中已关闭；旧v3/BR04审查及旧缓存保留。此状态只验收本次离线绑定修复，不验收最终768窗口或物理策略。

实际对照了v3→v4 source diff、测试diff、16项实际PASS日志、冻结清单、supersession记录及loader。`fresh_data=load(cell_001.npz)` 和全key/逐数组 `array_equal` 已移动至缓存分支之前，因此命中、未命中共享同一门。额外变化是v4缓存目录/schema及对应工具文件名，没有改变forecast审计算术；独立AST比较 `forecast_audit` 完全相同，`load_source` 也与旧单次bytes SHA/compile/exec函数完全相同。

新 `test_cached_stale_caller_dense_array_rejected` 先调用实际缓存函数生成合法cache，再把sentinel改为999，要求AssertionError且audit调用数仍为1。实际日志为16 tests / OK。我审读并核对其SHA，没有重复运行父线程16项测试。

另独立执行了当前v4实际函数AST的三个内存seam：合法cache+相同dense正常返回且fresh load调用1次；合法cache+999被拒绝；无cache+999也被拒绝。fixture具备完整37项输入和规范closed身份，全部只在内存中，不写真实cache/raw或测试临时文件。这是函数边界证据，不是full9021 forecast重算。

冻结清单中的六个源/日志实际SHA全部一致。新版worker仍负责全部closed cell逐个重审；本侧线没有运行该worker或替它填最终结果。旧v3 PASS receipt不会自动成为v4验收，最终证据须绑定v4实际执行结果。

| 文件 | 实际SHA256 |
|---|---|
| analyze_v4.py | `76b04971b40d9a439ba4faf30da012bf5eb457db6ba71984cd5c6b5206c81255` |
| frozen_tools_v4.py | `bbf59b40a166524339a0fb35b19361e7b48027a5e105d344c08fa3e10934bc79` |
| audit_follow_v4.py | `69b3c63df5ffbd5ebeb64f8182dc5641f62203746d56858052f0f2392ee16520` |
| verify_native_visual_v2.py | `26a9c3579e570c4ea5b43937f249228a079c8442821a2c7619126815f63f43ba` |
| test_evidence_bindings_v4.py | `e30e7cfbc7bb4aa98b9fd957f24c8633a36aa8cfed28996f5a0ed533373b497f` |
| evidence_binding_tests_v4.log | `e99f065e2c3025c117aacc10ab9a5945005a42e793945f6a0915e588d927e24f` |
| offline_tool_freeze_v4.json | `5a8133e916d34a4a20e3a6a37a048bf125bb0511d51b996a73c3de794a5696b2` |
| ANALYSIS_SUPERSESSION_V4.json | `b384d58832e95984f1e010ee8e0a7c4829bf0be83c2a35492a35b3b0b2699a07` |

无该最小修复范围内的新增阻断发现。未改控制/相机/银行/tape/physics/生产源码；**production_promoted=false，hardware_approved=false**。最终768尚未审完，不作全轮PASS或安全批准。
