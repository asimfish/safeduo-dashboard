# 随机初态与执行端延迟评测

**最终候选尚未通过安全回归，未推广。** 严格100ms FIFO延迟仍有违规，原初态漫游新增桌面违规；旧bounded_guard保持冻结参考。CPU回归93项通过只证明被断言的软件行为，不替代这些仿真负结果。

所有配对初态SHA、源命令SHA和初态违规标记逐窗口完全一致；延迟队列按实测发送目标验证FIFO序列。初态碰撞不拒绝，避碰计数只取初态无违规窗口。

|组/方法|初态±rad|延迟ms|保持/时长s|初态违规/全部|无违规初态后的违规|最小臂/手球距mm|实际关节路径rad|
|---|---:|---:|---|---:|---:|---:|---:|
|pilot_initial_delay · System 0|0.3|0.0|1.50/10.0|0/32|1/32|-9.024|81.645|
|pilot_initial_delay · System 0|0.3|100.0|1.50/10.0|0/32|11/32|-30.313|79.121|
|latency_candidate_pilot · System 0|0.3|0.0|1.50/10.0|0/32|1/32|-4.177|86.280|
|latency_candidate_pilot · System 0|0.3|100.0|1.50/10.0|0/32|11/32|-27.348|86.606|
|stored_verified_pilot · System 0|0.3|0.0|1.50/10.0|0/32|0/32|10.690|86.385|
|stored_verified_pilot · System 0|0.3|100.0|1.50/10.0|0/32|11/32|-26.236|85.767|
|stored_verified_matrix · 无保护|0.1|0.0|0.50/5.0|0/32|30/32|-79.611|48.560|
|stored_verified_matrix · 仅解析兜底|0.1|0.0|0.50/5.0|0/32|0/32|13.036|47.506|
|stored_verified_matrix · System 0|0.1|0.0|0.50/5.0|0/32|0/32|13.728|46.686|
|stored_verified_matrix · 无保护|0.1|100.0|0.50/5.0|0/32|30/32|-79.548|47.431|
|stored_verified_matrix · 仅解析兜底|0.1|100.0|0.50/5.0|0/32|1/32|1.765|46.735|
|stored_verified_matrix · System 0|0.1|100.0|0.50/5.0|0/32|1/32|1.764|45.807|
|stored_verified_matrix · 无保护|0.3|0.0|0.50/5.0|0/32|30/32|-82.944|48.369|
|stored_verified_matrix · 仅解析兜底|0.3|0.0|0.50/5.0|0/32|0/32|8.385|47.424|
|stored_verified_matrix · System 0|0.3|0.0|0.50/5.0|0/32|0/32|10.739|46.490|
|stored_verified_matrix · 无保护|0.3|100.0|0.50/5.0|0/32|30/32|-68.363|47.329|
|stored_verified_matrix · 仅解析兜底|0.3|100.0|0.50/5.0|0/32|1/32|5.572|46.661|
|stored_verified_matrix · System 0|0.3|100.0|0.50/5.0|0/32|1/32|5.571|44.968|
|pending_final_pilot · System 0|0.3|0.0|1.50/10.0|0/32|0/32|10.690|86.385|
|pending_final_pilot · System 0|0.3|100.0|1.50/10.0|0/32|11/32|-25.244|85.612|
|pending_final_matrix · 仅解析兜底|0.1|0.0|0.50/5.0|0/32|0/32|13.045|47.504|
|pending_final_matrix · System 0|0.1|0.0|0.50/5.0|0/32|0/32|13.728|46.686|
|pending_final_matrix · 仅解析兜底|0.1|100.0|0.50/5.0|0/32|1/32|1.765|46.735|
|pending_final_matrix · System 0|0.1|100.0|0.50/5.0|0/32|1/32|1.765|45.809|
|pending_final_matrix · 仅解析兜底|0.3|0.0|0.50/5.0|0/32|0/32|8.386|47.424|
|pending_final_matrix · System 0|0.3|0.0|0.50/5.0|0/32|0/32|10.738|46.490|
|pending_final_matrix · 仅解析兜底|0.3|100.0|0.50/5.0|0/32|1/32|5.574|46.660|
|pending_final_matrix · System 0|0.3|100.0|0.50/5.0|0/32|1/32|5.575|44.979|
|pending_nominal_long · System 0|0.0|0.0|1.50/10.0|0/32|0/32|10.377|82.018|
|pending_nominal_l1 · System 0|0.0|0.0|漫游/5.0|0/32|1/32|11.999|29.926|

## 对比审计

无保护矩阵控制来自stored_verified_matrix四个已完成cell，因新历史仅参与RawShim绕过的安全投影而复用；初态和完整命令SHA一致，不增加控制样本。
原失败重播的新失败/修复环境列表：[{"delay_steps": 0, "new_failures": [], "corrected_failures": [31]}, {"delay_steps": 6, "new_failures": [6, 13, 15, 20, 25], "corrected_failures": [0, 11, 18, 23, 27]}]
最终候选保护结果（异质相关分组，非独立安全置信证明）：{"episodes": 384, "initial_violations": 0, "initially_safe": 384, "violations": 16, "safe_initial_violations": 16, "safe_initial_damaging": 9, "by_class": {"cross": 0, "self_F": 4, "self_U": 4, "table": 12}, "min_arm_hand_mm": -25.243714451789856, "mean_joint_range": 0.1853569929371588, "mean_joint_path_rad": 54.68560283879439, "output_ratio": 0.9166264538378271, "pair_union_exposed": 6}

## 范围与限制

局部关节初态扰动，不是全空间均匀采样；初态碰撞未筛除，预防结果仅取无违规初态子集。
手指、动力学、传感时间和真实物体任务固定或未测；执行端目标延迟不等于感知延迟。
球距在每个约16.67ms控制周期结束时采样，未证明两个采样点之间不存在短暂碰撞。
窗口和保持动作相关，不提供独立样本安全置信界；实际运动量和输出比例不是任务成功率。

源码/协议见results.json与protocols，逐窗口配对见paired.csv。计数不与上一轮512窗口混算；重复探针和修复重播不增加独立样本数。
