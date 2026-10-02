# 第二轮诊断记录

## 质心雅可比参考点

`kinematic_diagnostic` / `com_diagnostic` 原策略同条件重复：seed8、l1_full、amp0.015、32环境、5秒，保持 actor checkpoint 与几何。威胁 row7276 = U_L/forearm_link 与 U_L/hand/wrist_3_link。

在 pre-step130，arm-only 及 all-joint/link 预测为 +0.04202 m/s，body velocity 为 -0.01807 m/s，COM-offset/all-joint 预测为 -0.017996 m/s。前者把实际闭合错判为远离。全活跃行的 all-joint/link 绝对误差 P99=0.068087 m/s，all-joint/COM P99=0.00026882 m/s。被忽略手指自由度假设不能解释这条 wrist 行；旋转点独立有限差分回归测试原实现得到0.6，正确结果0.4（RED日志保留）。

仅修正 COM 的 `com_candidate`：原危险位置最小余量 0.7808 →18.1266 mm，但总违规0/32→4/32（env4/15/21 table，env23 self_F）。此候选不可作为安全改善推广。

## 活跃集容量饥饿

`selection_diagnostic` 再次复现4/32。row6156 = F_L/hand/left_ring_1 与 F_R/fr3_link5。记录完整球心以独立重算距离，不使用 min-bucket 代替单一球对。

| pre-step | 距离mm | 中心差分闭合m/s | 估计TTC秒 | 入兜底约束 |
|---|---:|---:|---:|---|
|230|147.736|0.430|0.343|否|
|240|79.889|0.392|0.204|否|
|244|54.454|0.374|0.145|否|
|246|42.003|0.372|0.113|否|
|248|29.849|0.348|0.086|是|

现有选择先以TTC筛候选，再仅按当前绝对间距取32条；较近的静止 self/table 行可挤掉已接近制动时间的球对。候选安全层按 `d - horizon*max(closing,0) - effective_dmin` 选32条，保留跨机 quota。actor的原始32条观测独立保留，避免将几何选集改动混成 actor 输入改动。仍不保证容量足以涵盖所有危险行。

## 条件接触在正cap阶段退出阻尼

env15 row8988 = U_R/hand/right_index_2 对桌。d约5mm且开离时，near-table条件覆写d_min为5mm，struct_exempt使正cap行退出投影，允许下一条大闭合指令；随后速度超过50mm/s，豁免失效时已接近/穿过零距离。新增开关只让条件接触行在正cap阶段继续限闭合速度，永久结构行保留原语义。独立测试：12mm间距、5mm边界时，不能因当前速度低就放行1.2m/s的闭合指令；原实现RED，修正GREEN。

## 风险排序阶段候选

`duo_env_a31_priority_guard.yaml` 组合 COM 几何、150ms风险排序、条件接触阻尼和既有 class lag horizons。保留仅COM的失败证据；后续测试结果以冻结protocol及逐窗口计数为准，不把上述因果探针加算成独立安全样本。

## 长保持揭示目标限位翻转

1.5秒纯随机保持、seed11、amp0.015、32环境、10秒：priority_guard有1/32条F_L–F_R违规，env3最深self_F球重叠32.6786mm。row6193为F_L/hand/left_pinky_1与F_R/fr3_link7；关节6和10已到目标上限。

target_bounds_diagnostic记录每行 `J·projected_delta` 和 `J·实际目标增量`。pre-step513：投影要求开离+1.245mm，后置关节目标clamp抹掉可开离的方向后，实际增量变成闭合−6.439mm；solver residual约1e−9，却并非最终施加增量的可行性。因此不能仅看求解残差声称安全。

修正把 `[q_soft_min−q_target, q_soft_max−q_target]` 与速度箱在Dykstra里一起投影；行权柄下限也按该非对称可行箱计算。原alpha预算方向及最近原请求增量的目标保留，不先裁改命令方向。新增交集也暴露了仅以迭代位移早停的假收敛（u不动时修正项仍变），故限位模式下早停必须同时检查alpha、安全行及箱的可行性。无新限位参数的历史路径保持原逻辑。

独立测试证明：后置clamp可把原已批准开离变成闭合；把饱和方向加入求解则改用可执行方向，满足同一安全不等式。原alpha预算方向、无限位兼容、固定基座及条件接触回归一起验证。最新冻结候选为 `duo_env_a31_bounded_guard.yaml`，结果以重新运行的protocol hash匹配集为准。
