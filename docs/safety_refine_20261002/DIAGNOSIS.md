# System 0 残余违规诊断（2026-10-02）

观察：既有六臂对压力实验的 22 个 System 0 违规均为标准速度下 F_L–F_R 球层重叠。alpha 锁止后继续接近 7–10 帧，最小 margin 为 -6.410 mm。期望：在同样压力源和速度下，提前制动，阻止此类重叠，同时保留安全区动作。

冻结复现：`PYTHONPATH=src python -m safeduo.eval.research_battery --ckpt artifacts/runs/a31b_refine_cross_20260912/model_last.pt --env-yaml duo_env_a31.yaml --num-envs 32 --duration-s 5 --seeds 0 --amps 0.015 --flows pair_stratified --methods system0 --causal-trace --out <新目录> --headless`。使用 safeduo 环境和 GPU 0；协议记录源文件、配置及权重 SHA256。原失败环境为 env 6、12、24。

| 排序 | 假设 | 区分性预测 | 探针 |
|---|---|---|---|
| 1 | 60 ms 速度预测无法覆盖持续目标驱动的实际制动距离 | solver 残差小，但锁止后 qd 和目标积压仍产生闭合；加大提前量应提前干预并减少后续闭合 | 决策前 q/qd/target、J·qd、J·(target-q)，相同配置单变量对照 |
| 2 | 数值投影未满足近距离安全行 | 首个失败前 solver residual 显著大于 1e-6 | 记录两个机器人的已计算约束残差 |
| 3 | 活跃集漏掉威胁球对 | 违规前没有相关球对行或有效 margin 远离完整球层最小值 | 记录 pre-step pair_id、valid、distance、class |

口径限制：此压力源为状态反馈目标运动，不是纯随机，也不是各方法执行完全相同命令。schedule seed 主要轮换设计单元，不能将重复几何轨迹视为独立统计样本；距离带是 EE 目标中心间距，recede 也包含从 home 向近目标的初始接近。这里只判断球层安全，没有物理接触、夹持任务成功或现实机器人安全证明。
