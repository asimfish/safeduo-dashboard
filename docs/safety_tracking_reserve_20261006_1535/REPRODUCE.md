# 跟踪余量实验复现

本研究是独立的新随机比较，旧封存结果只用于开发背景。先读 RANDOM_EXPERIMENT_DESIGN.json、NUMERIC_REGISTRATION.json 和三个 plans/holdout_*_plan.json；它们保留完整实际参数、演员、生产源码和研究依赖 SHA。

三个新银行位于 NAS 同名实验目录 banks/901654451、banks/1966909750、banks/1854992723。生成采用原 risk_bank.py、全几何检验及60步原始零输入资格检查，不查看候选策略结果，不重新挑选表现良好的窗口。所有几何候选、拒绝和稳定性记录已保留。

在具有原 Isaac Lab/Sim、GPU/驱动和计划 env 的环境，以一个全新输出目录运行每个计划的实际 argv，并注入该 job/env。每个数值进程仅运行一个64环境条件，完整960步，不改原阈值/豁免、演员、速率盒或真实六步 FIFO。新控制只改变参考范围；不清空或重排待执行目标。具体执行程序和参数取 canonical campaign.json，不能把复现参数当作已经恢复的实际进程参数。

策略代码和数学冻结后再执行新窗口。基准 joint_reference 保持0.050rad；tight_reference 固定0.010rad；delay_reserve按注册的当前/真实待执行目标/速度端点估计选择0.010..050rad。风险是启发式，未证明动力学保守性或物理可恢复性。

离线复算请将本页的 analyze.py、dense_audit.py、RANDOM_EXPERIMENT_DESIGN.json、plans/ 放入原 artifacts 下的新相邻审计目录；保留 plans 绑定的原绝对生产/研究/银行路径以及 sibling safety_random_space_20261004/coverage_metrics.py。analyze.py holdout 每次读取真实cell、输入、完整行分块和FIFO，不接受已计算结果作输入。execute_analysis.py 使用两线程执行同一 frozen inspect，再保持原串行汇总顺序，且验证前后源码绑定。默认分析使用注册的系统 Python/NumPy，不能擅自切换浮点执行语义。

LP审计使用预登记的既有 /home/liyufeng/miniforge3/envs/safeduo/bin/python（Python3.11.15，NumPy1.26.0，SciPy1.15.3）。执行 failure_audit.py 和 audit_fixed_feasibility.py，固定步骤0/60/62/66/71/75/180/480/959，全环境全方法保留。LP以保存float32集合转换为float64计算，归一化行后HiGHS容差1e-9；容差不应用到严格物理计分。判定只针对保存的选中线性集合，不是完整非线性可达性证明。

真实相机执行严格遵守 visual_plan.json，使用CUDA0和原图像尺寸。42个固定组来自七个预定初态槽×三个时间点×两种主要方法，额外拍摄本次相机轨迹各分层最早负几何。每组九视角、原PNG、实际状态JSON以及渲染前后全64环境native q/qd/root/root_vel，均可以下载。先核验native不变性、实际USD六平面、自己的目标/排队状态，再独立检验是否数值精确复现；不能假定相机轨迹等于数值轨迹。

完整 NAS archive：/mnt/nas/data/lyf/double_hand/safety_tracking_reserve_20261006_1535 。原始大体积逐帧数据留在该封存档案。公开固定证据commit保留原图像/状态、输入、资格银行、首次失败快照及完整逐案例gzip；面板保留可复现源码、计划、独立审查和实际结果。不重复累计相机演示、先前研究或失败重试到新随机分母。

这是一份软件/仿真实验复现说明，不授权实机或生产部署。
