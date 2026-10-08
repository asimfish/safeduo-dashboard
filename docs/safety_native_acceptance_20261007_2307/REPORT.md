# 完整原生状态配对实验与验收

完成512/512方法窗口，128个不同配对初态，四方法×两独立初态批次。每窗口960控制步、1920物理子步；候选结论 **REJECTED**，父分析与GPT‑6 Astra xhigh独立验收一致。

| 方法 | strict/128 | deep/128 | 原生法向超0.1N/128 | 法向峰值N | 实际平均q路径 | 四臂均运动比例 |
|---|---:|---:|---:|---:|---:|---:|
| joint_reference | 20 | 11 | 4 | 60.2925491 | 12.3521416 | 0.466153971 |
| zero_inclusive | 7 | 3 | 3 | 28.5428829 | 3.12640871 | 0.401961263 |
| box_admission | 7 | 3 | 3 | 33.7024002 | 3.10075654 | 0.383447266 |
| adaptive_joint | 12 | 8 | 4 | 181.923981 | 3.6643355 | 0.421386719 |

预登记strict为原生float32间隙<0m，deep为<-0.005m，无评分epsilon。另要求候选实际初态全9021行无负值、无新增配对失败、整体与七分层路径至少保留零包含基线的90%、四臂运动比例至少90%、26关节各平均范围≥0.001rad，以及完整原生初态严格配对。手部全伙伴聚合法向0.1N为独立资格门槛，同手自碰全部保留。

候选拒绝原因：

- candidate strict represented-sphere violations
- candidate deep represented-sphere violations
- new paired strict failure vs zero_inclusive
- candidate actual initialized geometry negative before first physics step
- raw native hand normal contact exceeds registered diagnostic .1N

实际运动路径比 1.1720590113781726；四臂运动比例比 1.0483266859676472。原球模型判定 REJECTED；实际初始化几何判定 REJECTED。

- zero_inclusive → adaptive_joint：{"rescued": 3, "new_failure": 8, "both_failure": 4, "neither_failure": 113}。
- zero_inclusive → box_admission：{"rescued": 1, "new_failure": 1, "both_failure": 6, "neither_failure": 120}。
- box_admission → adaptive_joint：{"rescued": 3, "new_failure": 8, "both_failure": 4, "neither_failure": 113}。
- joint_reference → adaptive_joint：{"rescued": 11, "new_failure": 3, "both_failure": 9, "neither_failure": 105}。

两批独立随机初态、128个不同配对初态、四方法512个完整窗口；每窗口960控制步/1920物理子步；七个风险分层。实际21视角原图4494张。初态经过旧球模型和零指令稳定性条件筛选；质量/摩擦/增益/布局/FIFO6固定，未进行动力学域随机化。

全部64环境逐控制步保存三个原生边界；逐物理子步记录手部全伙伴法向/净力/计数、全部关节与实际位置目标。所有视图绑定同次运行，手部逐图核验全部64环境的关节、根状态、连杆、执行器目标和物理时钟逐字节不变。214组×21视角，共4494张原始1280×720图；未裁切、缩放或改图。0.06m连杆中心包围球在实际USD六个视锥平面均有≥0.05m余量；这不证明所有网格或无遮挡。

公共U拇指初态/目标0.35rad在四方法一致，受控臂、演员、增益、自碰、原球豁免及FIFO6不变。初态资格采样仍在旧手部默认条件下完成，未据策略结局筛选或重采；构造阶段接触未观测。开发64窗口原生接触超限64/64→0/64只是开发证据；实际约3.6mrad追踪误差保留，正式结论只按新随机完整窗口计算。

联合可行性机制只在当前联合LP返回不可行status2时扩宽到原可达界，并重新调用原30遍投影；LP见证不替代控制，未来队列始终UNKNOWN。当前残差与LP分类属于线性诊断，不能代替实际几何、接触或未来非线性安全。

两批当前LP不可行2071个环境步、扩宽1940个环境步，最终仍有17376个当前线性残差超限环境步。[8个新增失败逐案例诊断](https://raw.githubusercontent.com/asimfish/safeduo-dashboard/19e37d1172ee105b4f99416b81cc48aa45285f42/safety_native_acceptance_20261007_2307/ADVERSE_CASE_DIAGNOSIS.md)；时间上的先后与相关性不等于已证明因果。

失败历史全部保留：短磁带协议错误（物理子进程0、监督1）、无子进程资源拒绝、旧25GiB等待器明确关闭actual−15/outer241。正式资源门槛预登记20GiB空闲、12GiB自身上限、6GiB运行空闲底线；实际退出与协议闭合分别检查，Kit退出0不代表成功。

验收覆盖有限模拟中的原球模型、原生手部全伙伴聚合法向接触和实际运动量。候选采用须同时满足预登记门槛；整机、摩擦、构造阶段、未来非线性动态与硬件安全均未认证。

公开资源保存全部原图/状态、512案例、完整960/1920曲线、原始路径/SHA映射、验收登记与源码。全9021距离/Jacobian/预测及完整原生流保存在NAS原始实验与归档；公开曲线不替代未公开的大型原始文件。

[固定资源清单](https://raw.githubusercontent.com/asimfish/safeduo-dashboard/19e37d1172ee105b4f99416b81cc48aa45285f42/safety_native_acceptance_20261007_2307/asset_manifest.json) · [全部512案例](https://raw.githubusercontent.com/asimfish/safeduo-dashboard/19e37d1172ee105b4f99416b81cc48aa45285f42/safety_native_acceptance_20261007_2307/case_table.json) · [实际随机覆盖](https://raw.githubusercontent.com/asimfish/safeduo-dashboard/19e37d1172ee105b4f99416b81cc48aa45285f42/safety_native_acceptance_20261007_2307/COVERAGE_RESULT.json) · [独立验收](https://raw.githubusercontent.com/asimfish/safeduo-dashboard/19e37d1172ee105b4f99416b81cc48aa45285f42/safety_native_acceptance_20261007_2307/ASTRA_ACCEPTANCE_REAL.json) · [完整原始文件SHA清单](https://raw.githubusercontent.com/asimfish/safeduo-dashboard/19e37d1172ee105b4f99416b81cc48aa45285f42/safety_native_acceptance_20261007_2307/raw_experiment_manifest.json)

固定源与接收规范：

- criteria SHA `01a5b45d1ada7635946f15c1b2e0cee141e8128266bdbed423844cf833c054a3`
- registration SHA `55f4dfc4c648f44b6d42218ccbd9f94591017b543f8c9720a2a9651e373490cd`
- camera SHA `8b9012f50a6ce61476e10af27034f77eb2ce51dd2737a9db61dfb49a36df58d9`
- initial gate SHA `e7649098e7b5a054654b22f63208e7dfadf64273cbb799e31eaa689037c91bc9`
