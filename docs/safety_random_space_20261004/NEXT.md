# 强随机范围与覆盖：继续入口

2026-10-04 11:22 Asia/Shanghai。用户继续完善四臂安全策略，强调随机性和范围；保持真实失败，不宣称充分覆盖或安全通过。

## 已完成且网页核验通过

- 公共面板 https://asimfish.github.io/safeduo-dashboard/#scientific ，新增safety_random_space，所有历史科学字段保留。主分支72c5e5e1，data 88135559；完整交付证据见本目录DELIVERY.json。
- 当前发布33登记条件中的27个：18原始条件（9raw完整、9固定128行容量失败）+6新条件分布配对完整+3动态行原分布完整。完整1152窗口、无效576窗口、待完成384窗口；总登记2112相关配对窗口，768条不同命令窗口。不能把窗口当IID可靠性样本。
- 方向/时间/幅度独立CPU随机流；四臂26关节；IID逐步或每臂异步保持1/4/15/30/90/180步；幅度档0.005/0.015/0.025/0.05。局部初态±1.2rad，另全域LHS归一化软限位[0.025,0.975]。手部不随机。
- 全域未筛选192初态仅6个无违规，完整保留；额外5888候选按初始几何与采样顺序选192姿态，不按策略结果选取。全部候选NPZ保留NAS。
- 新种子13447771/27180353/48921161，64环境×15秒×raw/System0，输入/初态/FIFO/软目标限位逐位核验。raw违规192/192；动态行候选115/192（36/38/41）；深度>5mm分别33/29/40。前6步首条随机指令到达前双方各8/5/5失败，仍计分母。静态球距合法不代表物理初态稳定。
- 扩大局部初态逐步IID三种子raw 37/37，动态行候选10/37；其他原分布动态条件仍运行，不能外推。
- 192合法初态候选实测：252/260关节边际分箱、平均每窗口跨度37.17%、总路径114.18rad；四臂跨种子10cm末端格并集2327/2317/2331/2312。非26维联合/Cartesian可达体积覆盖率。
- 六对80mm≥3帧风险暴露80/3/7/6/0/56（F_L-F_R,F_L-U_L,F_L-U_R,F_R-U_L,F_R-U_R,U_L-U_R）。F_R-U_R仍0，其余跨排稀少，不能称覆盖充分。探针球面余量包含豁免球对，官方违规用四通道非豁免定义。
- actor原32行、权重SHA4dc303940d9fa6de2dbdf1e38599e7219ec3e816179a0884920826be46abd0d6不变；生产src/config未改。新增仅评测动态完整临界行（最近full_dmin+10mm，最多512），无行丢弃，原128失败都保留，未推广默认策略。
- 19CPU契约通过；真实本地/线上Chromium141验证11汇总行、24覆盖选择、29新旧曲线、历史数据保留、390px无页面溢出；网页/报告/raw JSON HTTP200。DELIVERY保存全部实证。

## 正在执行，勿重复启动或改冻结文件

- NAS原始数据 /mnt/nas/data/lyf/double_hand/safety_random_space_20261004 。当地NVMe不足，NPZ不要改存本地。
- GPU0父进程1945246（isolated_campaign_v2 + campaign_plan_dynamic.json），尚6个wide_burst/global_burst动态条件，各64×15秒，串行独立进程。不可跳过、缩短、删除或算为安全；原128失败不是0/576。
- 单次终检进程2183632，**/usr/bin/python3** finish_when_done.py --pids1945246，Linux pidfd等进程退出。state见finalization_state.json。模拟环境Python无os.pidfd_open，首个失败见finalization_state_sim_python_failure.json，已换宿主Python实际等待，不重复运行。
- 父进程结束后运行release.py：源/actor快照、33条件分析、19契约、保留历史数据、本地浏览器、明确路径Git正常推送、线上页面/报告/raw+真实浏览器。任何失败保留日志和既有公共结果；不force、不绕hooks、不覆盖别人改动。
- source/hash冻结包括wide_random/wide_runner/wide_runner_dynamic/注册计划/v2驱动/已有测试等，勿编辑；分析、导出、一次终检脚本可修复。新initial bank和feasible6已经结束，不重复。
- finish_when_done.py由宿主3.10执行，release.py由模拟环境3.11执行，browser由宿主3.10执行。浏览器包 /tmp/safeduo_dashboard_browser_tools，Chromium /tmp/safeduo_dashboard_browser。
- 最终结果看finalization_state.json、final_release.log、DELIVERY.json。无自动定时新实验，只完成本次正在运行的有限批次。

## 下一科研边界

1. 完整9原分布动态候选结束后检查实际运动及六对暴露，不只命令随机性；不要把扩大关节角范围当作充分风险覆盖。
2. F_R-U_R零暴露优先补按几何风险带登记的随机初态/方向分层，保持纯随机组独立；必须记录尝试、未暴露、有效暴露和失败，不能重跑挑绿。旧workspace_audit_20260919是v7法兰FK审计，本轮EE点定义不同，不混算。
3. 分解初态物理不稳定与100ms队列/目标积累导致的后续失败。需要真实可稳定停留的初态准入证据，不能事后删掉前6步失败提高主安全率。
4. 前瞻停止仍须具备触发器、真实执行限幅与完整距离约束兼容性。上一轮事后停止7/32→1/32是机制诊断，不是新随机组安全性能；不推广默认策略。
5. 模型缺物体感知/安全球，物理任务与抓持恢复另有并行工作，STATUS.md不要覆盖；现有曲线不证明物体/接触力/连续碰撞/实机安全。

## 发布边界

两个Git工作区当前干净且分支未受保护。发布前fetch+检查+ff-only；data bot只改gates/matrix/runs/runs_5090，保留它。异常冲突停并留证，不覆盖。用户此前明确授权更新公共主板，无需重复索取确认。AnyGent旧token401已恢复过本地上下文，不从零读旧历史。
