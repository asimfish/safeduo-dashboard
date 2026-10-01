# SafeDuo 科研型安全评测报告

日期：2026-10-01。该报告把随机电池、纯均匀随机、四臂配对覆盖和软件回归分成独立证据层；不同 checkpoint、场景几何或时间窗口不会混合计算。

## 1. 固定口径

- 违规：球层非豁免 margin < 0；桌面仍使用已有 near-table 豁免。
- 统计单位：一个并行环境在一个窗口内为一个 episode；正文的 CP95 是一侧 Clopper–Pearson 95% 上置信界。
- 预注册主种子：0、1、2。
- 协议级窗口：32 环境 × 600 控制步（10 秒，约 60 Hz）；`l1_full` 覆盖工作空间漫游，`directed_all` 覆盖 8 类定向冲突场景。
- 机械臂配对：F_L-F_R、F_L-U_L、F_L-U_R、F_R-U_L、F_R-U_R、U_L-U_R；报告同时记录每个配对是否有过告警带暴露。
- 本轮全量仿真使用同一个 checkpoint：`artifacts/runs/a31b_refine_cross_20260912/model_last.pt`。结果不代表换 checkpoint 后的行为。

## 2. 协议级 L1/定向冲突结果

| 配置 | episode | 违规 | 违规率 | CP95 上界 | l1_full | directed_all |
|---|---:|---:|---:|---:|---:|---:|
| 低速 `duo_env_a31_safe_low_speed.yaml`, amp=0.005 | 192 | 8 | 4.17% | 7.39% | 8/96 | 0/96 |
| 标准 `duo_env_a31.yaml`, amp=0.015 | 192 | 43 | 22.40% | 27.91% | 36/96 | 7/96 |

低速组的风险低于标准组，但两个 cell 同时改变了动作幅度和配置，不能把差异归因于单一变量。低速组仍有 8 个违规 episode，不能称为绝对安全。逐种子结果、Wilson 辅助上界和六配对暴露见：

- [低速协议矩阵](scientific_eval_20261001/low_speed.md)
- [标准协议矩阵](scientific_eval_20261001/standard.md)

两组窗口都覆盖了六个配对的跨窗口并集；但单个 episode 没有达到“六个配对均至少 3 个告警步”的严格 exposure 条件，故不能声称每个 episode 都完整暴露了四臂冲突空间。

## 3. 纯均匀随机与安全消融

纯随机使用固定的、可哈希的 IID 均匀关节增量 tape；raw、backstop-only、System 0 在同一个 seed/幅度下读取相同命令。每个 cell 为 32 环境 × 600 步，3 个种子、2 个幅度、3 种方法，共 18 个 cell / 576 episode。

| 方法 | amp=0.005 | amp=0.015 | 合计 | CP95 上界 |
|---|---:|---:|---:|---:|
| raw（无 backstop） | 0/96 | 12/96 | 12/192 | 9.93% |
| backstop-only | 0/96 | 0/96 | 0/192 | 1.55% |
| System 0（checkpoint + clutch） | 0/96 | 0/96 | 0/192 | 1.55% |

纯随机结果支持“防护层能阻断该命令分布下的球层违规”：amp=0.015 时 raw 的 12 个违规被 backstop-only 消除。System 0 与 backstop-only 在这组固定 tape 上的违规结果完全相同，当前证据不能证明策略网络比安全防护层额外降低了碰撞风险。

纯随机命令没有进入任一机械臂配对的 80 mm exposure 带，因此这组结果主要是安全消融，不是四臂冲突空间覆盖实验；四臂覆盖仍由 `directed_all` 和工作空间审计承担。每个失败 episode 的命令哈希、首次违规类别、最小 margin、关节和末端轨迹均已保存：

- [纯随机完整证据包](scientific_eval_20261001.json)

## 4. 已保存的可复核证据

每个实验目录包含：

- `summary.json`：窗口级统计和分层统计；
- `window_*.json` 或 `cell_*.json`：seed、flow、幅度和方法元数据；
- `episodes.json` / `cell_*.json`：逐 episode 违规、damaging、类别、最小 margin、配对告警、干预率、命令哈希；
- `cell_*.npz`：关节、末端、原始命令、执行命令、margin、alpha 和 backstop 激活轨迹；
- `protocol.json`：checkpoint SHA256、解析后的 YAML、代码指纹、Python/Torch 版本、步长和完成状态。

汇总器 [scientific_matrix.py](scientific_matrix.py) 会拒绝混合不同 checkpoint、几何、并行环境数、窗口长度或阈值的结果，并检查设计矩阵是否缺 cell。纯随机评测器是 [research_battery.py](research_battery.py)。

## 5. 软件验证

- 新增汇总器、纯随机 tape 回归和配置回归：`14 passed`。
- 本轮受控回归组合：`82 passed`。
- 在 safeduo 环境中运行 `pytest -q tests src/safeduo/tests`：`1616 passed, 83 failed, 3 skipped, 10 errors`。失败没有被隐藏；主要是 IsaacLab runtime closure 漂移、缺少可选 `proxsuite`、缺少旧 renderer 路径、旧 golden/skill fixture，以及系统 Python 的 NumPy/Matplotlib ABI 问题。它们尚未构成“全仓库绿灯”，因此本报告不声称全套测试通过。
- `duo_env_v5.yaml` 增加显式 `safety.target_guard.enabled: false`，只是把默认关闭状态写入配置以通过 v5/v6 safety-only diff contract，不改变当前实验行为。

## 6. 限制和下一轮门槛

1. 当前矩阵是球层安全评测，没有把物体抓取成功率、物体放置、相机三视角感知和任务完成率作为主指标；这些必须作为独立 task layer 加入，不能从 sphere margin 推断。
2. 纯随机没有产生配对告警带暴露，说明均匀关节随机并不覆盖四臂危险空间；需要基于 workspace/配对距离的分层采样，而不是继续增加同分布样本。
3. Isaac Sim 启动时持续报告 RTX 5090 CUDA bad state 并回退到 `cuda:0` 路径；本轮运行完成，但在重新宣称结果前应重启 GPU 状态并复跑至少一个固定 cell。
4. 先前 target-rebase 反事实为 27/32 违规、CP95 上界约 93.63%，所以默认保持关闭；该结果不能作为 System 0 安全性证据。
5. 下一轮验收应固定同一个动作 tape，分别改变一个因素（amp、backstop、clutch、target guard），并加上按六配对分层的定向采样和真实物体任务成功判据。

## 历史恢复状态

本轮沿用了先前会话已完成的相机、夹持、四臂 workspace 审计、A31 随机电池和 target-rebase 反事实结果；未重复旧实验。尝试从 Anygent 只读接口增量读取时接口返回 401，因此本报告以工作区已有 artifact 和当前可复核输出为准，没有把无法读取的历史消息当作新证据。
