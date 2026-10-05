# 复核与独立重放

原始环境为 `/home/liyufeng/safeduo`，Python在 `/home/liyufeng/miniforge3/envs/safeduo/bin/python`，IsaacLab在 `/home/liyufeng/safeduo_isaaclab`。本轮完整NAS根目录为 `/mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005`，元数据、代码及分析在 `/home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005`。

公开文件供审阅，运行还需要注册计划中的原始项目、旧研究辅助模块、actor和银行。`holdout_plan_v2.json`逐项列出argv、环境变量、预期参数与source/actor哈希；只复制公开runner到任意目录不构成可运行复现环境。

已有输出不得覆盖。原始 `execute_campaigns_v2.py` / `execute_visual.py` 的输出路径已经使用，封存后不要再次指向这些路径。新重放使用独立输出目录：

```bash
source /home/liyufeng/safeduo_setup/env.sh
export LD_PRELOAD=/home/liyufeng/miniforge3/envs/safeduo/lib/libstdc++.so.6
cd /home/liyufeng/safeduo
python artifacts/safety_joint_guard_20261005_1005/replay_registered_cell.py --id joint_guard_1701627244 --out /mnt/nas/data/lyf/double_hand/safeduo_joint_guard_independent_replay
```

该命令仅在全新目录存在性检查和所有冻结生产/研究/actor哈希检查通过后启动一个新OS进程。保留原960步、64环境、6步FIFO、相同银行和输入种子，产物记作后续重放，不增加本轮独立初态或输入案例。未执行的重放命令不是新实验结果；不得要求未来重放必然逐位一致。

CPU合同检查：

```bash
cd /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005
python -m unittest test_full_guard.py test_projection_diagnostics.py test_capacity_path.py
```

在已有原始数据上重算（分析会生成独立核验文件，不会修改原始模拟张量；封存后应在复制的元数据工作目录中执行并调整只读输入定位）：

```bash
python -B frozen_tools_v4.py analysis
python -B frozen_tools_v4.py visual
```

全量有限值、原始几何端点、关键行并集成员、六步FIFO及两份独立待执行历史必须一致。缓存仅接受同实际编译执行的分析source SHA、37项完整消费清单且每个消费文件SHA重新核对的已闭合窗口。J全程由生产者检查，离线只存九个selected J快照，不得声称离线重算全960帧9021行J。

Camera-enabled视觉运行是独立进程。42组378张图各绑定自身实测state和原生before/after文件，不能把跨运行不一致的视觉帧当作数值主运行同帧。实际USD/K和五平面覆盖可复核；实际near/far未独立保存。瞬时native不变不等价于未来轨迹无扰动。

浏览器检查需要已安装的Playwright与Chromium。只读检查公开面板：

```bash
python check_panel_v2.py --url https://asimfish.github.io/safeduo-dashboard/docs/safety_joint_guard_20261005_1005/ --live
```

检查全部配对案例×四类×四方法曲线、120筛选、四张边际热图、378张图片以及126个state/native HTTP SHA。检查程序的截图与回执写入其所在元数据目录，封存后需另设新的输出工作目录。

`REPORT.md`说明实验定义、所有新增反例、实际暴露不足、正投影残差与失败版本。软件停止边界通过、数学记录重算通过、窗口球层无违规、搬运任务完整门和硬件批准是独立结论；本批不提供硬件批准。

证据工具另有16项绑定回归检查：`python -B -m unittest test_evidence_bindings_v4.py`。它们使用实际函数和独立临时元数据/字节夹具，不新增物理窗口。旧v2工具和三个旧缓存审计保留于独立位置，v4重新审计，未将旧结果升级成新版本验收。
