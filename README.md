# safeduo-dashboard

SafeDuo 项目计划面板（GitHub Pages）：<https://asimfish.github.io/safeduo-dashboard/>

- `index.html`：面板页。北极星与目标进度 / 工作项路线图 R1–R35 / 四门实验矩阵 / 任务族与物体资产（R34 提案）/ 待 owner 决策 / 判决 / 工作队列 / 文献启发 / 变更日志 / 维护契约；服务器实时快照（bjxy_5090）。
- `data` 分支：JSON 快照，页面用 `raw.githubusercontent.com` 读取。
  - 自动：`runs.json`（`collect_remote.py` 在服务器上采集）、`gates.json`（`build_gates.py` 从 `artifacts/clutch/*/cell_*.json` 生成）、`matrix.json`（`build_matrix.py` = plan ⨝ gates ⨝ runs）
  - 手工（agent 里程碑更新）：`status.json`、`roadmap.json`、`tasks.json`、`plan.json`
- `docs/`：设计文档（R34 工业协作装配场景与资产方案、SAI/TRACE 文献启发）。
- `MASTER_REPORT.html`：SafeDuo 活文档镜像（源在 `safeduo/paper/`，由实验线维护，脚本自动同步）。
- `gallery/`：任务视频关键帧缩略图。
- 视频库：独立仓库 [asimfish/safeduo-media](https://github.com/asimfish/safeduo-media)（GitHub Pages 托管 mp4 + `index.json`）；`sync_media.sh` 从 `bjxy_5090:~/Code/safeduo_media` 与 `~/safeduo/artifacts/viz/a8_v4demo`（近 3 天）拉取新片，`build_media_index.py` 转码到 ≤960 px / CRF 29（每部 ~0.5–1 MB）并生成索引，随 `update_dashboard.sh` 每 10 分钟自动运行。

## 更新

```bash
bash update_dashboard.sh            # 采集服务器 + 重建矩阵 + 推送 data；MASTER 变化则同步
bash update_dashboard.sh --no-server # 只重建/推送手工内容
```

Mac launchd `com.liyufeng.safeduo-dashboard` 每 10 分钟运行一次。

## agent 协作契约

见面板第 10 节。要点：新增实验先写 `plan.json` 再发车；判决必须带证据路径；数字只来自 `gates.json`（禁止手改）；设计线只加新文件、不与实验线同时改同一文件；含中文文件用 shell heredoc 写入并验证编码。
