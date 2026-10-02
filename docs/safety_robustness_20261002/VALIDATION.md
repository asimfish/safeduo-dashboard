# 交付核验

候选为 `duo_env_a31_bounded_guard.yaml`。512个 System 0 保护窗口全部完成，非豁免球层违规0，初态违规0。结果包含不同且相关的分布，不使用独立样本安全置信界。

- 相关CPU回归78项通过，见 `delivery_tests.log`；三个原失效的RED日志一并保留。
- 六个最终实验清单的 geometry/backstop/sphere_distance/duo_env 源码hash一致，并与交付源码快照逐文件核对。构建报告同时核对冻结checkpoint、初态、配对命令SHA和无保护控制复用的配置前提。
- 所有公开Python文件通过语法解析；页面内JavaScript通过 `node --check`；HTML没有重复ID；`git diff --check`通过。有限凭据模式扫描未命中，这不等于全面安全审计。
- Chromium 141真实浏览器验证桌面1280×900和手机390×844：8行随机结果、4行回归结果、3个最新结果卡片、4种曲线切换、SVG有限数值、手机无页面横向溢出、无JavaScript页面错误。见 `browser_check.log` 和 `check_browser.py`。
- 本地核验使用冻结预览数据；发布后独立浏览器从真实网页及其默认远程数据接口加载，重复检查通过。在线记录见 `live_browser_check.log`，实现提交及数据提交见 `DELIVERY.json`。

公开SVG和日志副本仅清除行尾空白以通过仓库格式检查，数值与图形内容保持不变；原始本地日志保留。

复核入口：`build_report.py` 生成研究汇总，`make_plots.py` 生成图表，`prepare_dashboard.py --preview-json <文件>` 生成预览，`check_browser.py` 验证本地面板；发布后 `check_browser.py --live-url https://asimfish.github.io/safeduo-dashboard/#scientific` 验证真实网页加载的候选和完整统计。

本轮使用 headless 数值仿真。图形层GPU枚举警告尚未消除；不将本轮曲线包装成新视频，不将数值结果当作实物接触安全或物体任务完成的证据。后续优先覆盖随机初始姿态、动力学与延迟扰动，并单独测真实物体夹持和四臂任务成功。
