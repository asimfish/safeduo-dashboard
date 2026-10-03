# 本轮核验记录

本轮交付的是独立启动工具和限幅停止的事后诊断证据；不是生产安全策略通过。

|核验项|证据|结果|
|---|---|---|
|独立进程|campaign.json中的4个不同PID，每个child协议仅一个cell且complete|通过|
|初态和命令|4组q_initial与完整cmd逐位核对，无初态违规|通过|
|执行端队列|全部送达目标逐位等于6步前实际发送目标；禁止抢占|通过|
|输出约束|所有26受控关节、32环境、600步的实际目标增量和软限位；容差5e-7rad|通过|
|全环境失败|NPZ四通道非豁免球距与逐episode JSON核对；全部128窗口CSV|通过：7/32、1/32、1/32、1/32|
|观测前缀|q/qd/球心/目标/命令/待执行输入|17/18逐位一致；12步组env30最大q差0.110mrad，明确降级|
|当前控制复现旧控制|q/qd/exec/球距/目标/球心整段|逐位一致|
|生产配置和权重|各组完整源快照、解析配置、协调器、backstop及actor SHA|一致；与上轮仅记录的可视化源差异|
|CPU工具契约|固定采样、限幅移动、软限位、真实FIFO、真实子进程隔离、拒绝覆盖和失败保留|17通过|
|网页展示|Chromium 141，4行主表、18条停止响应、8张新曲线、全部旧曲线/表、390px布局|本地通过；线上结果见DELIVERY.json|

三个限幅探针全部只剩未干预env31重叠（最小球距约-0.101mm）；没有新增失败环境。原始控制最小球距约-28.128mm。六个被干预环境没有测到球层重叠，触发时刻来自历史失败，不属于前瞻安全性能。12步组env12目标需要24次限幅更新才能完全达到，不能把首条停止消息到达解释为机器人已经静止。

执行入口：

```bash
/home/liyufeng/miniforge3/envs/safeduo/bin/python artifacts/safety_bounded_stop_20261004/isolated_campaign.py --plan artifacts/safety_bounded_stop_20261004/campaign_plan.json --out NEW_OUTPUT_DIRECTORY
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/liyufeng/miniforge3/envs/safeduo/bin/python -m pytest -q artifacts/safety_bounded_stop_20261004/test_bounded_stop.py artifacts/safety_bounded_stop_20261004/test_isolated_campaign.py tests/test_eval_perturbations.py
/home/liyufeng/miniforge3/envs/safeduo/bin/python artifacts/safety_bounded_stop_20261004/analyze.py
PLAYWRIGHT_BROWSERS_PATH=/tmp/safeduo_dashboard_browser PYTHONPATH=/tmp/safeduo_dashboard_browser_tools /usr/bin/python3 artifacts/safety_bounded_stop_20261004/check_browser.py
```

第一条入口已运行完整4个条件，示例NEW_OUTPUT_DIRECTORY须替换为新的空目录；不重复运行来扩充统计。旧连续cell顺序电池没有改成默认新行为，本轮以及后续配对campaign可使用新的独立进程入口。

浏览器工具原先用Python3.10构建，首次误用仿真Python3.11导致greenlet扩展导入失败，保留browser_runtime_mismatch.log；改用现有/usr/bin/python3后通过，没有安装或更换依赖。该环境错误与仿真结果无关。

原始NPZ在本地保留；网页发布协议、完整episode JSON、聚合结果、128行环境CSV和逐帧曲线。CPU工具契约不是机器人安全测试；未重跑未改动安全核心的98项旧检查来重复计数。没有新增视频、接触力、物体任务、连续碰撞、实机或全部四臂操作空间证明。当前完整距离投影与停止目标兼容性、自动提前触发和新种子泛化尚未验证。

线上核验通过：页面/报告/数据均HTTP200，Chromium在真实线上数据下验证4行主表、18条响应、8张新曲线、全部旧图表和390px布局，无页面异常。发布版本、时间、源码核验和剩余限制见DELIVERY.json。Pages构建API返回404，未声称通过远程CI；使用真实页面、报告、数据和浏览器核验作为发布证据。
