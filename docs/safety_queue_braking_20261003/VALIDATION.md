# 队列与固定目标制动验证记录

## 仿真与独立数据审计

所有生产源码、完整解析配置、协调器配置与上一轮critical-only相同；actor SHA不变。七个延迟控制cell、两个停止探针cell均complete，32环境、10秒，无初态违规，不筛选初态。纯随机命令与26关节初态逐位相同。逐帧送达目标与声明FIFO逐位一致，停止消息未抢占队列。

新进程首窗口主比较：0ms 0/32，50ms 1/32，100ms 7/32。100ms首窗口的q、pre_qd、exec、官方球距、发送目标、球心与上一轮逐位一致。连续窗口第三cell100ms变为6/32，50ms第二cell在不同顺序中0/32或1/32；存在运行历史影响，具体隐藏状态尚未定位，不视为策略改善。

停止时刻由既有第三cell失败步减12固定，原触发表不变，属于事后诊断。对冷启动未干预控制，stored六条、measured五条观测前缀一致；measured env30最大q差0.110mrad。两停止方式之间五条q/qd/球心/目标/命令和停止前FIFO输入前缀一致。未记录全部隐藏物理求解器状态。

诊断中受干预六条轨迹：stored到达后两条重叠，measured零条。全32环境：stored仍3/32、measured仍1/32，未干预env31失败保留。不能将事后诊断计入预防通过率。实测q固定目标到达后的可观测逆目标超调0.002923–0.040129rad；控制周期末采样，不证明周期内无碰撞。

运行命令：

```bash
/home/liyufeng/miniforge3/envs/safeduo/bin/python artifacts/safety_queue_braking_20261003/analyze.py
PYTHONPATH=src:artifacts/safety_queue_braking_20261003 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/liyufeng/miniforge3/envs/safeduo/bin/python -m pytest artifacts/safety_queue_braking_20261003/test_stop_latch.py tests/test_eval_perturbations.py -q
```

停止探针/FIFO工具契约检查13项通过；不将工具检查称为安全策略通过。统计审计断言通过，详见results.json、各cell协议和stop_observations.csv。主比较复用首cell，未重复计数；旧候选7/192、seed14 0/128与冻结参考512窗口保持原样。

## 面板验证

Chromium检查9条延迟展示行（主比较复用3行、顺序组6行）、12条停止观察、全部新旧曲线、历史表和390px布局；检查日志随报告保存。所有旧科学数据字段保持逐项一致。线上核验完成后，发布提交和HTTP结果记录到DELIVERY.json。

## 仍未解决

100ms目标FIFO仍发生违规；当前停止诊断依赖事后时刻，尚无可部署检测器及其制动轨迹验证；连续cell的隐藏状态/重置原因待定位。未改变默认策略或学习权重，无新增实机、物体任务、接触力、视频或连续碰撞安全证据。

既有多cell结果继续保留为其当时顺序协议下的实测值，不能自动等价于独立进程冷启动。新发现不删除历史计数，但扩大了其解释边界；需按新的隔离条件补复核。
