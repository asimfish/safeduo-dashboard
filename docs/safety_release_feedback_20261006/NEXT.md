# 下一轮接续点

本轮全部24任务、30密集块、534原图、183连续视频来源帧已完成。旧原门槛2/8、静止释放2/8，反馈0/8且8/8超时；候选拒绝采用，未修改共享策略/SDK/资产。完整结果RESULTS.json，原因GUARD_INPUT_AUDIT.json，干预前分歧POSTRUN_DIAGNOSIS.json，视频来源VIDEO_RECEIPT.json，归档MEDIA_ARCHIVE.json。formal/final新增0，不并入旧实验分母。

原始文件在 /mnt/nas/data/lyf/double_hand/safety_release_feedback_20261006/block_{0,1}。当前研究代码 /home/liyufeng/safeduo/artifacts/safety_release_feedback_20261006。主板自有工作树 /home/liyufeng/safeduo-dashboard-benchmark-protocol-20261006，归档自有工作树 /home/liyufeng/safeduo-dashboard-object-media-20261006。保持其他tracking_reserve研究和主板卡片。

先完成以下资格检查，再设计新的控制候选，不在本数据上放宽门槛求通过：

1. 原生读取每个手关节的名字、默认值、真实限位和实际开度；固定机械臂于拾取/放置/退离姿态，进行独立无物体接触的开手正负对照。当前0.2rad条件未经这些姿态校准，约0.27rad误差可能不代表卡住。只读作者USD48关节没有正下限；尚未锁定误差根因，不能归因于限位。
2. 对保留fixture（尤其U侧白色任务板）以实例代理遍历完整碰撞形体，增加逐伙伴接触归因。当前慢速无物体手接触时，U桌约0.34–0.86N、原生净力约1.962N，存在未完整解释支撑。先验证净力、法向/摩擦分解、接触对象归因和容量，再把支撑判据用于闭环放置。
3. 分开定位放置失败与开手失败，控制动作必须前移至放置/开手过程。当前8反馈窗口中4个在18.7秒门控前已经>80°倾斜，另外停新指令也不能恢复已失稳物体。需要物体位置、倾斜、速度、实际夹持接触反馈，而非只在退离时等待。
4. 控制对照需要新的复现方案：同布局方法在干预前物体位置最多已相差175mm，16对中仅11对参考逐位相同。采用完整槽位均衡、更多独立重复，记录干预前状态并预先规定可归因比较；不能按已观察结果筛掉失败初态。若采用状态恢复，先验证原生接触/求解器内部状态是否可复现，不把表面关节/物体状态复制当成完整物理快照。
5. G0资格通过后再展开既有正式协议的四臂共持/交接、四臂工作空间和强随机矩阵。原±20mm/±8°只覆盖小范围布局，本轮没有扩大空间；不要重新宣称已充分或硬件安全。

CUDA_VISIBLE_DEVICES必须0,1；执行cuda:0、renderer0。独立资产/SDK不修改；app退出前SimulationContext.clear_instance()避免STOP回调渲染挂起。正式任务要求完整receipt、无failure、正常进程退出、独立SHA和评分全通过。原图归档用独立Git分支，主Pages现接近1GiB，不删除旧实验以腾空间。
