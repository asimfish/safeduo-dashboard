# 原生释放与退离对照

24个有意设计的开发任务：4个既有准入布局×三条件×两个物理种子。名义F中心(0.50,0,0.84)、U中心(-0.40,0,0.84)，XY ±20mm、朝向±8°。两个种子和环境槽位同时变化，不单独归因物理种子影响；没有增加空间范围/IID/正式留出。四臂分别成对搬运两件物体，没有四臂共持一物。条件、手指原事件、初态、控制器、限位、物理参数及门槛在执行前固定于REGISTRATION.json。

## 离线复算

从MEDIA_ARCHIVE.json下载固定提交ZIP，解压完整归档。下载本页面analyze.py和同SHA的REGISTRATION.json放在同一目录，装有NumPy的Python即可运行：

```sh
python analyze.py --raw-root /path/to/unpacked/safeduo-dashboard-COMMIT/safety_release_feedback_20261006
```

评分器自动检查完整receipt、登记/所有数据块/所有图片SHA及图像关联物体状态。将RESULTS.json和timeseries.csv写入脚本所在目录。对比本页原RESULTS.json的全部字段/数值；换路径不应改变脚本SHA和登记SHA。没有挑选成功任务或降低门槛。

## 完整物理复现

需要登记列出的研究工程源代码、机器人/型材/静态道具资产、a27_v7_r18_s46_coop权重和IsaacSim5.1/IsaacLab0.54.2环境。这些第三方资产和模型未全部随网页公开。源码登记加279个共享源码/配置及六个直接碰撞资产根的前后快照；不是完整嵌套纹理/引用依赖封存。

在新目录登记输入、源码、软件和资产后运行，不写回已封存原始目录。两个GPU都需可见，执行选cuda:0与renderer0；隐藏GPU1会导致Omniverse GPU索引失败。示例（原研究路径）：

```sh
env PYTHONPATH=/home/liyufeng/safeduo/artifacts/safety_release_feedback_20261006:/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006:/home/liyufeng/safeduo/src CUDA_VISIBLE_DEVICES=0,1 LD_PRELOAD=/home/liyufeng/miniforge3/envs/safeduo/lib/libstdc++.so.6 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 python native_runner.py --registration REGISTRATION.json --out /path/to/new/block_0 --block 0 --headless --device cuda:0 --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'
```

另执行block1。源文件哈希在创建场景前核对。完整receipt、无failure.txt、正常退出和离线评分共同决定是否完成；Kit退出码0不单独算完成。仅试验初始设置写入物体状态，循环没有物体跟随/吸附/速度清零。检查没有Obj命名FixedJoint仅限该路径断言，不是全约束系统审计。

## 机制边界

scheduled_debt保留R37补偿；stationary_release在15.2≤任务相位<18.7秒清零机械臂名义和补偿指令，原手指事件按共同物理钟执行，已经积分的目标不重定基。clearance_feedback增加18.7秒退离准入：两物体本桌法向>0.1N、作者bbox底部距0.8m桌面≤6mm、速度≤0.03m/s、角速度≤0.3rad/s、倾斜≤10°、四手实测关节相对默认开度最大误差≤0.2rad、各分配手法向≤0.1N。连续6个控制状态通过；等待2秒未通过显式中止并停止四臂新指令。准入后手接触>0.1N、物体速度>0.1m/s或角速度>1rad/s连续3状态触发中止；这是碰后反应，不是预测避碰或已证明的安全停机。

三个条件共同改用每步权威相位重新采样（清空旧pending缓存）、12环境/组、30秒观察期；不能当作上轮v5逐位重放。两组初始化参考按原生初始位姿绑定，物体实时反馈只给任务层门控，不进入神经策略观察或重新训练。控制步0.016666s、物理子步0.008333s、参考栅格1/60s分别记录。碰撞物体/fixtures保留，仅非碰撞装饰不显示。

退离接触/再接触诊断按任务相位≥18.7秒统计，包含门控停等或中止后的物理状态；不能把这些计数全部解释为实际执行了退离。关节路径、held期间实际运动和最后相位另列。原生命令清零也不强制要求安全投影产生的执行量完全为零。

## 原始图像和视频

360张登记固定时刻的俯视/正视/侧视覆盖全部24任务。第一组名义布局三条件在14.8–20.8秒每6控制步额外采样正视实际帧，共183帧；三段1280×800视频上720px来自1280×720原图，下80px只加条件/物理钟/任务钟/状态文字，不裁剪、插值或修改原图。两编码全部帧逐张核对解码RMSE。编码10fps、源采样0.099996s，末帧保持0.1秒，约6.1秒，不是完整30秒视频。

每次渲染前后直接重读PhysX物体transform/velocity及所有四臂DOFposition/velocity，而不是只检查IsaacLab缓存。逐值不变及图像关联物体状态存于receipt。这不是相机内外参、遮挡、全几何标定验收。

## 保留的未通过项

原任务门槛不变；新增合格完成额外要求未中止且参考运行到24.2秒。原安全球未报警、曾抬升、开手目标达到0都不等于稳定抓持或完整任务安全。净接触与84伙伴法向过滤和的差值不是完整摩擦扳手测量，静态fixture与地面等未完整归因；旧FAIL_UNCALIBRATED不撤销。根因、真实可靠抓持/放置、视觉反馈、四臂共持、广域随机以及正式留出仍待完成。

本轮候选拒绝采用：8/8超时，0/8合格完成。真实任务中未进入退离准入状态，碰后反应路径只有软件测试，不能宣称已验证原生恢复效果。0.2rad开度条件未先在这些手的自由开度/姿态基线上完成校准；8条该条件始终失败，部分未与物体接触时仍有约0.27rad偏差，不能直接认定是物体卡住。只读USD手限位探针读取48关节，未见正下限，可排除这个作者限位解释；它未读取完整原生关节限位或校准开放姿态，远程Franka基础USD在独立USD读取器中未解析，手本地子树可读取，不作完整场景验证。

实际放置与物体夹持已在干预前分歧：同布局方法之间干预前物体位置差最大175mm，16比较中11组参考逐位相同，另外5组有原生坐标舍入与绑定计算差异。此结果不提供机制因果收益估计，不能把候选与基线翻转计数差归因为候选伤害或保护。后续需要可归因的复现与独立手开度/完整支撑准入，而不是按这批失败直接放宽判据。
