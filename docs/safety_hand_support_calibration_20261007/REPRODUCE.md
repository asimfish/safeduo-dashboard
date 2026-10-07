# 原生手开度与支撑校准复算

本轮是20个静态诊断环境，3次完整原生记录；不是20条搬运任务。旧反馈组8/8超时与旧评分保持不变。

## 离线评分

从MEDIA_ARCHIVE.json的固定ZIP下载完整原始归档，校验MEDIA_MANIFEST.json所有文件SHA。下述两版本评分器与各自登记匹配：

```sh
python analyze.py --raw /path/to/safety_hand_support_calibration_20261007/baseline_v2_gpu0 --registration REGISTRATION_V2.json --out baseline.json
python analyze.py --raw /path/to/safety_hand_support_calibration_20261007/self_off_v2_gpu0 --registration REGISTRATION_SELF_OFF_V2.json --out self_off.json
python analyze_open_pose.py --raw /path/to/safety_hand_support_calibration_20261007/open_pose_v3_gpu0 --registration REGISTRATION_OPEN_POSE.json --out opening.json
```

评分器校验登记、全部记录块、原始图像、原生元数据、接触身份、接触点和连续步号。逐字段对比公开的三个结果JSON；不要回改评分门槛。NumPy足以离线评分。通用静态评分器在第一版原生启动之后、原生元数据和任何记录出现之前封存，诊断门槛更早写入原登记；V3评分器与验收门槛均在V3启动前封存，时间分别记录在ANALYSIS_SEAL/OPEN_POSE_ANALYSIS_SEAL。

## 物理复现

需要登记SHA对应研究工程、IsaacSim5.1/IsaacLab0.54.2、机器人/物体/fixture资产。工程所有资产与SDK不全部随网页开放；原始数据足以复算数值，不等于完整可运行软件发行。Python3.11环境路径见启动日志。两个GPU都保持可见；本记录器用cuda:0与renderer0。GPU1的USDRT报不支持，失败记录保留。

```sh
env PYTHONPATH=/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006:/home/liyufeng/safeduo/src CUDA_VISIBLE_DEVICES=0,1 LD_PRELOAD=/home/liyufeng/miniforge3/envs/safeduo/lib/libstdc++.so.6 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 /home/liyufeng/miniforge3/envs/safeduo/bin/python native_probe_v2.py --registration REGISTRATION_V2.json --out /path/to/new/baseline --headless --device cuda:0 --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'
```

同命令换REGISTRATION_SELF_OFF_V2.json生成自碰诊断组；换native_probe_v3.py与REGISTRATION_OPEN_POSE.json生成开放姿态对照。输出必须是新目录；不写回封存原目录。原HandDriver从已封存文件导入，其SHA记录于登记，源码也随原始归档提供。

## 信号与边界

原生物理步0.008333秒，1440步约11.99952秒。每步固定机械臂位置参考并写手目标，再推进一次物理步；绕过System0/policy任务动作，不运行神经控制器。只在初始化写物体位姿/速度，采样循环无吸附、物体随手或速度清零。实际机械臂可追踪不完全，参考误差保留。两个支撑环境仅用于原配置及关闭自碰诊断；V3八环境全部移开物体。

场景遍历包含实例代理。先按每个碰撞几何查找最近刚体祖先，把刚体主人或无刚体的静态碰撞几何作为互不重叠伙伴，连同地面过滤。物体接触计数、起点和所有接触/摩擦输出在下一次访问前复制，避免复用缓冲污染。每120步保留支撑对象原始接触点，其余逐步保存法向矩阵、摩擦合力、净力和重建法向。V3额外记录全手部伙伴法向矩阵及身份；包括手基座。API行为参见[NVIDIA官方PhysX Tensor文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)。

本版本记录中net与全部过滤法向和逐值相同，添加摩擦则有非零差；不能把net当作完整摩擦/接触扳手。静态法向支撑重量合格不替代动态摩擦、力矩、视觉或抓持标定。每次渲染前后直接PhysX读回全部DOF与物体位置/速度，逐值不变。原图保留材质纹理不可用等警告，不声称完整资产纹理依赖或相机标定验收。

V3只提议U侧thumb1开放目标0.35rad，未改变资产、自碰、增益、原旧任务门槛。9–12秒静态开放验收与完整任务成功严格分开。初始化依旧使用旧手角，启动/闭合过程所有数据保留，静态验收通过不说明整段无碰撞或能够可靠夹持。
