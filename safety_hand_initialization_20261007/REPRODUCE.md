# 原生初始化与绝对闭合目标复算

本轮包含两次静态原生运行，每次6环境×2160物理步，dt=0.008333秒。
3个不同四臂参考配置来自同一旧任务的0/4.5/15.2秒；不是随机初态银行。
构造前默认U thumb1为0或0.35rad；各运行再对照reset后写入0或0.35rad，循环目标开放0.35rad。
所有手部自碰、原生增益和资产保留，移开物体只在初始化发生。

三类状态独立保存：构造后、reset后、诊断初始化写入后。Tensor接触视图在这些阶段之后建立，
所以首个明确循环物理步起的力测量不能证明构造器内部步骤无碰撞。
两次绝对闭合终点分别使用原0.45全手比例与原任务各臂/拇指比例。新驱动从当前关节目标
向绝对终点线性插值，持续完整0.6秒；原驱动是比例按1/ramp速度变化，路径速度不同。
闭合终点逐值一致与释放路径是否安全必须分别判断。静态无物体测试不证明真实抓持。

封存REGISTRATION_SEAL.json后依次执行，不并发占用GPU：

```sh
env PYTHONPATH=/path/to/this_package:/home/liyufeng/safeduo/src CUDA_VISIBLE_DEVICES=0,1 LD_PRELOAD=/home/liyufeng/miniforge3/envs/safeduo/lib/libstdc++.so.6 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 /home/liyufeng/miniforge3/envs/safeduo/bin/python native_probe.py --registration REGISTRATION_old_default.json --out /path/to/new/old_default --headless --device cuda:0 --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'
```

第二次换成REGISTRATION_neutral_default.json以及新的输出目录。
原生循环每步存DOF位置、速度、目标、cache；完整手部伙伴法向矩阵包括合并掌基座；
物体法向/摩擦分别记录并在下一次访问前复制，保留接触计数与身份。
所有渲染前后直接读回DOF与物体位置/速度并要求逐值一致。
原始日志中的远程材质纹理缺失警告不删除；6个直接物理资产根和源文件SHA另行核验，
不是完整传递USD/纹理依赖封存。

离线评分仅需Python与NumPy：

```sh
python analyze.py --raw /path/to/raw/old_default --registration REGISTRATION_old_default.json --out old.json
python analyze.py --raw /path/to/raw/neutral_default --registration REGISTRATION_neutral_default.json --out neutral.json
```

评分器独立从旧原生元数据与当前软限位重算闭合终点，不导入候选驱动。核验登记、记录块、
连续步号、元数据、原图SHA和渲染状态。对比公开RESULTS文件全部字段；门槛见冻结登记。
全部原图、记录、脚本和失败均位于MEDIA_ARCHIVE.json固定提交ZIP及逐文件MEDIA_MANIFEST。
研究工程、SDK和全部资产没有完整开放；原始数据可数值复算不等同完整可运行发行。

Anygent历史接口本轮status与recent均401，新增读取消息0条，无最后消息可报告。
工作状态从本地上一轮DELIVERY_CLOSURE及公共main新增零输入实验恢复；未假称读取到远程历史。
