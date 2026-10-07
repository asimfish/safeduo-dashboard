# 复现实验与范围

读取 REGISTRATION_DEVELOPMENT.json、REGISTRATION_SEAL.json 和 FAILURE_CONTRACT.md。源码及六个直接USD资产根有SHA256；嵌套远程材质未完整封存，缺失材质告警保留日志。IsaacSim5.1/IsaacLab.54.2/RTX5090。两GPU可见、cuda:0与renderer0；不停止其他GPU上的无关任务。

先对注册 source_sha256 验证文件；在相同源码与资产环境运行 native_probe.py，输出必须是全新目录。OMNI_KIT_ACCEPT_EULA=YES，PRIVACY_CONSENT=Y，CUDA_VISIBLE_DEVICES=0,1，LD_PRELOAD指向该环境libstdc++.so.6，PYTHONPATH包含本目录及SafeDuo/src；OMP/MKL/OPENBLAS线程1。

```
python native_probe.py --registration REGISTRATION_DEVELOPMENT.json --out <NEW_PATH> --headless --device cuda:0 --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'
python analyze.py --registration REGISTRATION_DEVELOPMENT.json --raw <NEW_PATH> --out <NEW_RESULTS.json>
```

完整性需要 recording_receipt.json、没有failure.txt、注册步数完整且每块SHA正确，不能仅凭Kit进程exit0。开发结果不全部通过是预期保留的实测结果，不得删除失败档。闭合目标与往返轨迹不是原任务时序等价替换。

VALIDATION_SELECTION_RULE.md在开发评分前冻结；register_validation.py依据全部六路径通过的精确目标生成封存护照与新机械臂参考注册。native_validation.py逐步调用手部准入候选；32未知随机目标和旧不合格目标拒绝，不执行它们的物理路径；所有候选申请与锁存中止另有admission_records.json。

视频由recording_receipt.json标识的连续原生PNG按10fps编码，既有H264/MP4，也有VP9/WebM，未作图像编辑。overview top/front/side视角保持原三路；detail视角使用真实U_L手连杆位置均值构图。原始图片1280×720；渲染前后原生q/qd/物体状态逐值核对。

DISPLAY_PAYLOAD.json.gz包含完整U路径结果和供浏览的曲线；法向力取每12个物理步最大值保留冲击，角度取代表状态，不代替逐步NPZ。MEDIA_MANIFEST.json列出原始及派生文件SHA；公共页面只引用固定归档commit，所有媒体可下载。主面板通过Pages精确提交与公开字节核验。

不可宣称：完整48手关节独立随机、26臂DOF/六臂对/全工作空间覆盖、真夹持/搬运/释放、硬件人体伤害门槛、构造内部无冲击、System0臂安全已解决。握持需要接触、承重、搬运阶段持续真实抬升与无夹具干扰等独立验收，仍未执行。
