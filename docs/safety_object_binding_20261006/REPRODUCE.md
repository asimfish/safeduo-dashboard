# 复现、来源与判定边界

本轮全部 14 个密集原生记录块、132 张原始 PNG 归档在同仓库的独立数据分支，面板通过固定 Git 提交直接引用全部原图。下载入口、逐文件 SHA 和已实际回读核验见 MEDIA_ARCHIVE.json；数据归档的 raw/、images/ 分别包含全部原始块和原图，没有仅挑选成功轨迹。PUBLIC_MANIFEST.json 逐文件给出 SHA，recording_receipt.json 绑定原始块及原图，native_results.json 绑定登记和评分器。原生日志按 gzip 保留原字节，LOG_MANIFEST.json 给出解压后 SHA。

## 已有研究环境离线复核

研究脚本位置：/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006

```sh
export PYTHONPATH=/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/liyufeng/miniforge3/envs/safeduo/bin/python \
  /home/liyufeng/safeduo/artifacts/safety_object_binding_20261006/analyze_native_v2.py \
  --out /mnt/nas/data/lyf/double_hand/safety_object_binding_20261006/native_v5
/home/liyufeng/miniforge3/envs/safeduo/bin/python \
  /home/liyufeng/safeduo/artifacts/safety_object_binding_20261006/phase_diagnosis.py
```

第二个脚本是运行后追加的描述性分析，不改变第一个脚本的任务判定。原始冻结评分器仅因输出字段重名无法完成，修正版唯一表达式差异见 ANALYSIS_OUTPUT_REPAIR.json；原脚本、失败日志保留。

若在其他机器从公开包重算，先从 MEDIA_ARCHIVE.json 的 archive_zip_url 下载完整固定提交归档，将其中 raw/dense_*.npz 与 native_initial.json、recording_receipt.json、bound_reference.npz、images/ 放在同一个数据目录中。评分器当前使用研究机绝对路径来核对 REGISTRATION_V5.json，也用 __file__ 所在目录输出结果。需准备对应目录，或另存一个明确标记为路径适配的脚本，在 reg_path 解析处指向同 SHA 的本地 REGISTRATION_V5.json；不要改动原公开脚本、原登记或数值门槛。其他机器上的脚本路径改变不会保留 corrected_source_sha256，必须报告适配。离线分析只需要 NumPy、Pillow 和 Torch；object_binding 使用研究工程的 safeduo 模块，完整物理复现还需要原机器人资产、IsaacSim/IsaacLab 和登记中的 checkpoint，公开面板本身不包含全部第三方资产/模型权重。

## 原生物理执行

本轮实际脚本 native_runner_v5.py、登记 REGISTRATION_V5.json，执行设备 cuda:0，CUDA_VISIBLE_DEVICES=0,1。环境版本见 ENVIRONMENT.json。仅初始设置时写物体状态；正式循环没有物体跟随、吸附、固定关节抓持或速度清零辅助。完整 receipt、无 failure.txt、正常进程退出和离线校验共同用于确认录制完成；Kit 退出码 0 本身不足以证明 Python 执行成功。

再次执行会产生新的物理结果，**不要写入已经封存的 native_v5 目录**。将本公开脚本/登记作为来源建立新的登记和输出目录，先固定输入、代码、资产、软件、随机种子和分析，再运行。名义参考相同并不保证不同环境槽位的接触末态逐位相同。应保留重复波动，不能只报告最好一次。

控制器沿用 a27_v7_r18_s46_coop，未修改 A31 纯随机关节实验策略；两者不可合并分母。原生控制步 0.016666 s、物理步 0.008333 s、轨迹栅格 1/60 s 分开。21 字段物体输入使用原生初态，无视觉误差验证，仅位置/朝向参与任务规划。qualified NPZ 内继承的旧 validation 和 object_events 元数据不是新轨迹验证或已执行的吸附指令；新准入以 LAYOUT_QUALIFICATION_V2.json 为准，本轮实际状态以 raw/ 为准。

## 局部拒绝与判定

原 0.53 m 布局及两组名义中心候选的求解拒绝全部保留。第二次事先声明的候选顺序中首个 0.50 m 通过，后续四个未测试；不能宣称整张工作空间已覆盖或求解器拒绝的中心全局不可达。V1–V4 没有完整任务试验，详情见 NATIVE_REJECTED_ATTEMPTS.json。限位读取探针不属于任务分母。

接触只有末物理子步的物体归因法向数据，84 个过滤伙伴并非全场景危害测量。不涵盖摩擦全扳手、完整冲量、全部机器人相互/机器人桌面接触。bbox 使用作者尺寸，不是逐三角形网格离桌判据。静置支撑、错误桌面和离桌负对照为有限验证，不撤销旧 FAIL_UNCALIBRATED。手接触>0.1 N 和抬升>50 mm 的状态仅作描述，不是正式协议 100 mm/2 s 的通过证据。

软件/交付门禁 PASS 只说明本公开包的限定验证完成。8 条有意设计的开发对照不属于 IID 随机可靠性证据，新增正式留出为 0，full_g0_pass=false，task_reliability_accepted=false，没有硬件安全验收。

## 网页回顾兼容性

本轮保留 MP4，另提供 VP9 WebM 给不支持 H264 的测试 Chromium。两种编码的 44 个图组/132 张来源图均逐组检查。预览视频跳转需要 HTTP byte range；Python 的默认 SimpleHTTPRequestHandler 没有这一功能，使用公开的 range_preview.py 进行本地门禁。编码不支持和无 range 的旧门禁失败日志保留，公开端还需独立进行全部案例的播放/跳转验证。
