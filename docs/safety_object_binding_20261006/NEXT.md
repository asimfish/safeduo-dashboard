# SafeDuo 继续工作状态

用户目标：充分、规范的 System0 安全实验，覆盖四臂操作空间、广随机和真实协作任务，并持续更新主面板。

本轮完整物理证据：native_runner_v5.py / REGISTRATION_V5.json；
/mnt/nas/data/lyf/double_hand/safety_object_binding_20261006/native_v5。
8 个有意设计的开发任务、每例 1572 步、12576 状态、132 同次三视角图；进程正常结束，完整 receipt，无 failure.txt。不要重新执行或覆盖封存数据。

固定轨迹 1/4 原门槛通过，位姿绑定 1/4，全部 official 安全球违规 0；6/8 任务失败。通过的槽位 3、6 的 U 物体只在退离期被抬高，抬高时没有双手同时接触。输入绑定已验证，稳定抓持/释放和全面安全尚未修复。U 在计划搬运有双手接触的案例1/4→3/4仅为开发描述，不代表可靠性改善；名义槽位0/4参考逐位相同但物理末态有差异，重复与槽位交叉未验证。

analyze_native.py 是运行前冻结版本，输出字段重名报错保留；analyze_native_v2.py 唯一变化是注册 objects 另存 layout_objects。ANALYSIS_OUTPUT_REPAIR.json 含逐字逆向还原证明；不要修改门槛或丢弃失败。PHASE_DIAGNOSIS 为运行后追加的描述。旧19/192任务结果、FAIL_UNCALIBRATED保持；旧128开发复核没有动块2。此前384/768随机窗口的控制器和分母保持独立。

V1–V4各0完整任务，保留失败；读取限位探针0任务。原F中心0.53m扰动参考超出限位后局部IK仍拒绝，选事先声明第二组首个准入中心0.50m。两个方案共用新名义轨迹，±20mm和±8°保持；没有物理结果选择布局。原名义轨迹在原生限位内。更远五中心局部求解失败不能称全局不可达。合格轨迹继承旧validation元数据不能用作新验证。

279共用源码/配置及6直接资产根SHA前后相同。没有冻结所有嵌套资产/纹理。执行循环无物体位姿/速度写回；不消费旧object_events吸附事件。当前物体输入为模拟器初态真值，21字段中的位置/朝向用于任务规划，其他字段不参与在线反馈。法向接触是最后物理子步、84精确伙伴过滤、无完整摩擦/危害测量；完整G0=false、新正式留出0、硬件验收false。

公开目标：docs/safety_object_binding_20261006/，主面板卡片objectBindingEvidence。独占工作树 /home/liyufeng/safeduo-dashboard-benchmark-protocol-20261006，分支exp/object-binding-20261006；仅修改自己的根卡片和这一目录，保留并行更新。已有present/verify_local/check_browser/verify_deployment脚本；交付证据分别在TERMINAL_GATE、delivery.json、FINAL_DELIVERY_CLOSURE。已授权更新main，无需重复询问。主板链接 https://asimfish.github.io/safeduo-dashboard/index.html#scientific 。

所有132原图、14原始块及5元数据文件已在数据分支exp/object-binding-media-20261006归档，固定提交8195a91895b5b93735c8a8e111165a0618b83a0c，151文件完整公网SHA回读通过。主站只新增曲线/代码/日志/关键帧视频，引用固定提交原图，不复制95MB媒体到Pages，保留所有旧卡片/资料。

视频仅为132原图组成44个三视角关键帧，88s，不是连续运动录像。原图关联状态缓存已对照数值；渲染前后只比较环境缓存中的物体完整状态及关节位置，没有额外PhysX底层状态读回、关节速度/相机几何独立校准。F双臂/U双臂分别搬运两个物体；四臂共持与交接一个物体仍缺证据。

下一研究前沿：先固定物体承载/支撑/释放/退离的原生独立测量，再登记在线物体反馈状态机、失败停止和消融。用同布局重复、槽位交叉和冻结种子区别环境变异与机制效应。补时间顺序、双手承载、稳定放置及释放解除的任务门槛，旧门槛结果保留。之后按完整benchmark协议扩展26关节可达域、所有6臂对、动作过程、质量/摩擦/感知/延迟/故障和正式留出矩阵。

另一独立 safety_tracking_reserve_20261006_1535 管线存在，不要停止、覆盖或修改它的进程/代码/证据。本轮只操作自己的artifact、数据和卡片。
