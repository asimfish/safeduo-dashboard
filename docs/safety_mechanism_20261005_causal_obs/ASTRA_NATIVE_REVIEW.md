# Astra closed v4 native/camera bounded review

**结论：PASS_BOUNDED_NATIVE_CAMERA_EVIDENCE。** 只接受本次 v4 的原图/state/native/USD camera 绑定证据；策略/硬件/生产晋级均 false。当前 visual 仍17/64违规，visual/numerical trajectory exact FAIL 保留。此前 v3 calibration BLOCKED 和既有审查原字节未改。

唯一新增文件为本报告。CPU只读计算完成于 **2026-10-04T17:41:31.480380+00:00**（北京时间2026-10-05 01:41:31）；随后读取并比较 H/native_visual_verification.json，未运行该主线程 verifier、模拟、Kit/GPU或修改源。对象仅为 RAW/visual_development_v4，runtime protocol complete、64×960，非holdout现场照片、非新随机命令试验。本轮不重复 v1 审计。

## 已实际执行的独立核验

- **12组 /84视角 /84 PNG**：slots0/14/35/47 ×steps71/75/959，各7视角。全部state SHA对camera_receipts、全部PNG SHA对同组state.images严格一致；receipt与state的env/step/images一致，12组和84路径唯一。全部PNG实际解码1280×720且非均匀空白；逐张视觉检查仅下面两张。
- **三份 native-before**：16张量/份均64槽、finite，F每臂19DOF、U每臂18DOF。native-selected q/qd/root/root_vel与原始snapshot对应槽逐值一致；四臂前7/7/6/6个受控DOF的q在steps71/75/959、全部64槽，与同次dense严格全等。其他hand DOF仅snapshot及所选槽记录支持，未声称它们全部绑定到只含26个arm DOF的dense。
- **qd范围明确**：steps71/75全部64槽的受控native qd与下一步pre_qd_compact严格全等。step959不存在下一步pre；该帧qd验证为snapshot与4个capture selected-state的严格绑定，不宣称终态全部64槽qd已与缺失的next-pre记录比较。
- **同次 state/dense**：12组controller/actuator targets、sphere centers严格一致；full distances按冻结row_class、排除同组full_exempt后重算四class minima，与official_margins严格一致。完整已存float dense finite，实际actuator targets严格FIFO6。独立重算该visual的official<0窗口数为17，无epsilon清零。
- **84个 USD pose/optics**：从保存的真实camera prim world row-matrix重算平移、正交rotation、det≈1、OpenGL quaternion一致性及eye+env-origin。K由实际authored focal length/apertures与1280×720重算；本数据两个aperture offsets均0。未将stale SDK position字段改名当actual。
- **离线projection实际重算**：用每组记录的world sphere centers、USD matrix逆变换及K，按F/U/overview真实组掩码重算depth、pixel min/max、count及all_inside；84view均positive depth且目标组球中心在画幅内，pixel摘要零数值差异、depth摘要误差<1e−12m。另将look-at target转换并投影到principal point，检查相机朝向。
- source与registration两项SHA实际匹配；protocol observer SHA与真实v4源相同。原development recipe的tape/q_initial/limits比较全等；对照仅用于说明这是旧development命令的独立replay，不涉及重新审计holdout。

|独立camera复算项 |实测最大误差 |
| --- | ---: |
|actual position 对 eye+origin |1.0662403e−6m |
|USD quaternion rotation 对 row-matrix |3.3306691e−16 |
|authored optics重算K 对保存K |2.8906500e−5px |
|look-at target对principal point |2.5822882e−4px |
|重算pixel min/max对保存projection摘要 |0px |

使用源中camera validation容差（position1e−5m、K2e−4px等），未改变任何安全判违门。上述camera误差范围不代表机器人几何/碰撞计量精度。

H/native_visual_verification.json的12组、84图、84 calibrated views、3 native snapshots、staleSDK84、17违规、FIFO及trajectory exact字段，与本次实际计算一致，无发现的数值冲突。

## Producer assertion 与离线证据的界限

v4源码在每次capture开始用真实PhysX getters并clone全部64槽的q/qd/root/root_vel；每slot七次render后再次fresh getter/clone并torch.equal，state写入在成功assert之后。它有实际fresh native读取，区别于v2仅SDK缓存比较。

**after arrays未保存。** 离线独立重算的是已保存before snapshot与dense/state绑定；render前后全部64槽相等由已绑定producer源码、成功capture及完成记录支持。未声称离线重建了不存在的完整after数组，也未验证native contact force buffers或连续子步。

实际USD pose/optics通过的同时，**84/84 sdk_position_matches_actual仍False**。有效SDK默认update_latest_camera_pose=False的旧字段单独保留；v3元数据失败不被追改或重新标为通过。

与cuda1 numerical development全轨迹比较，cmd和初态严格相同；q、ee、exec、official_margins、controller_target、actuator_target各自strict exact FALSE。无法把v4照片标作数值对照现场，不能唯一归因设备、render或PD，也不批准安全策略。

## 实际原图与可见性

已用view_image直接查看：

- [env_047/step_0075_u_opposite_high.png ](/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/visual_development_v4/multiview/env_047/step_0075_u_opposite_high.png)
- [env_047/step_0959_u_pair.png ](/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/visual_development_v4/multiview/env_047/step_0959_u_pair.png)

t75对侧高视角可辨U臂与桌面，中央手/臂仍互相遮挡；t959 u_pair可辨U臂终态，前景另一侧机器人局部裁切。两张可用作同次布局/姿态证据；center-framing不证明完整link/球表面轮廓均入镜、无pixel occlusion、真实接触身份、连续时间碰撞安全或物体搬运成功。未把17违规用终态图片覆盖。

## Digests与完成范围

raw项相对 RAW/visual_development_v4；源/报告/主receipt项相对H。camera_receipts SHA绑定全部12 state及84 images，均已逐项核验。先前三个审查的SHA仅用于确认原字节保留，未再次审计其旧实验。

|文件 |SHA256 |
| --- | --- |
| `cell_001.npz` | `331290a26725dee799a7035a612f79ba9bb006f01257321f178d446e60aad65f` |
| `visual_protocol.json` | `0f3979785f96a990c15d3fa64158e20072049744acaff24244d47b2ae9e1893a` |
| `camera_receipts.json` | `086a1f3ca9dfb12f70e57075fa079856122b478e4f15ebb2caf7d953665c2b90` |
| `episodes.json` | `d457e8b1a28e7fce2c15ba3908132cffded8177bd7792f650a6d5818062d0981` |
| `native_render_step_0071.npz` | `9887fbcd6d4028a140387e0ffb4466c49219d70f14947c85001ef6c0c175f299` |
| `native_render_step_0075.npz` | `628c8a49c688dfea92bb77b5b516478aaa7cc0a7c7c7f4d7444651379ef31d32` |
| `native_render_step_0959.npz` | `c80ca39e766ca8f235426061984c3fe9416d2efe46c2cd9abe3d7fa366409da4` |
| `visual_runner_v4.py` | `ad137d70ed376dbd3686652ec62890c2fc69dd50d00d73f20e83506590176afb` |
| `visual_registration_v4.json` | `73d1188379af9216e552484e24dbdd59c2a3631685559c403feb05fbf9160be1` |
| `verify_native_visual.py` | `8bebbe8563389c92f7aac6fa23945d9fdddb89e017370ac131e8c53ac82e951a` |
| `native_visual_verification.json` | `8b84fa585c95725b4975d8e8d93d72a9c59bda32350a2b2055f5916c6950d4fb` |
| `native_visual_verification.log` | `10a592cb1d3456826e3a3e7ee45058afaeb33f9e6f7b85e9ec1ed824daae0f4a` |
| `CAMERA_POSE_DIAGNOSIS.md` | `c22060ae7fab576c70542462292fa4d340555aa462178a96a4926c944a3a16e4` |
| `multiview/env_047/step_0075_u_opposite_high.png` | `5d7421d4e5b281688c6d723ef344b93afb0e0e51daa360a1861a6cfdd77f6898` |
| `multiview/env_047/step_0959_u_pair.png` | `1e30ef4e97f1df4b109c725bd31a404c91bc11adaa11f739258017bb34b709b4` |
| `ASTRA_FINAL_REVIEW.md` | `04b47041e9a1441dde04d36ad0f838104b0439502e702ddda39e60351960b7ab` |
| `ASTRA_EVIDENCE_REVIEW.md` | `32dfcb46c061379f8ddd0decc41cfd37abef5516866186012ed0a0f33ebd4421` |
| `ASTRA_DESIGN_REVIEW.md` | `aef3063b738fac52c9391f4bdf612df5695533e74e3c7b10d6b6fd1fa9d4910f` |

本报告关闭 **v4有界 native/camera 证据审查**；无未解决的该范围数据绑定阻断。保留after离线不可复算、中心投影/遮挡范围、跨run exact FAIL和17违规的限制。无hardware approval、production promotion、holdout照片替代或安全成功声明。
