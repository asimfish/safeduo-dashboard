# ASTRA：joint终态原图与state有界观察复核

审查时间：2026-10-05T04:23:13.088439+00:00；**PASS_BOUNDED_CAMERA_OBSERVATION**。仅visual joint的env024/040/056、step959；不作策略安全或跨run轨迹验收。

实际用view_image看用户指定3张原PNG及各自1张同帧对侧俯视，共6图。Python完整解析3份state，只输出摘要；独立核对3组全部27张PNG字节SHA、receipt↔state规范绑定、27view camera几何和3组native before/after。未重跑父线程42组/378图全审，未模拟或修改冻结相机/候选。

## 图像可辨识性与局限

| 场景与实际查看视角 | 可见内容与限制 |
|---|---|
| env024 overview + u_opposite_high | overview中四个arm基座、主要关节链、手和台面边缘可定位；距离较远、台面占画面局部，细指节和毫米级间隙难分。对侧俯视补足白色低姿态arm和桌缘附近的遮挡。 |
| env040 u_opposite_low + u_opposite_high | 蓝灰双臂低姿态、腕手位于桌缘附近可辨；低视角中臂间重叠、背景白色arm被遮住，不能只凭此图辨清四臂全部轮廓。对侧高视角能分开蓝灰双臂并补出白色双臂，仍看不到被实体遮住的接触面。 |
| env056 f_pair + f_opposite_high | f_pair中四臂主要姿态和台面可辨，前景白色双臂较清楚；对侧高视角补到背面和桌缘，但非目标蓝灰arm部分出框。目标球体全入框不等于四臂完整轮廓全入框。 |

整体明暗差足够辨认arm与台面，但白色外壳/手指和浅色台面有亮部细节较弱处；env040低视角的大面积白背景也不适合推断精细间隙。这是视觉观察而非曝光标定。臂/手在投影中接近或跨过桌缘，不等于已证明物理接触或穿透；背景地面与桌下阴影不能给出载荷支持结论。

## state与保存的真实native记录

三份state_time_s均为15.99936。逐组核对保存的16个native数组（四臂q/qd/root/root_vel）中全部64环境before/after逐位相同；按保存controlled indices将全64的native q逐位绑定同visual dense终态。三slot的q/qd及root选定值逐位绑定state，controller/actuator targets逐位绑定dense；6queue等于本visual dense的controller_target[954:960]。终态qd只与保存native/state绑定，没有next dense frame。

完整state的9021条distance/dmin/closing均有限；非豁免distance按row identity映射四桶，与同visual dense terminal official_margins逐位相同。这是记录一致性和归约，不是重新算9021条几何或full closing/J。142个sphere centers/radii逐位核对dense/identity。

| env | cross/selfF/selfU/table终态margin (mm) | 非豁免负距离行数 | qd L2 F_L/F_R/U_L/U_R (rad/s) |
|---|---|---:|---|
| 024 | 1058.497071 / 23.083180 / 80.729708 / 0.952691 | 0 | 0.406045 / 0.004858 / 0.019502 / 0.540455 |
| 040 | 641.131520 / 13.488024 / 36.447465 / 1.137048 | 0 | 0.269441 / 0.400740 / 0.564940 / 0.069412 |
| 056 | 1208.689809 / 23.082748 / 34.458145 / 1.196831 | 0 | 0.420624 / 0.388240 / 0.066980 / 0.049144 |

终态margin只描述该帧，不能推出全960步是否违规；主数值joint6 vs visual7的FAIL_EXACT_REPLAY保留。预设7slots×3steps独立于结果，这3slot不代表全部失败/峰值样本。

## actual USD/K和fiveplanes的准确解释

27个camera-to-world row matrix的平移/旋转一致性通过，位置与eye+env origin一致。producer源码确实在render后从USD ComputeLocalToWorldTransform和Camera optics attrs取值；本审计复核保存读回及来源，未连接live USD或重渲染。K自身来自Camera.data.intrinsic_matrices，不是USD getter；这里由保存USD focal/apertures独立重建K核其一致性。

27view均为零aperture offset；focal=18，fx/fy≈1099.4989px、principal point=(640,360)。算术最大差：K=2.89e-05px，center投影摘要=0px，五平面摘要=3.33e-16m，eye位置=9.54e-07m。这些比较尺度不是物理安全门。

以row matrix逆变换world centers至OpenGL camera坐标，depth=−z，图像y向下。独立单位平面法向计算球心到左/右/上/下侧平面及z=0正深度面的距离，再扣radius。F视角只检查62个F球，U视角80个U球，overview/front/reverse全部142球；27view目标球均满足5cm预设余量。第五面是正深度面，不是USD实际near clipping；无第六far面。state未保存actual clipping_range，不能宣称near/far authored值已校准。

| 用户指定原图 | 球数 | camera-target距离 (m) | 球表面五平面最小余量 (m) | center像素bbox |
|---|---:|---:|---:|---|
| env024 overview | 142 | 6.640783 | 0.806287 | 415.78,238.25 → 877.24,542.10 |
| env040 u_opposite_low | 80 | 3.583295 | 0.742597 | 447.21,310.41 → 839.03,429.83 |
| env056 f_pair | 62 | 3.884585 | 0.416060 | 442.22,257.88 → 716.57,554.18 |

本3组27view最小球表面平面余量为0.240478m。27view的SDK pose均标stale，不能以[0,0,0]校准；保存actualUSD矩阵与错误SDK值分开。球体入框不证明完整link silhouette、无遮挡、像素特征配准或连续接触。producer字符串“all view planes”仅能按明确fiveplanes范围解释。

## 文件与字节绑定

实际view_image的6个原图路径，未经编辑：

| 绝对路径 | SHA256 |
|---|---|
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/multiview/env_024/step_0959_overview.png | `b1599aef2bb3967950105c55d1076fc9e69090d7e92b54eb58c3aa07ae03ba9b` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/multiview/env_024/step_0959_u_opposite_high.png | `40570a29abf75462765ebed4c45fe6e5c52af9ace6d82c6566af1b675bff8ea6` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/multiview/env_040/step_0959_u_opposite_low.png | `c903a90ffa950e105a2d5e7467ad7ae02200d38ed3d2564672056a1e04724e4d` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/multiview/env_040/step_0959_u_opposite_high.png | `c5e94c94d8b0c70ce3748c06b11efb312799617f1b2f886d977c793bf2e222a8` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/multiview/env_056/step_0959_f_pair.png | `71ca7db8f07f26c7f685781348e885368f3c0b04cb96e67973f79f3d90324915` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/multiview/env_056/step_0959_f_opposite_high.png | `f59f4ca10b620b35109a7a88e89e8e0b57ed536ad6d2a42aa71c5697254bc2c8` |

| state / 关键输入 | SHA256 |
|---|---|
| multiview/env_024/step_0959_state.json | `5f91626df2c94ae3ae4ae38c5d65781117b0ba720b32e3a0438dc55f0a66e88b` |
| multiview/env_040/step_0959_state.json | `ea9177dec03474435226d5e00e14a95588eb595a8f8de6ceeddb9c2f8c49520f` |
| multiview/env_056/step_0959_state.json | `577bc7571eae7877b61af687746ba81b97436788f29dafc1ff5f58571d1092fd` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/cell_001.npz | `e384e54b46d77f4385db3b72b5a5c50303a96dc69a19992120daa3892c7a9048` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/full_row_identity.json | `3e9acf5a52ffaff1def8dfb48b7a8300dff8c8595fea0df21449e4d56c19dd12` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/camera_receipts.json | `3c923cd12098848aca77f895e9a3d0c8cc81e9a38091b311d4c5881f190dd0b4` |
| /mnt/nas/data/lyf/double_hand/safety_joint_guard_20261005_1005/visual/joint_guard/visual_protocol.json | `2190a35efd910622f37101b1fd74a86c194aa2b9411cdc8363a563a8bcfaaeb9` |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/visual_registration_v5.json | `b8310ede26582f0126e6115a82ab3a58877f01e1bd63edb73e72fef797c9986e` |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/visual_runner_v5.py | `dfddc2f1c56b1fb4b428a829f9e602fd0648517f26e28ea187af5d2ce1436523` |
| /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/native_visual_verification.json | `17c540c5ae8d66012ca8372427dfe5d399d566a7bc13f5ac6a8da987b533446a` |

三份state引用的before/after native文件均实际核SHA，统一为 `8df33a6a444b8ed617a2a06aaf863b590e130c39f08b92058d47a63790b74d15`。本次after数组已保存且独立比较，不是仅接受producer assertion。

父线程PASS42/378另属全量验证证据；本次独立范围3state/27PNG SHA/27view算术/6原图/3组native绑定。before/after只证明渲染组瞬时native相等，不证明后续轨迹或GPU0↔GPU1前向重放无扰动。**该有界相机观察无新增阻断；production_promoted=false，hardware_approved=false，物理安全未批准。**
