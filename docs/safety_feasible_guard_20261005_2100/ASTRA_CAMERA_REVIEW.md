# Astra：两个闭合九视角组的有界独立复核

结论：**PASS_BOUNDED_NATIVE_SIX_PLANE_CAMERA_REVIEW**。只审 joint_reference 的 env000/step75（scheduled）和 env038/step116（first_failure），18张未经编辑原图已逐张用 view_image 实际查看。未读 cell_001、整轮运行协议或仍变化的 capture receipt，未执行最终 dense scorer。

每组16个 native before/after 数组含全部64环境，q、qd、root transform、root velocity 的 dtype、shape、字节及数值逐一相同；selected 全关节读回和实际 controlled indices 下的 q/qd 与该组 state 精确绑定。这个结论限于保存的渲染边界瞬时状态，不证明之后960步轨迹不受影响，也没有把图绑定到独立数值运行。

实际USD相机矩阵、光学属性和clippingRange是 producer保存的直接读回证据。本侧线用正交基变换、由USD光学值重建的K和归一化六面半空间独立算术复核；没有启动Kit再读USD。SDK pos_w仍是过期零值，与actualUSD分开记录，不冒称有效校准读回。

总览/front/reverse核验142个球；F角度只核62个F球，U角度只核80个U球。球半径在六面法向距离中扣除，所有所审目标球距六面均≥50mm。这个包络只覆盖模型所列球，不认证所有网格、桌边、手指轮廓或无遮挡。像素/K/法向残差比较容差只是保存float32及独立算术核对，未用于放宽物理或包络判据。

|组|native标量相同比较|最小目标球六面余量m|保存的最小非豁免几何m|
|---|---:|---:|---:|
|scheduled env000 step75|12800|0.620594756|0.001625984907|
|first_failure env038 step116|12800|0.505422926|-0.004022486508|

env038/116保存了真实负的非豁免球几何裕度；这不是照片可直接识别的穿透/真实接触证据。本次未读前序全stream，不能独立确认“首次”时间顺序，更不能由两状态判定唯一失败机制或成功率。

两组图的灰色F与白色U臂和桌面总体可辨，未见邻槽绿架横贯前景；多个方向仍存在臂间、工具和自身壳体遮挡。总览中手/腕占画面较小，白色壳体较亮，不能解析毫米级间隙；部分pair/俯视图桌边裁切。对侧角度补充可见性，不等同完整网格轮廓证明。

|组/视角|实际原图观察|
|---|---|
|env000/75/overview|四个基座、四臂和桌面可辨；主体较小，手指细节不足以判断毫米间隙。|
|env000/75/front|F双臂投影部分重叠，U前臂/手部也有重叠；桌面边界清楚。|
|env000/75/reverse|反侧补出另一侧关节，但两U臂局部重叠；F侧远处指尖尺寸较小。|
|env000/75/f_pair|F两臂可辨；同桌U臂位于近景，视觉上较大，不能把标签当作无前景遮挡证明。|
|env000/75/f_opposite_low|前后F肩肘/手部有重叠，U局部也重叠；桌面右侧超出画面。|
|env000/75/f_opposite_high|俯视有助分开F前后臂，桌面下边裁切；末端细节仍有限。|
|env000/75/u_pair|U两臂和大部分桌面可辨，U腕/末端相互重叠；下方桌边裁切。|
|env000/75/u_opposite_low|灰色F前景遮住部分白色U臂/末端；四个基座及桌面仍可辨。|
|env000/75/u_opposite_high|俯视补足白色U臂姿态，但另一臂/自身壳体后的表面不可见。|
|env038/116/overview|四个基座及桌面可辨，低伸展灰色臂和上方伸展白色臂可定位；末端太小，无法读出负裕度。|
|env038/116/front|灰色前臂低伸展和白色臂向桌心伸展较清楚；两白色臂和手部局部重叠。|
|env038/116/reverse|反侧可见低伸展灰色臂，中央多个手/腕投影靠近并遮挡，不能判定真实接触。|
|env038/116/f_pair|F低伸展姿态可辨，白色U前景遮挡部分F腕/工具邻近区域。|
|env038/116/f_opposite_low|低侧更清楚显示灰色肩肘与手，但两灰色手/前臂投影重叠；桌后侧/底面不可见。|
|env038/116/f_opposite_high|俯视补出灰色低伸展臂和两个末端；桌面下边裁切，不能测接触深度。|
|env038/116/u_pair|白色U两臂及朝桌心的腕较清楚；局部手部重叠，桌面下角裁切。|
|env038/116/u_opposite_low|四臂可辨但灰色近景遮住白色U臂/中央手部，单图不能显示全部最近间隙。|
|env038/116/u_opposite_high|上方反侧补足四臂相对位置及低伸展灰色臂；中央手/腕仍局部重叠。|

旧launch保持失败：Kit exit0，但visual_protocol failed，0有效window/PNG；错误为显式argv缺少--methods。v2→v3计划除了显式原默认system0和新visual_retry_1输出路径，actor、种子、环境、视角/stage及控制/observer source SHA均不变。两组成功capture不替代整轮retry闭合验收。

与各照片同组的state/native/PNG文件，以及必要的已闭合共享球identity、冻结source/计划和旧失败元数据均记录before/after SHA；未哈希新运行的整体protocol/receipt。全部具体路径、算术误差和逐数组结果见JSON。

执行oracle SHA256：`e20cf6abd2e54554de1b80783fa341f900d7dff93489e459f6e58d14afb0fcc2`。

物理策略仍 **BLOCKED**；hardware/production approval均false。最终12条件评分、整轮相机/数值跨run比较和最终科学报告审阅仍待后续材料。
