# Astra 最终科学交付声明审阅

结论：**PASS_BOUNDED_SCIENTIFIC_CLAIM_REVIEW**。最新版报告、图表与已闭合独立评分一致，未发现阻断科学声明的事实差异。物理安全仍为 **BLOCKED**：本轮不能认定修复改善安全，更不能批准生产、硬件或真实任务策略。面板、公开下载及最终 metadata 归档验收属于父线后续交付，未由本侧线验收。

## 数值与图表

12 个闭合数值条件共 768 方法窗口、无效 0；分母为 192 个配对初态/输入案例，每案四种相关方法。沿用全部 960 帧原非豁免几何严格 `<0`、深违规 `<−0.005 m`，未引入消负容差。

| 方法 | 严格失败 /192 | 深失败 | 平均实测路径 rad | 平均范围/软限位 |
|---|---:|---:|---:|---:|
| admission_full | 50 | 42 | 125.273033 | 40.089723% |
| admission_scaled_036 | 16 | 10 | 65.639977 | 24.480340% |
| joint_reference | 25 | 17 | 51.165195 | 15.951179% |
| joint_repair | 30 | 20 | 51.758991 | 15.948863% |

主比较救回 4、新增失败 9、双方失败 21、双方无失败 158；**25→30、17→20，不能称物理安全改善**。其他报告配对为 full→repair 救回35/新增15，scaled→repair 救回10/新增24，与独立评分一致。四类失败可在同案重叠，其和不是独立案例总数。

0.36/repair 的路径比 **1.268185003746598**、范围比 **1.5349269337923654** 均未落入±10%；这是未匹配运动对照。路径、范围和四臂移动帧占比是描述性实测运动，不是任务完成或风险等价指标。

本次实际通过 `view_image` 查看最终 [comparison.png](comparison.png)。四面板的失败柱、路径散点、救回/新增柱和24个暴露格与结果、报告一致；深蓝格白字清楚，分母/仿真范围清楚。SVG 的24个“暴露数/24”标签另与结果逐值核对。PDF 仅绑定字节和文件头、阅读共同生成源，未另声称视觉查看PDF。暴露不足未删除；热图不能当作全26维或场景覆盖率。

## 首次失败与可行性范围

本次将 [first_failure_audit.json](first_failure_audit.json) 的121个方法案例与已有独立dense逐案台账交叉核对：全部首次步号、首失败类别、负裕度与身份一致；实际120个唯一NPZ文件（有共享帧），并非少一个案例。121条父记录均记载最近负行在前一步selected集合中且至少一个机器人相关gate为真；本次核的是该记录/台账绑定，未重新读取121份J来重做全部membership或LP。

| 方法 | 首失败案例 | 父记录安全/alpha残差均≤1e−6 | 至少一机器人硬线性集合不可行 |
|---|---:|---:|---:|
| full | 50 | 29 | 4 |
| scaled | 16 | 13 | 0 |
| reference | 25 | 2 | 20 |
| repair | 30 | 2 | 29 |

这些计数与报告一致。范围残差另属全量审计，不应把表中两项残差当作所有约束通过。父LP是保存float32约束转binary64、行归一化及HiGHS数值判断，没有额外全量双证书；不证明完整动力学可行或唯一根因。本侧线之前实际独立算术/LP只覆盖冻结的18个旧快照、4个候选案例、6个新full首失败案例，不能扩写成121次独立LP。新6例中3个实际目标满足原容差（1例严格零）随后出现负几何，支持“线性投影通过不能推出物理安全”，不支持唯一PD/队列归因。selected身份也不能单独推出前一步非豁免或有效约束；动态mask与时刻必须对应。raw proposal准入相对reference proposal不保守的既有边界保留。

## 相机证据

父闭合回执为52组/468张：42个预设组与10个视觉运行自身首负组。独立核对全部gallery身份与唯一图像引用、42组预设集合双射及manifest SHA对应；**没有在本次重新查看468图或重算全部native/视锥**。

之前独立相机报告仅覆盖参考运行 env000/75、env038/116 两组18张实际原图，及该两组保存的全64环境16数组before/after逐位相等、USD/K/六面目标球余量计算。本次核对它们在最终gallery/manifest中的身份和SHA延续。遮挡、末端像素小、局部桌边裁切、SDK姿态过期零值的限制仍成立；目标球视锥通过不是完整网格或无遮挡证明。

| 运行 | 数值失败 /64 | 视觉失败 /64 | q首差帧 | q最大差 rad | 前向绑定 |
|---|---:|---:|---:|---:|---|
| reference | 9 | 9 | 0 | 1.069526014 | FAIL_EXACT_REPLAY |
| repair | 10 | 10 | 0 | 1.113624871 | FAIL_EXACT_REPLAY |

失败数相同不能覆盖轨迹不一致；照片只绑定自身视觉状态，不能当作数值holdout的逐位匹配失败照片。瞬时原生相等不证明未来非干扰。旧启动失败仍为0有效窗口/0PNG；重试只补显式默认 `--methods system0` 和新输出路径，不能把失败启动计成完成任务。

## 精度、执行与复现

[ASTRA_FINAL_SCORE.json](ASTRA_FINAL_SCORE.json) 保持 **PASS_ENDPOINTS_AND_NATIVE_FLOAT32_REDUCTION_BINDING**：6272个映射比较通过，8个原生float32 ratio/range描述量精确复现；原binary64分支的三项描述性比较 **FAIL**、旧attempt/source字节与全部双精度值保留。没有放宽原比较rtol，更没有改物理端点、配对或消除正残差。共同代码SHA与各块登记bank文件SHA分别绑定，修复评分器接口假设不等于改变实验控制。

父完整PASS结果/执行回执与外层退出 **143** 并存；终止原因未确认，不称clean CLI exit或退出0。独立评分成功是另一条证据。全9021预测/selection审计由父已执行回执支持；全J仅producer检查、固定快照/首失败selected J有存档，不能称逐帧全J离线归档。

最新版 [REPRODUCE.md](REPRODUCE.md) 改用绝对路径只读 `recheck_closed.py`，禁止在sealed目录重跑写回执CLI，另要求新namespace及绝对路径重登记。实际基础smoke的12端点/类别/FIFO与独立评分一致；本次仅读源码/日志，未再次执行该CLI；可选 `--dense` 未声称第二次执行。`earlier visual runners ... unlaunched` 仅可指更早未执行源版本，不涵盖已失败的v2启动计划；REPORT明确保留了该失败。

## 闭合绑定与交付界限

[raw_evidence_manifest.json](raw_evidence_manifest.json) 为父双读PASS：**1613文件、64,449,692,574字节**。本次独立核对清单路径唯一、字节和、父消费468项、相机消费711项、已有独立评分raw输入81项及全部首次失败/gallery引用SHA，均对应。没有把读取manifest冒称本侧线再双读64GB。

22个本次消费文件在检查开始/结束SHA一致，report_build同时绑定实际生成源、结果与视觉回执；VERIFICATION的12个回执SHA一致。报告数值表、比值和SVG标签另经小范围独立算术核对。[NEXT.md](NEXT.md) 明确是未执行前沿，没有把未来预测/覆盖实验当成果。

无IID泛化区间、完整26维覆盖、硬件、实时deadline或搬运任务验收；旧任务分母不混入。VERIFICATION当前 **PENDING_DELIVERY**，科学状态与物理/发布状态分开。REPORT“公开报告及图像可以逐文件下载”按发布后的交付说明理解，本审查未验证当前公开可用；父线须以真实public/UI/archive回执闭合该声明。

## 最终文件SHA-256

- REPORT.md：`a5103d6a9a504b75386c61252d32dc6e949171e0405f27a98841ab93623e7135`
- REPRODUCE.md：`9a10cf63f4fb0d745daaf748a5139c81a7b922ee697e983b1f2f11db2df4b19a`
- comparison.png：`1afd481f6f9809b86b867508305c564dcec6d5687306dddfbcb80801241047d5`
- comparison.pdf：`c16cc269b1146b2d79a25bfbcc3c0911d51d650be48d846adfd1000485e98612`
- comparison.svg：`6d30d51c01ac627b436fc0e5ebe3c66a5232083dee589eb42cd12c773abd1fbc`
- 独立评分：`d55b8450da7dbd88ce451f39f59da07453e65b41d4593a60b378eb325b9aaae9`
- raw manifest：`a209da3728e4822d40a475ac04379ad78489a3ec0faaedf7777f69a059336874`

完整消费SHA、实际检查范围与本MD全文SHA见 [ASTRA_FINAL_REVIEW.json](ASTRA_FINAL_REVIEW.json)。本侧线仅写这两个新review文件，没有运行physics/GPU、修改父源/阈值/旧证据或发布。

