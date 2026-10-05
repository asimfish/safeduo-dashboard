# ASTRA：fresh admission_full 六例首失败旁审

**PASS_BOUNDED_FIRST_FAILURE_REVIEW**；只审三个block各最早两个env，不审全192/768结果。策略／硬件批准均false。

选择冻结时间 2026-10-05T13:45:46.000562+00:00，当时LP调用数0；选择SHA `ceba0e4a8b7b733bacda781ad23e1fe2015dd09762db97126f6c67ae16239a3e`。

实际执行源SHA `866b36995fec7884441fc0388a4f6c6300467ecf332815a641dfd6b8bb7ac43d`；复用既有NumPy LP/证书函数源SHA `f5fa617bc3d4d500c27154a98b66c82db99df3a9acbfb3b3f641497ed83e8610`。未导入torch/Isaac/parent审计或模拟。

计划、v2 runner/helper、初始native input_recipe/bank、identity、首失败snapshot及消费的已闭合32-step chunks均SHA绑定；允许协议处于running；未哈希mutable整个protocol，消费时状态变化不会改变已绑定的config子树。
只绑定消费的effective config/dt/safety/args/checkpoint子树，并用原YAML继承链、BackstopConfig字面默认值及env构造映射重建核对。

native初态soft limits在input_recipe与银行逐位一致；bounds从native limits−pre issued target和原速度箱重建。无reference的admission_full不套用.05 envelope。
Jqd/pending/d_eff/gamma cap、严格gate/rel、budget/h/authority、alpha方向行独立重建。LP用核验后的原保存float32系数，完整26维固定p，不分配新预算。
actual_project_return是project返回，returned_cmd是实际target增量；后者用真实soft clamp积分重建，二者分别重算所有安全/alpha/bounds残差。

| block/seed | step/env | 首post最差非豁免几何 m | row | 前步selected/gate/exempt | target safety m | target alpha rad | target bounds rad | 原tol内 | 联合硬集合 |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| 0/278060357 | 158/60 | -0.0006270539016 | 9019 | True/True/False | 0 | 0 | 0 | True | FEASIBLE |
| 0/278060357 | 162/59 | -0.005090905353 | 8941 | True/True/False | 1.028182828e-08 | 1.601314388e-05 | 2.235174179e-08 | False | FEASIBLE |
| 1/973625877 | 106/17 | -0.006662532687 | 8973 | True/True/False | 3.641976498e-09 | 8.711179333e-09 | 0 | True | FEASIBLE |
| 1/973625877 | 110/39 | -0.0008350089192 | 4939 | True/True/False | 0 | 3.922939135e-05 | 0 | False | FEASIBLE |
| 2/517962307 | 140/22 | -0.001262828708 | 8973 | True/True/False | 0.0002334509642 | 0 | 0 | False | FEASIBLE |
| 2/517962307 | 234/12 | -0.005354225636 | 8887 | True/True/False | 0 | 1.959619936e-07 | 0 | True | FEASIBLE |

实际重建／绑定检查 972项，最大跨精度算术差 4.609577564e-08；比较尺度5e−7仅为算术，不是物理门。
6例中 6例在消费时已存在closed dense，额外绑定chosen env的q、target、queue、class margins并独立核首个官方负post步；其余first身份只由被冻结producer的first-seen逻辑和最早snapshot顺序支持，没有冒称完整post历史复算。

所有6例post_full_d/dmin/exempt均与下一pre帧逐位匹配，selected pre d/dmin/IDs与前步chunk逐位匹配。前步官方exempt从真实bitpacked全行mask读取，不把selected或structural mask当成非豁免证明。

## 可支持的结论与限制

JSON中的case classification以实际target残差是否≤原1e−6及硬集合LP witness/phase-I对偶证书为据；任何小正残差仍原值保留。原official geometric failure严格按post margin<0且post不exempt，不加碰撞epsilon。
若出现小/零target残差后实际负几何，只能说明该步selected线性控制可行性／投影容差不足以推出下一测量安全。FIFO6使新issued increment与本步实际delivered pending target不同；PD、线性化、未选行和动态mask仍须区分，不能认唯一原因。
nearest_row分别保存前步official exemption、actualgate、F/U rel、结构/条件mask和post exemption；selected不保证实际relevant或前步nonexempt。
未独立FK重算9021 sphere几何；post几何是SHA绑定的producer实测sphere margin，不能升级为全link/native碰撞或硬件证据。未复制parent完整failure_audit/full-stream评分，不给全局率、泛化CI或候选批准。

执行：先`astra_first_failure_review.py --freeze`（拒绝覆盖），再`--evaluate`；复算已有冻结选择只用`--evaluate`。JSON含每个输入SHA、stable protocol子树、LP证书和可用closed dense绑定。

结果JSON SHA256 `e778a403141ec9ee51a32550f97a8e4fb37d75994afed55b840e2ba4bfde3dda`。仅写astra_first_failure_review.py、ASTRA_FIRST_FAILURE_REVIEW.json/.md；旧报告与父源/计划/raw未写入。

## 六例实际发现（限定抽样）

6个selected硬集合全部有LP witness；3例actual-target所有残差在原tol内，另外3例分别为2个alpha超门和1个safety超门。没有在这6例发现联合不可行；这不能否定旧study已经证实的R2反例。6例最近负行的前步official exemption均false且actual gate／相关robot rel为true，因此这些病例没有“nearest仅selected但被豁免”的歧义。

三个小／零残差反例是block0/env60@158、block1/env17@106、block2/env12@234。首个最差post margin分别为−0.627054、−6.662533、−5.354226mm。env60的actual target安全／alpha／bounds残差均严格0；另两例保留所有小正值，没有用epsilon抹去官方负几何。

另外block0/env59@162与block1/env39@110的actual-target alpha残差分别为1.601314388e−5rad、3.922939135e−5rad，不能因safety残差≤1e−6称完整投影容差PASS；block2/env22@140 actual-target safety残差2.334509642e−4m，虽然其联合硬集合可行。所有6例无R19 bypass。

在三个小／零残差反例的nearest行，J·新target增量分别为+3.203523、+17.106041、+15.817454mm，J·真实q变化分别为−0.725731、−11.031271、−11.274162mm；pre target debt最大值分别为0.359743、0.356716、1.373053rad。这里只是“新issued增量线性指向打开，但本步真实运动仍闭合”的同步事实；FIFO6、PD及线性化/authority等机制尚不能唯一归因，也不建议根据这6例自动换策略。

消费时全部三个admission_full cell已complete；6个chosen env的第一官方负post步均用closed dense独立核对，绑定了3份dense SHA。没有等待或评分其余方法。972项是程序记录的重建/selected绑定计数，另有每例8项closed-dense字段绑定和首步核对；终检还逐例核selected class与arm-mask等于full identity映射。本轮只调用6个完整联合集合的LP；没有重复旧18例或父全体failure审计。
