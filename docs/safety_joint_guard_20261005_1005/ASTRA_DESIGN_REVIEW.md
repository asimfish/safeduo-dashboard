# Astra：joint guard 事前设计独立审查

审查时间：2026-10-05 02:15 UTC（10:15 +08）。范围仅本轮 DESIGN/PLAN、旧 reference/admission/finite helper 及其实际调用的 projector。未运行模拟、未读取新方法结局、未实施采样或修改代码；本文件是唯一写入。结论是**配对四格设计可检验两因素及其交互，但新 helper 的实现验收、实际新 bank 独立性和结果均待核验；不批准物理安全策略**。

## 1. 固定实验及解释口径

| 方法 | forecast admission A | reference envelope E |
|---|---:|---:|
| baseline_guard | 0，原 dynamic 512 | 0 |
| admission_guard | 1，union 容量1024 | 0 |
| envelope_guard | 0 | 1，0.050rad |
| joint_guard | 1 | 1，0.050rad |

DESIGN 固定三对 initial/command seeds：1709489568/1701627244、1109912610/1651261960、497208883/779659058。计划是192个新 matched case、各四方法、共768个960步窗口，前60步零输入。它们现在是**注册数量**，不等于已实现的新初态覆盖或完成数量。采样按同一风险规则与 raw 静置资格筛选，属于条件风险域，不能视为 IID总体随机样本。旧 development seed60317411不计新覆盖。

四格须共用初态 bank、外部 command tape、原 actor/checkpoint、actor32行观测路径、物理配置及完整 finite 检查。闭环状态分叉后 alpha/p、豁免掩码、R19 mask 可以自然不同；要求计算规则一致，不能要求这些闭环数值一直相同。报告每case四格违规/深违规、rescued/new失败、运动量、debt及实际暴露；交互差 Y11−Y10−Y01+Y00 只作这批配对样本的描述，不附 IID泛化CI。保留不足暴露、异常、capacity abort和部分任务，不按保护结果重采或补种子。

## 2. 投影接口：正确顺序与易混淆的 baseline

旧 reference_envelope.py:35–45 将 soft limits、原速度箱、tracking tube 合成 delta_bounds，**先限制传入 cmd，再给原 project 传 bounds**。这是应保留的顺序：backstop.py 的 R19 在投影后返回传入 cmd，本身不受箱/alpha约束；只给 solver bounds而不限制 bypass输入会漏过 reference，投影后才裁剪也不能继承 solver 的安全残差。

但旧 helper 的 `box_only` 也预裁剪 cmd。因此，若新 E=0 路径调用它，R19直通值、alpha方向/预算或 active标记可能不同于原 baseline；不能凭“gap=None”称完全原样。**E=0须保留原 project 输入/接口语义；或以实际 source/CPU边界对照证明 common shim 恒等，尤其 soft-limit、R19和出箱输入。** planned development equivalence不能只查最终 q，需连 cmd/exec/issued target/alpha/bypass/原行选择一起核对。

E=1的预裁剪还有两项真实算法作用，不能称只加一个出口限制：原 `_alpha_rows` 使用受限 cmd 的方向及范数；authority clamp 使用 bounds 计算最小 Gu，改变有效 h。`active | reference_governor_changed` 也改变反馈标记，必须单列 governor改写与 collision projection，核对其后续消费者，不悄悄归入碰撞兜底。原 retreat 判据在 project 前读取 raw cmd，须保持其原语义。

## 3. 两个已执行的 CPU 反例

我从旧源 AST 提取实际 `reference_bounds`，只在 CPU/float64执行；下面是合成代数反例，不是新物理轨迹或发生频率。

**R1：raw-boxed proposal 的单点 forecast 不是 reference proposal 的保守下界。** 取 q=(0,0)、当前issued target=(0.049,0)、box=0.025、gap=0.050、raw delta=(0.025,0.025)、J=(1,−3)。实际 helper 给出第一维delta上界0.001，因此受限delta=(0.001,0.025)。若当前d=0.060m，所有pending targets与当前target相同，则：

| 预测对象 | d+J(target−q) |
|---|---:|
| 当前issued/pending target | 0.109m |
| raw-boxed proposal | 0.059m |
| envelope proposal | 0.035m |

含当前d的原forecast最小值仍为0.059m；dmin=0.030、band=0.010时不触发forecast admission，而envelope proposal已低于0.040m。假设该行未被原选择集合选中，raw-critical也不救回它。缩小多维参考区间可能移除一维开离分量，因此“区间更小”不推出其单点预测更安全。原raw proposal可能多收行，也可能漏掉受限 proposal；current/queue最小值不能排除这个反例。

**可执行建议：** 保留注册的 raw-proposal admission 定义并明确这项局限；同一帧另记录 reference-limited proposal 的 row forecast/是否本应入列，作为不改变控制的诊断。若打算将它或区间最坏值加入真正 admission，必须在新结果前明确冻结因子定义与所有适用cell，不能运行后偷偷补入joint cell。全forecast持久化不能自动证明该预测集合覆盖了受限proposal，也不证明线性预测覆盖非线性PD运动。

**R2：可达参考区间非空，不等于它和安全/alpha半空间共同可行。** 取 q=0.100、target=0、gap=0.050、box=0.025。旧helper实际返回 singleton delta=0.025；下一issued target=0.025，tracking debt仍为0.075rad，故0.050不是无条件debt上限。对安全行 G=1、原h=0，最小可达Gu=0.025；原 authority规则 `h=max(h,0.9*min Gu)` 给h=0.0225，仍留下至少0.0025的安全不等式残差。若此臂alpha=0，沿正受限cmd的alpha行还要求delta≤0，与singleton冲突。

这并非 NaN，也不是多迭代即可消除的残差；完整finite通过不能证明可行。不得为消除它自动改 authority、alpha、豁免或solver。本轮需记录 singleton/zero-not-in-bounds、minGu、clamp前后h、alpha约束、solver passes、出口实际残差，并保留该情形为潜在失败机制。仅单行可行也不保证多个半空间共同可行。

## 4. FIFO/debt与有效残差必须对齐实际执行

严格FIFO6、无target rebasing、无preemption是共同硬约束。reference限制新issued target的下一可达增量，不会删掉旧6个pending targets，也不是限制实际q或加速度的设备侧控制器。旧helper的nearest endpoint恢复分支允许暂时超过0.050 tracking gap；须标注受限分支、恢复所需步数、各臂/关节max debt及J方向closing debt，不能把0.050称始终满足的物理包络。

每帧绑定 pre-q/qd、原rawcmd、受限cmd、bounds、原选择/新增row IDs、alpha/p、issued-before/after、实际FIFO序号与6个target、本步delivered actuator target、measured next-q。尤其区分 admission读取的实际adapter queue，与原project通过 `_pending_target_history` 接收的6个历史targets；不能因长度同为6就替代其中一个。

原backstop.residual在R19覆写之前生成。**同时保存原residual与R19/target-limit之后真正effective exec的安全及alpha残差**，后者按同一有效行、h和bounds重算，不靠原residual=0宣称实际输出可行。不要更改bypassmask计算规则，A因素通过新增行自然影响mask属于该因素的总效应。

有上述字段才能区分：未admit/预测漏项；bounds–alpha–安全集不可行；有限轮数投影未收敛；R19出口覆写；以及数学可行输出之后由旧queue、PD跟踪或线性化误差造成的实测继续闭合。观测上有关联仍不自动证明唯一根因。

## 5. 新 helper/source 与落盘验收项

- 全四格都检查9021行的d/dmin/closing/J及预测，含baseline未selected行；同时检查q/qd、actual queue、issued/proposal、soft limits、gap/box/dt、bounds、受限cmd、effective exec/新target。检查每个预测中间量 **在 minimum/reduction前** 有限，防止正Inf被有限current-d掩盖；无行号丢弃、无exception后继续physics。
- A=1保留原selected union raw-critical union注册forecast-dangerous，stable row IDs去重、padding不计；超过1024按注册abort并落失败receipt，不取top1024继续。A=0原512截取规则保留；另报full required数量，不将仅selected512全有限误称全9021已验证。检查完整forecast chunks的step/slot/row轴、row-dmin/exemption身份、全部帧和terminal覆盖。若未保存full J，不把有限性assert或完整forecast文件称成独立离线重算了全部J预测。
- 保留plan中完整16s finite-guard development equivalence，明确它只证明该development有效输入上的非干预性。核source install顺序、退出恢复、原kwargs转发、actor32路径、backlog_aware=false/predict_backlog及全部原参数；有限性检查不得悄悄开启debit、改priority或重写队列。
- 新source、三个bank与tape、每cellmanifest和完成receipt冻结绑SHA；检验同case四cell初态/输入相同，并以actual sampled数组而非仅seed不同证明新bank与旧bank无重复。统计完成/异常数量独立于shell exit0。任何guard/capacity失败均单列，不能算零违规成功。
- PLAN承诺保存native render前**及后**完整张量和实际USD optics/pose，可补上上一轮after仅producer assertion的缺口；最终仍需实际文件审计。独立视觉replay与数值run须各绑自身state，若跨run轨迹不等继续披露，不将图像冒充matched holdout证据。本次设计审查没有看新照片或校准实测。

以上风险不要求另开大矩阵。先冻结并核实际新helper的四格语义、诊断字段与abort合同，再依据已注册数据解释结局；本审查不认可任何新方法安全成功，硬件/物理策略批准均为false。

## 6. 本次实际阅读与证据绑定

实际读取 DESIGN.json、PLAN.md、bank_registration.json的注册信息、旧reference_envelope.py、mechanism_runner.py、guarded_diagnostic.py，以及生产backstop.py和duo_env.py相关调用路径。执行了上述两个CPU反例和文件SHA计算；没有执行新runner/source测试、采样独立性实测、结果重算或simulation。

| 文件 | SHA256 |
|---|---|
| H/DESIGN.json | `eccd0e4de83aa6780a26f9ef9133438a13823bc331c0258c962afab872516b76` |
| H/PLAN.md | `bf7c1389d81c47c53e305da1e9c4098bd1f0c355a36187e71cccf387db9967ec` |
| safety_mechanism_20261005/reference_envelope.py | `7cf0ccccce2f1a81d0ea657797a71eb1c5cb8b852aa94e84767a2dfca1ab9c47` |
| safety_mechanism_20261005_causal_obs/mechanism_runner.py | `174e297cdd9c8e48c00894e37426d7afbc48245f717de69fe2d3d4c6d173ce12` |
| safety_mechanism_20261005_causal_obs/guarded_diagnostic.py | `ff76f1071b9619a7dc07764799bc8b496ba6e4e440fdea3c05b07e2ed95fc880` |
| src/safeduo/safety/backstop.py | `82a0f0e5ddbbe1b9eecbdbbd249a9354b50d78ab6e121473286e8dbb28f19afa` |
| src/safeduo/envs/duo_env.py | `cd5d3b8b473a7d713da7a4dbc400dcbe231a2dd9563732671c4e3b7047e3fe0c` |

这是读取时的source绑定；新helper准备好后须另核其实际SHA及上述组合接口。旧封存design/evidence/final/native review均未修改。

## 7. 新源到位后的具体 composition 复核

随后实际读取了H/guard_runner.py、full_finite_guard.py、reference_envelope.py、test_full_guard.py及full_guard_tests.log，追读继承的 risk_runner → wide_runner_feasible → EpisodeTrace 调用链。四份新Python文件均通过我执行的CPU AST语法解析；H/reference_envelope.py与旧helper逐字节相同。以下结论绑定本节SHA快照，不替代修改后source复核或实际development结果。

**已确认的实现性质：** GuardTrace的四个mode都安装同一个 `safety()`，都构造full row IDs和全true active_mask，producer检查full J后调用同一个full_forecast；没有把全部9021行送入原actor。E=0没有调用 `box_only`，因此§2指出的非干预风险在此source的安装分支已避免；E=1才安装原reference wrapper，bounds和受限cmd仍在原solver/R19之前。A=1的select保留原观测distance/full_dmin/full_viol_exempt，使用forecast只扩大选择，不以预测替换measured geometry或关闭豁免；select本身1024 hard abort/no dropping的语义正确。

full_finite_guard.py:15–36检查full d/dmin/closing、full J、q/qd、issued/rawcmd/limits和boxed proposal；对actual pending、issued、raw proposal逐一检查target、displacement、线性delta、predicted，**然后才做minimum**。正overflow被最终有限min遮住的旧漏洞在这一运算顺序中被封住。这个结论限于这些预测运算，不能扩展为所有solver内部/物理出口都有finite断言。

**P1，holdout前必须澄清/修正的预算继承问题：** guard_runner.py:32在 `super().start()` 之后捕获 `env.safety_dist_out` 为original_selected，:52先调用它。实际 `WideTrace.start()` 在system0路径已把该函数包装成 `merge_critical_rows(...capacity=None)` 加 **512 hard cap**（wide_runner_feasible.py:85–98）。因此A=1也会先经过旧512 cap：原selected+raw-critical需要513行、最终union仍≤1024时，尚未执行A=1的1024 select就提前abort。旧机制source使用 `self.original_safety()`，即这层wrapper之前的选择函数，新source使用 `self.original_selected()`，二者不能视为同一接口。

建议A=1明确从wrapper前原选择开始，再由注册union/select1024统一处理；A=0保留原system0 dynamic512路径。须用512+1且≤1024、以及1024+1的边界seam验证，前者不应触发继承512门，后者必须fail closed。不要为此关闭原raw-critical或豁免。若运行参数不是system0，则该继承条件需据真实launch contract重新核对；不能靠变更method绕开原baseline定义。**本快照尚不满足“admission/joint只有注册1024预算门”的source验收。**

**剩余有限性/诊断边界：** 当前forecast helper没有验证box/dt/gap配置本身或queue长度；这些常量/运行配置须由冻结launch及runtime receipt验证。reference wrapper仍没有独立检查computed bounds、limited cmd、原project输出的finite；它的bounds比较不能拒绝NaN输出，finite-valid输入也不是solver输出有限的证明。若交付宣称“所有送入physics的target均finite”，还需在实际出口/入队前的共同gate提供证据。原有bounds/alpha不可行、R19后有效残差、raw forecast非保守反例均未因新full-guard自动消失。现source保存pre_target_debt/unreachable标记，但尚未保存本报告提出的有效G/h/bounds/alpha与R19后残差诊断；如不补存，应收窄因果辨识与独立重建的声明。

32步chunk机制将每帧full forecast复制到CPU，step检查(64,9021)，chunk按start/stop/SHA列receipt，正常960步应有30个chunk、共64×9021×960个forecast值。这里只确认source设计，未见已完成chunk/receipt。保存的是跨target的最终minimum；每个target的中间预测和full J未全部持久化。因此可以独立审最终数组完整性/有限性与抽样binding，但全J有限及每个中间量的正确性仍依赖冻结producer执行，不可声称离线重算了所有target的9021行预测。guard_calls断言在post-step observer执行，实际pre-physics调用保证仍需结合正常callback链与运行receipt核验。

**5个测试的证明范围：** 主线程日志记录5/5 OK。我读过所有测试，未另跑整套测试。它们覆盖常量oracle/queued最低值、far-row d/J的NaN和±Inf、正overflow不能被min掩盖、命令/queue/q/qd/limits故障，以及实际reference wrapper向mock R19接口传bounds并预限制输入。它们不实例化GuardTrace/WideTrace组合，因而没有发现上述继承512门；也未证明actual solver可行、所有输出finite、真实FIFO/4mode语义、完整960步等价、chunk完整性或actor路径不变。wrapper测试模拟直通，不等同真实solver的R19/exemption综合验收。

| 本节新增实际读取文件 | SHA256 |
|---|---|
| H/guard_runner.py | `2b47e943b2a15aeb45841a2b5f7f4c58448f7e81e1807577cad7962e27b57ce2` |
| H/full_finite_guard.py | `a1f046bb7f9576198488041e621a5c664fbff66ad1ce12734dfe7768cbe65988` |
| H/reference_envelope.py | `7cf0ccccce2f1a81d0ea657797a71eb1c5cb8b852aa94e84767a2dfca1ab9c47` |
| H/test_full_guard.py | `7744752df695450fa3700d67443b5d55aae593c326f685aa1e29faf96054cb4c` |
| H/full_guard_tests.log（producer测试记录） | `b8a2b09f445ac39230a70fb88e4e762cc4710aca854b8954de7f9eb4ed8fadce` |

本轮此时未开始holdout；新bank sampling的结果与独立性未在本审查验收。新source提供了明确的full-row/pre-min numeric改进，但P1与上述声明边界必须保留；不据这份设计/source审查批准任何安全策略。
