# R34 工业协作装配任务族与真实物体资产（方案 v0.1，2026-09-06）

> owner 09-06 指令：参照「四臂协作装配 MuJoCo benchmark」（4×Franka 搭铝型材结构 + 精细分拣 + 协作包装，面向工业场景），
> 好好搭建我们的场景，需要的话搜集合适的物体资产；目前的物体太粗糙。
> 本文是设计线的方案，**尚未动代码**；启动时机与首个子族待 owner 拍板（面板决策 D5）。

## 0. 一句话

把 SafeDuo 的演示/评测场景从「参数化几何体 + 搬积木」升级为**工业协作装配**：4040 铝型材框架搭建（首选）、
KLT 料箱分拣、包装三个子族，全部复用 R27 已验证的抓取原语，不引入新的手势探索；资产走 Isaac 原生 → 装配数据集 →
扫描件 → 程序化型材的优先级；GPU 只在探针/成片阶段占用。

## 1. 现状与差距

| 项 | 现状 | 差距 |
|---|---|---|
| 物体 | `assets_src/tasks/task_assets.yaml → generate_task_assets.py` 生成的 USDA 几何体：圆柱瓶、盒、T 形工具、长板、方 peg/socket；R26/R27 追加 4×4×20 立棒、4×30×8 梁、15×20×10 箱 | 无纹理、无真实形态；观感「只有 lego」；不像任何真实工作 |
| 任务 | S9 三棒（grasp / pick_place / handover）、R24 三族（吸附口径）、R26 四族（bottle / rod / box / relay，已纯物理化） | 任务语义是「搬来搬去」，没有装配目标；跨机协作只有递送和共抬 |
| 场景 | 两桌面对面（桌 0.8×1.2，间距 0.5），四臂固定基座，无夹具/料架/工位标识 | 无工位语义；相机取景缺乏可读的任务上下文 |
| 演示 | 46 部多机位成片，VLM 逐帧质检 | 视觉上不像工业协作，难以说明「安全采数」的价值 |

## 2. 硬约束（来自 R27 两天的标定，任何新资产必须满足）

1. **可捏特征宽度 3.5–5 cm**：F2 捏合接触开度 4.3 cm（f=0.5）、最小 2.6–3.7 cm；DFX 接触态开度 4.4 cm 且闭合时四指沿轴回缩 8 cm，
   对 4 cm 物侧捏窗口只有 1–2 cm。更大的物体走 **PalmFrame 双掌对夹**（box_colift 已验证）。
2. **质量 0.05–0.5 kg**，摩擦按材质（塑料-木 0.4–0.55、金属-木 0.3–0.5、铝-铝 0.3），恢复系数低值。
3. **几何流水线**：CoACD 凸分解（复用 `assets_src/real/convex_decomp_v5`）→ 球包/球审计 → `contact_semantics` 白名单（hand×object 允许、arm×object 禁止）。
4. **臂展**：UR5 在车道处竖直物体顶部 ≤1.10 m；DFX 爪抓需从上方进入；两只 Franka 腕在中线打架时物体需外移 y ∓0.35。
5. **物理旋钮**沿 r27 录制 yaml：手指执行器 100/5/5、solver 迭代 64、armature 0.05、双臂重力补偿。

## 3. 三个子族（映射到已验证原语）

### R34-A 4040 铝型材框架搭建（首选）

- **物体**：4×4 cm T 槽铝型材 30 / 45 / 60 cm（截面程序化生成、阳极氧化铝材质；或 Misumi / OpenBuilds STEP 转 USD）；
  角码 / 内置连接件（放大到 ≥3.5 cm 可捏，或只由 F2 处理）；U 桌上的底框夹具（固定体，起工位语义）。
- **任务**
  1. **型材递送**：F 中段捏（瓶族原语）→ 横向搬到车道 → U 侧 DFX 顶抓另一端（接力原语）。横握 30–60 cm 型材两手天然相距 ≥20 cm，
     即 R32(b) 的长物体条件，**直接双手同握交接不再需要跨机豁免**，这是本子族最大的附带收益。
  2. **共持插接**：U 扶稳底框，F 把型材端部插入角码（单边间隙 3–5 mm；复用 T3 peg-in-socket 的轴对准 / 横向对中 / 插入深度判据）。
     这是全库第一个「一方施力一方支撑」的接触密集协作。
  3. **框架共抬**：装好的矩形框四臂共抬平移（替代 obj_plank 的 four_lift），闭链耦合最强。
- **验收**：raw + a27 门控 success、零违规、≥2 机位成片（B 近景 / C 斜俯 / 新增沿车道机位）、VLM 逐帧 PASS；型材 obj−flange 恒定（真随手）；
  递送时两手最近距离 ≥ 跨机安全线 3 cm 且 alpha 全程 ≥0.9。

### R34-B KLT 料箱分拣（次选）

- **物体**：Isaac Sim Nucleus `Isaac/Props/KLT_Bin`（工业小型物流箱）；YCB（cracker box、mustard、power drill、large clamp、wood block）；
  Google Scanned Objects 工业小件（扳手、接头、盒）。
- **任务**：共享车道散放 8–12 件，四臂按类别分拣入各自料箱；随机流下抢件冲突最密，是离合器的展示台
  （对应 SAI 的 Laundry Collection 共享工作区干扰场景）。
- **验收**：每臂 ≥3 件入箱、零违规、介入不造成死锁；新增「抢件冲突数 / phase-sync 保持率」过程指标。

### R34-C 包装（备选）

- **物体**：真实纸箱网格（GSO）或带盖 KLT（关节盖）、待装物。
- **任务**：装箱 → 双臂共抬满箱 → 合盖（推压）。
- **验收**：装箱 ≥2 件、共抬 ≥10 cm、盖角 <10°。

## 4. 资产来源（优先级）

| 优先 | 来源 | 为什么 | 许可 |
|---|---|---|---|
| 1 | Isaac Sim Nucleus Props：`KLT_Bin`、YCB 全套、Blocks、工业托盘/仓储件 | USD 原生、物理已配、零转换 | Omniverse 资产许可（研究可用） |
| 2 | Isaac Lab Factory / AutoMate 装配资产（螺母螺栓 M16–M20、齿轮、100 对插头插座） | 接触密集装配在 PhysX 下已验证，公差可参数化 | BSD-3 / 数据集许可 |
| 3 | Google Scanned Objects（1000+ 扫描件） | 真实纹理与尺寸；OBJ→USD 一步 | CC-BY 4.0 |
| 4 | 程序化 T 槽型材（trimesh 截面挤出）/ Misumi、OpenBuilds STEP | 尺寸精确可控，4040 与已验证的 4 cm 捏取一致 | 自制 / 厂商 CAD 条款 |
| 5 | Objaverse(-XL) | 只作补充：版权与网格质量不齐，需人工筛选 | 逐件 |

## 5. 流水线与文件边界（不触碰原会话在改的文件）

1. `assets_src/tasks/r34/`：新资产 USD + `r34_assets.yaml`（沿 task_assets.yaml schema：质量 / 摩擦 / 随机化）。
2. `assets_src/real/convex_decomp_v5` 流程跑凸分解；球包进 `assets_src/spheres`；`contact_semantics` 增白名单条目（新增文件 patch，不改现有 yaml）。
3. `src/safeduo/delta/task_library_r34.py`：新文件，复用 `_screen/_screened_combos`、`IK_MULTISEED`、`PalmFrame`、PHYS 模式；
   任务：`profile_handover` / `profile_insert` / `frame_colift` / `klt_sort` / `pack_box`。
4. 设计期 IK 审计（CPU）→ GPU 探针（`--no_video`，GPU1 空档）→ 多机位成片 + VLM 质检 → 入 tasks.json 与 MASTER §9 R34 行。
5. 发车前把实验写进面板 `plan.json`；结果由 `update_dashboard.sh` 自动进四门矩阵。

## 6. 工作量与风险

- 资产采集/转换/审计：1–1.5 天（CPU）；R34-A 三任务设计与探针：1–2 天（含 GPU 空档等待）；成片与质检：0.5 天。
- 风险：① 角码等小件低于 DFX 可捏窗口 → 放大或只由 F2 处理；② T 槽边缘细节让凸分解件数暴涨 → 用简化碰撞体 + 精细视觉网格分离；
  ③ 插接的 3–5 mm 公差对 a27 门控下 1–2 cm 到位偏差敏感（bottle 顶抓已遇到）→ 角码开口做导向斜面，或把插接判据放宽到 8 mm 再收紧。

## 7. 待 owner 决定（面板 D5）

1. 启动时机：现在并行开工（只加新文件、不占 GPU）还是等原会话空闲。
2. 首个子族：A 型材（推荐）/ B 分拣 / C 包装。
3. 小件尺寸策略：角码放大到 ≥3.5 cm，还是保留真实尺寸只由 F2 处理。
