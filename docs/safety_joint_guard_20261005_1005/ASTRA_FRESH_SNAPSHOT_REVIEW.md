# ASTRA：第一fresh seed E1快照独立算术审查

时间：2026-10-05T04:09:42.719807+00:00；状态：**PASS_BOUNDED_FRESH_SNAPSHOT_REVIEW**。

已消费closed modes：envelope_guard, joint_guard；待审：无。18个快照 / 1152 env-snapshots / 4564项检查。

## 实际执行与界限

仅NumPy读取已确认protocol complete的cell和9个预注册step；未模拟、未导入torch/Isaac/production/helper。复用被冻结SHA的旧独立NumPy基础算术函数，但没有执行旧E0审计或改其文件。

reachable bounds从原soft limits−issued target、速度箱及q±.05重建；不可达时检查最近端点单点恢复。limited input/governor changed/返回命令/actual target delta、Jqd/pending/d_eff/cap、gate/rel、h/authority/minGu和alpha逐层核。

原solver info残差为producer记录：当该robot没有R19 bypass时可用返回u独立重算；有bypass时其内部pre-bypass解未保存，只保留原报告并核bypass arm逐位返回limited input，不能声称独立重建未知solver解。R19返回与真正target增量的全部安全/alpha/bounds残差分别计算。

binary64与CUDA float32算术比较abs≤5e−7；严格bounds、limited command、pending、gate等比较不放宽。该尺度不是物理门；正残差保留，诊断>tol计数用原1e−6。minGu单行可行不证明联合可行；batch-global passes不得当每env失败。

## 结果摘要

| mode/robot | original reported max | returned safety max | target safety max | returned alpha max | target alpha max | returned/target bound max | returned/target safety >1e−6 | bypass snapshots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| envelope_guard/F | 0.003303891281 | 0.003303891048 | 0.003303893154 | 1.199509537e-06 | 1.231622591e-06 | 0/2.235174179e-08 | 4/4 | 0 |
| envelope_guard/U | 0.0009478451684 | 0.0009478443794 | 0.0009478622475 | 5.560407074e-05 | 5.559865284e-05 | 0/1.415610313e-07 | 18/18 | 0 |
| joint_guard/F | 8.212064131e-06 | 8.211905103e-06 | 8.222321406e-06 | 1.199509537e-06 | 1.231622591e-06 | 0/2.235174179e-08 | 3/3 | 0 |
| joint_guard/U | 0.001514661592 | 0.001514661619 | 0.001514660864 | 0.0008369291514 | 0.0008368901291 | 0/1.415610313e-07 | 15/15 | 0 |

快照算术PASS不表示残差为零、PD保证、原official任务安全或策略批准；不从这些9帧重评分整960步结局。full9021 J仅producer检查，selected J在这里做有限快照复算；无相机/跨run轨迹验收。

两组各有1个不可达joint坐标，均核对了最近可达端点的单点恢复。U侧 `minGu−h>0` 的环境快照数分别为12（envelope）和11（joint），F均为0；这是环境×快照计数，不能当独立失败病例数。此处较窄reference bounds下，`h=max(raw_h,.9*minGu)` 在正minGu时仍可能小于minGu，因此不能把authority clamp称为逐行可行性保证。两组全部选定快照都没有R19 bypass，原solver报告残差可与实际返回命令独立核对。正安全/alpha残差及target浮点bounds残差均保留，尚不能据此认定唯一物理失败机制。

## 执行绑定

Oracle SHA：`938218b37a647a05c61993aabef0975ee8c94b5783c6ef3232a59fe4a2354e23`；基础独立oracle SHA：`e28d44bbcf1829705a8e5df96a87ff3a78c172190d513c6dfb9f5c2433336151`。
同名JSON保存每个消费文件SHA、每帧shape/最坏位置、原始正残差统计、全部比较最大误差及failure；只有完整两组才标最终18帧PASS。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/liyufeng/miniforge3/envs/safeduo/bin/python -B /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/ASTRA_FRESH_SNAPSHOT_REVIEW.py --include-joint
```

**production_promoted=false；hardware_approved=false；物理安全策略未批准。**
