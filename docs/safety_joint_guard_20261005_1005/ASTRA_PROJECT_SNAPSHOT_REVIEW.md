# ASTRA：两组开发 cell 的 selected projection 快照独立核验

审计时间：2026-10-05T03:21:26.398714+00:00。状态：**PASS_BOUNDED_PROJECT_SNAPSHOT_REVIEW**。

仅审 baseline_guard / admission_guard，seed60317411，既定9步 `0,60,62,66,71,75,180,480,959`，18个快照×64槽=1152 env-snapshots。两个开发 modes均为E0；不验收未完成的fresh768、E1/envelope/joint或camera，不重复完整analyzer/QA/NAS封存。

独立辅助程序只用NumPy读取冻结快照、对应dense必要字段和diagnostics；未导入torch、Isaac、runner或原diagnostic helper，未模拟/修改源。本次只写本MD与同名.py。Oracle执行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/liyufeng/miniforge3/envs/safeduo/bin/python -B /home/liyufeng/safeduo/artifacts/safety_joint_guard_20261005_1005/ASTRA_PROJECT_SNAPSHOT_REVIEW.py
```

## 已实际执行的检查

1. closed状态、mode/E0、完整9快照集合、全部snapshot数值finite、selected ID/valid/padding/唯一/范围。同次dense raw/inside/returned/outer命令、target-after minus-before、pre debt/qd与pending project history相对q的绑定均用严格相等。
2. 以binary64独立计算 selected J·qd 和全部6 pending backlogs的最不利位移，核d_eff；核原cap公式。engage/structural/conditional gate在保存的float32操作数上用原严格分支，rel再按实际arm involvement核。
3. E0原bounds以soft limits−issued target与速度箱相交逐位重建；它不是纯速度箱。核G=−J、cross的p分配与负cap不分摊、backlog_aware=false时零debit、authority前后h、bounds内minGu、alpha方向/预算/原rel。
4. 不重求解：对R19后的实际返回u和真正target增量分别计算max(Gu−h,0)、alpha及bounds残差，对照producer。非bypass robot才额外比原solver残差；bypass造成的原solver/returned区别保留。核最坏行和individual infeasibility及passes标志。

共执行4490项比较/条件；记录0项不一致。跨binary64 oracle与CUDA float32运算的数值一致性门为abs≤5e-07，不声称bit-exact；它仅是算术比较尺度，不是物理安全门。正残差原样保留；诊断超过solver tol计数使用原1e−6，未更改official负裕度门。

## 数值结果

| mode/robot | returned safety max | actual target safety max | returned alpha max | target alpha max | returned bound max | target bound max | returned/target safety >1e−6 | solver=0且returned>1e−6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_guard/F | 0.01374986175 | 0.01374983509 | 6.191230616e-05 | 6.193297235e-05 | 0 | 1.415610313e-07 | 7/7 | 0 |
| baseline_guard/U | 0.0007963446891 | 0.0007963443794 | 0.0003000329586 | 0.0002999729949 | 0 | 1.415610313e-07 | 19/19 | 0 |
| admission_guard/F | 0.01077116674 | 0.01077119404 | 6.571005029e-09 | 3.878032118e-09 | 0 | 1.415610313e-07 | 7/6 | 0 |
| admission_guard/U | 0.007125369736 | 0.007125378082 | 5.89759233e-05 | 5.896569151e-05 | 0 | 1.415610313e-07 | 17/17 | 0 |

算术一致性PASS不意味着这些残差为零或输出安全。R19出口、target clamp与PD物理响应是不同阶段。个别minGu可行不证明联合约束/alpha交集可行；batch-global passes达到上限不能解释成每个环境都失败。

## 可复核的完整结果与输入绑定

以下JSON保留18帧宽度、最坏env/selected位置/full row、所有比较最大误差、失败项、统计及输入SHA。所读取cell整体SHA用于绑定字节；计算只消费上述snapshot/dense必要字段，未重评分整个cell结局。

```json
{
  "status": "PASS_BOUNDED_PROJECT_SNAPSHOT_REVIEW",
  "utc": "2026-10-05T03:21:26.398714+00:00",
  "seed": 60317411,
  "snapshot_files": 18,
  "env_snapshots": 1152,
  "checks": 4490,
  "arithmetic_atol": 5e-07,
  "failures": [],
  "maximum_errors": {
    "d_eff_from_J_qd_history": 2.0296639480221756e-07,
    "cap_from_saved_d_eff": 2.7413070181891985e-09,
    "min_Gu_F": 8.01847237219544e-09,
    "raw_h_F": 8.290194086768565e-10,
    "post_authority_h_F": 8.290194086768565e-10,
    "alpha_G_F": 9.209512097374528e-08,
    "alpha_h_F": 5.758793997223677e-09,
    "returned_safety_residual_F": 1.522363998751608e-09,
    "returned_alpha_residual_F": 1.8108363723443044e-10,
    "returned_bound_residual_F": 0.0,
    "target_safety_residual_F": 1.2116696049280229e-09,
    "target_alpha_residual_F": 1.0618528278882877e-10,
    "target_bound_residual_F": 0.0,
    "original_residual_without_robot_bypass_F": 1.522363998751608e-09,
    "individual_infeasibility_F": 0.0,
    "worst_row_value_F": 8.885743263149087e-10,
    "min_Gu_U": 9.564067579570512e-09,
    "raw_h_U": 1.1237228925153886e-09,
    "post_authority_h_U": 1.7294133833334335e-09,
    "alpha_G_U": 7.929874557000005e-08,
    "alpha_h_U": 4.352160222487189e-09,
    "returned_safety_residual_U": 1.3784447541742573e-09,
    "returned_alpha_residual_U": 7.400668133872301e-10,
    "returned_bound_residual_U": 0.0,
    "target_safety_residual_U": 2.289874068139852e-09,
    "target_alpha_residual_U": 9.973045722588836e-10,
    "target_bound_residual_U": 0.0,
    "original_residual_without_robot_bypass_U": 1.3784447541742573e-09,
    "individual_infeasibility_U": 0.0,
    "worst_row_value_U": 1.2699195618692816e-09
  },
  "summaries": [
    {
      "mode": "baseline_guard",
      "robots": {
        "F": {
          "returned_safety_max": 0.013749861749472,
          "target_safety_max": 0.013749835088355589,
          "returned_alpha_max": 6.19123061616142e-05,
          "target_alpha_max": 6.193297235412842e-05,
          "returned_bound_max": 0.0,
          "target_bound_max": 1.4156103134155273e-07,
          "returned_safety_gt_solver_tol": 7,
          "target_safety_gt_solver_tol": 7,
          "original_solver_zero_returned_gt_tol": 0,
          "no_bypass_original_vs_returned_max_error": 1.522363998751608e-09,
          "bypass_env_snapshots": 0,
          "passes_hit_limit_env_snapshots": 512,
          "individual_infeasible_env_snapshots": 0
        },
        "U": {
          "returned_safety_max": 0.0007963446891393125,
          "target_safety_max": 0.0007963443793973513,
          "returned_alpha_max": 0.0003000329585648026,
          "target_alpha_max": 0.0002999729949495489,
          "returned_bound_max": 0.0,
          "target_bound_max": 1.4156103134155273e-07,
          "returned_safety_gt_solver_tol": 19,
          "target_safety_gt_solver_tol": 19,
          "original_solver_zero_returned_gt_tol": 0,
          "no_bypass_original_vs_returned_max_error": 1.3784447541742573e-09,
          "bypass_env_snapshots": 0,
          "passes_hit_limit_env_snapshots": 448,
          "individual_infeasible_env_snapshots": 0
        }
      }
    },
    {
      "mode": "admission_guard",
      "robots": {
        "F": {
          "returned_safety_max": 0.010771166738418152,
          "target_safety_max": 0.010771194043478372,
          "returned_alpha_max": 6.571005028727789e-09,
          "target_alpha_max": 3.878032117654584e-09,
          "returned_bound_max": 0.0,
          "target_bound_max": 1.4156103134155273e-07,
          "returned_safety_gt_solver_tol": 7,
          "target_safety_gt_solver_tol": 6,
          "original_solver_zero_returned_gt_tol": 0,
          "no_bypass_original_vs_returned_max_error": 8.306684441856949e-10,
          "bypass_env_snapshots": 0,
          "passes_hit_limit_env_snapshots": 576,
          "individual_infeasible_env_snapshots": 0
        },
        "U": {
          "returned_safety_max": 0.00712536973578334,
          "target_safety_max": 0.007125378082258216,
          "returned_alpha_max": 5.897592330285882e-05,
          "target_alpha_max": 5.896569150720676e-05,
          "returned_bound_max": 0.0,
          "target_bound_max": 1.4156103134155273e-07,
          "returned_safety_gt_solver_tol": 17,
          "target_safety_gt_solver_tol": 17,
          "original_solver_zero_returned_gt_tol": 0,
          "no_bypass_original_vs_returned_max_error": 1.3784447541742573e-09,
          "bypass_env_snapshots": 0,
          "passes_hit_limit_env_snapshots": 448,
          "individual_infeasible_env_snapshots": 0
        }
      }
    }
  ],
  "snapshot_details": [
    {
      "mode": "baseline_guard",
      "step": 0,
      "width": 60,
      "valid_rows": 933,
      "F": {
        "returned_safety_max": 1.116876018752988e-10,
        "target_safety_max": 6.255722001524688e-08,
        "worst_env": 29,
        "worst_selected_position": 5,
        "worst_full_row_id": 4503,
        "original_solver_residual_at_worst": 1.1641532182693481e-10,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 5.19846207096622e-11,
        "target_safety_max": 3.9346590097188994e-08,
        "worst_env": 13,
        "worst_selected_position": 5,
        "worst_full_row_id": 2103,
        "original_solver_residual_at_worst": 5.820766091346741e-11,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "baseline_guard",
      "step": 60,
      "width": 24,
      "valid_rows": 809,
      "F": {
        "returned_safety_max": 5.999986583959949e-10,
        "target_safety_max": 2.592242154264568e-08,
        "worst_env": 6,
        "worst_selected_position": 8,
        "worst_full_row_id": 6183,
        "original_solver_residual_at_worst": 1.1641532182693481e-09,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 4.703396035057178e-09,
        "target_safety_max": 1.1200149335754972e-08,
        "worst_env": 2,
        "worst_selected_position": 12,
        "worst_full_row_id": 6586,
        "original_solver_residual_at_worst": 5.005858838558197e-09,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "baseline_guard",
      "step": 62,
      "width": 24,
      "valid_rows": 809,
      "F": {
        "returned_safety_max": 1.2382484065875943e-09,
        "target_safety_max": 4.294747674160604e-08,
        "worst_env": 6,
        "worst_selected_position": 8,
        "worst_full_row_id": 6183,
        "original_solver_residual_at_worst": 1.5133991837501526e-09,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 7.777657858032416e-10,
        "target_safety_max": 6.76747891059648e-08,
        "worst_env": 17,
        "worst_selected_position": 0,
        "worst_full_row_id": 2227,
        "original_solver_residual_at_worst": 8.149072527885437e-10,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "baseline_guard",
      "step": 66,
      "width": 24,
      "valid_rows": 809,
      "F": {
        "returned_safety_max": 1.10278563928215e-09,
        "target_safety_max": 4.267629383214455e-08,
        "worst_env": 6,
        "worst_selected_position": 8,
        "worst_full_row_id": 6183,
        "original_solver_residual_at_worst": 6.984919309616089e-10,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 1.7609620810155047e-09,
        "target_safety_max": 3.0160311840880805e-08,
        "worst_env": 17,
        "worst_selected_position": 0,
        "worst_full_row_id": 2227,
        "original_solver_residual_at_worst": 1.979060471057892e-09,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "baseline_guard",
      "step": 71,
      "width": 47,
      "valid_rows": 1187,
      "F": {
        "returned_safety_max": 1.1080702345261928e-09,
        "target_safety_max": 3.110626217717183e-08,
        "worst_env": 25,
        "worst_selected_position": 5,
        "worst_full_row_id": 4814,
        "original_solver_residual_at_worst": 9.313225746154785e-10,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 0.00010390496889003953,
        "target_safety_max": 0.00010389814693656518,
        "worst_env": 54,
        "worst_selected_position": 3,
        "worst_full_row_id": 8968,
        "original_solver_residual_at_worst": 0.0001039053313434124,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "baseline_guard",
      "step": 75,
      "width": 130,
      "valid_rows": 1350,
      "F": {
        "returned_safety_max": 0.00012829472942600065,
        "target_safety_max": 0.00012831413387236375,
        "worst_env": 28,
        "worst_selected_position": 9,
        "worst_full_row_id": 3879,
        "original_solver_residual_at_worst": 0.00012829434126615524,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 3.7765963578653e-05,
        "target_safety_max": 3.7769288734307294e-05,
        "worst_env": 29,
        "worst_selected_position": 18,
        "worst_full_row_id": 4355,
        "original_solver_residual_at_worst": 3.776606172323227e-05,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "baseline_guard",
      "step": 180,
      "width": 38,
      "valid_rows": 1290,
      "F": {
        "returned_safety_max": 0.013749861749472,
        "target_safety_max": 0.013749835088355589,
        "worst_env": 62,
        "worst_selected_position": 35,
        "worst_full_row_id": 8837,
        "original_solver_residual_at_worst": 0.013749860227108002,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 9.016451664692338e-06,
        "target_safety_max": 9.059115299692166e-06,
        "worst_env": 62,
        "worst_selected_position": 1,
        "worst_full_row_id": 4039,
        "original_solver_residual_at_worst": 9.017065167427063e-06,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "baseline_guard",
      "step": 480,
      "width": 36,
      "valid_rows": 1373,
      "F": {
        "returned_safety_max": 2.1918848050496475e-05,
        "target_safety_max": 2.1934248583317054e-05,
        "worst_env": 29,
        "worst_selected_position": 21,
        "worst_full_row_id": 8766,
        "original_solver_residual_at_worst": 2.1918835045653395e-05,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 5.299063239191937e-06,
        "target_safety_max": 5.291159222320374e-06,
        "worst_env": 6,
        "worst_selected_position": 21,
        "worst_full_row_id": 9016,
        "original_solver_residual_at_worst": 5.298759788274765e-06,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "baseline_guard",
      "step": 959,
      "width": 32,
      "valid_rows": 1353,
      "F": {
        "returned_safety_max": 7.532786205979858e-07,
        "target_safety_max": 7.308159908347989e-07,
        "worst_env": 32,
        "worst_selected_position": 21,
        "worst_full_row_id": 8834,
        "original_solver_residual_at_worst": 7.532653398811817e-07,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 0.0007963446891393125,
        "target_safety_max": 0.0007963443793973513,
        "worst_env": 39,
        "worst_selected_position": 10,
        "worst_full_row_id": 7049,
        "original_solver_residual_at_worst": 0.0007963447133079171,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 0,
      "width": 60,
      "valid_rows": 933,
      "F": {
        "returned_safety_max": 1.116876018752988e-10,
        "target_safety_max": 6.255722001524688e-08,
        "worst_env": 29,
        "worst_selected_position": 5,
        "worst_full_row_id": 4503,
        "original_solver_residual_at_worst": 1.1641532182693481e-10,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 5.19846207096622e-11,
        "target_safety_max": 3.9346590097188994e-08,
        "worst_env": 13,
        "worst_selected_position": 5,
        "worst_full_row_id": 2103,
        "original_solver_residual_at_worst": 5.820766091346741e-11,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 60,
      "width": 32,
      "valid_rows": 891,
      "F": {
        "returned_safety_max": 5.999986583959949e-10,
        "target_safety_max": 2.4672336285114227e-08,
        "worst_env": 6,
        "worst_selected_position": 8,
        "worst_full_row_id": 6183,
        "original_solver_residual_at_worst": 1.1641532182693481e-09,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 4.703396035057178e-09,
        "target_safety_max": 1.5858021784609377e-08,
        "worst_env": 2,
        "worst_selected_position": 12,
        "worst_full_row_id": 6586,
        "original_solver_residual_at_worst": 5.005858838558197e-09,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 62,
      "width": 78,
      "valid_rows": 1013,
      "F": {
        "returned_safety_max": 1.2382484065875943e-09,
        "target_safety_max": 4.294747674160604e-08,
        "worst_env": 6,
        "worst_selected_position": 8,
        "worst_full_row_id": 6183,
        "original_solver_residual_at_worst": 1.5133991837501526e-09,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 6.007412645982202e-07,
        "target_safety_max": 5.814786062652288e-07,
        "worst_env": 20,
        "worst_selected_position": 20,
        "worst_full_row_id": 1915,
        "original_solver_residual_at_worst": 6.00797648075968e-07,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 66,
      "width": 42,
      "valid_rows": 937,
      "F": {
        "returned_safety_max": 2.803256404781962e-09,
        "target_safety_max": 4.267629383214455e-08,
        "worst_env": 35,
        "worst_selected_position": 2,
        "worst_full_row_id": 4623,
        "original_solver_residual_at_worst": 2.8230715543031693e-09,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 1.7229021570530634e-08,
        "target_safety_max": 6.324054835360471e-08,
        "worst_env": 29,
        "worst_selected_position": 7,
        "worst_full_row_id": 4183,
        "original_solver_residual_at_worst": 1.7724232748150826e-08,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 71,
      "width": 43,
      "valid_rows": 1145,
      "F": {
        "returned_safety_max": 9.719379467920675e-10,
        "target_safety_max": 4.808660403199383e-08,
        "worst_env": 20,
        "worst_selected_position": 27,
        "worst_full_row_id": 2459,
        "original_solver_residual_at_worst": 7.566995918750763e-10,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 4.1254985593472404e-05,
        "target_safety_max": 4.12972786332233e-05,
        "worst_env": 54,
        "worst_selected_position": 5,
        "worst_full_row_id": 8970,
        "original_solver_residual_at_worst": 4.1255028918385506e-05,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 75,
      "width": 49,
      "valid_rows": 1197,
      "F": {
        "returned_safety_max": 7.991460613913925e-10,
        "target_safety_max": 2.3838005680509866e-08,
        "worst_env": 6,
        "worst_selected_position": 6,
        "worst_full_row_id": 6183,
        "original_solver_residual_at_worst": 1.6298145055770874e-09,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 1.2458335667065237e-05,
        "target_safety_max": 1.2462927759804288e-05,
        "worst_env": 31,
        "worst_selected_position": 12,
        "worst_full_row_id": 8492,
        "original_solver_residual_at_worst": 1.2458302080631256e-05,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 180,
      "width": 32,
      "valid_rows": 1262,
      "F": {
        "returned_safety_max": 7.295291765346519e-06,
        "target_safety_max": 7.313629092333025e-06,
        "worst_env": 51,
        "worst_selected_position": 20,
        "worst_full_row_id": 8823,
        "original_solver_residual_at_worst": 7.295311661437154e-06,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 2.1510840735828804e-06,
        "target_safety_max": 2.140885343671073e-06,
        "worst_env": 19,
        "worst_selected_position": 15,
        "worst_full_row_id": 7110,
        "original_solver_residual_at_worst": 2.1511223167181015e-06,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 480,
      "width": 213,
      "valid_rows": 1580,
      "F": {
        "returned_safety_max": 1.245602252073917e-05,
        "target_safety_max": 1.2492170131489858e-05,
        "worst_env": 29,
        "worst_selected_position": 22,
        "worst_full_row_id": 8766,
        "original_solver_residual_at_worst": 1.2455915566533804e-05,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 0.004051611277702115,
        "target_safety_max": 0.004051602932080733,
        "worst_env": 32,
        "worst_selected_position": 115,
        "worst_full_row_id": 7409,
        "original_solver_residual_at_worst": 0.004051610827445984,
        "robot_bypass_at_worst": false
      }
    },
    {
      "mode": "admission_guard",
      "step": 959,
      "width": 185,
      "valid_rows": 1748,
      "F": {
        "returned_safety_max": 0.010771166738418152,
        "target_safety_max": 0.010771194043478372,
        "worst_env": 39,
        "worst_selected_position": 28,
        "worst_full_row_id": 827,
        "original_solver_residual_at_worst": 0.010771166533231735,
        "robot_bypass_at_worst": false
      },
      "U": {
        "returned_safety_max": 0.00712536973578334,
        "target_safety_max": 0.007125378082258216,
        "worst_env": 39,
        "worst_selected_position": 88,
        "worst_full_row_id": 7910,
        "original_solver_residual_at_worst": 0.0071253702044487,
        "robot_bypass_at_worst": false
      }
    }
  ],
  "input_sha256": [
    {
      "mode": "baseline_guard",
      "file": "protocol.json",
      "sha256": "059a0b11ca5a6748a6d64c59ff9ecbd8374c07cde1b5d9e28a20edd100d5cef0"
    },
    {
      "mode": "baseline_guard",
      "file": "guard_metadata.json",
      "sha256": "df07eb0954a3287b35cee7bece0032d34fab8b41679195cc7a8ce906bc417d21"
    },
    {
      "mode": "baseline_guard",
      "file": "cell_001.npz",
      "sha256": "167e99aaab4daaa06b7594042515bcb34d0f997197b8e92176b0eb58040467cf"
    },
    {
      "mode": "baseline_guard",
      "file": "project_diagnostics.npz",
      "sha256": "b3a52f5c024098920f2921ee1b6a9efa89d9f0069bffdb5544800b65d6b6d349"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0000.npz",
      "sha256": "fbe690fea5cb1710699a47b6de99f3eb29ac2a725cbe94dd09edb946ab21193b"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0060.npz",
      "sha256": "c4589f4b7caf5377178db7cd6a9979dbb3339165e8a766947bd77d037164b986"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0062.npz",
      "sha256": "60c5df5641d84bddb50f83b3cd806f9295189a72ebc423319aa25ab9673ca241"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0066.npz",
      "sha256": "28a2b08a15b24630ef8b9635319782ed6bef490ce07560594fcfbdd3e7b414a4"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0071.npz",
      "sha256": "c06cd2fbe614e3687c8228bde61bccf0ec6238ed2e5ba075aa93012b1037d6c1"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0075.npz",
      "sha256": "0975fc3fb94e36d027777544f8b665556dbaa1843ef34cd6bb102dccef88f609"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0180.npz",
      "sha256": "1dfce2f849781e7f1e168a13465ffa8030a1278770a5fa6bdf6a5dcb2f6addb1"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0480.npz",
      "sha256": "8a6a692ffa26e0165cd2debdc15dac08ed71555755228dc8b0485e6258753893"
    },
    {
      "mode": "baseline_guard",
      "file": "projection_snapshots/step_0959.npz",
      "sha256": "c77909649dc2d53793511a37a6671fc53fb092cc39121fc5f46066985878a211"
    },
    {
      "mode": "admission_guard",
      "file": "protocol.json",
      "sha256": "c1ba896138f65a053e67fd3efff24c3d4e42cb0e17fc15a56d00968ea085450c"
    },
    {
      "mode": "admission_guard",
      "file": "guard_metadata.json",
      "sha256": "81b3bb25204933fe3adc15abc8ae302b975c05a8c1b5ff86f1726b76b2caf231"
    },
    {
      "mode": "admission_guard",
      "file": "cell_001.npz",
      "sha256": "b0df3f82261c31053b35cc04349a06cad5665e5f2d3be0cb679c5f9e8991bd30"
    },
    {
      "mode": "admission_guard",
      "file": "project_diagnostics.npz",
      "sha256": "048402082e2ba353c918b1c72de49910ec8f4dfad312b13b2afb83a312c8ba99"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0000.npz",
      "sha256": "64c005c60c3cce9a66faff3759273a77d67a8d31d6bbbe91a519fd51f76007c3"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0060.npz",
      "sha256": "0ad91233f85119a85206993a0a786fbf48fe146b95bb35e43f2b4a13daf93356"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0062.npz",
      "sha256": "51f842f8f5a26f3a1c82abb2488f6d479fc006b6ca74d15c640e5cc006f7cbe8"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0066.npz",
      "sha256": "f26699e9940cdac73c69269917f242b31f66f9085d3560354fea1afa21c53946"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0071.npz",
      "sha256": "1015cff19545b3cd77f1b7c95a8da9b6b25fb0c071c629d3cc2c98d18f1b349c"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0075.npz",
      "sha256": "106ce868381701f8ae0cf52a7822709678de1466e31efa49387f2b6ba98f0e4a"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0180.npz",
      "sha256": "f1f39a9524f2bc047bf6c9c39b2ce301391f7d955f48abfef14b3737ee82675a"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0480.npz",
      "sha256": "6e4dfd1460f91d7b2423e09ee6ef83b8a8a35c90f91d6bdec779934a68c5efe5"
    },
    {
      "mode": "admission_guard",
      "file": "projection_snapshots/step_0959.npz",
      "sha256": "97a7075bb73e126f1acddcd858852a3c967b703bf61900a8db0b1c5af95742fd"
    }
  ],
  "oracle_sha256": "e28d44bbcf1829705a8e5df96a87ff3a78c172190d513c6dfb9f5c2433336151",
  "production_promoted": false,
  "hardware_approved": false
}
```

## Oracle自身修订及结论界限

首轮oracle错误假设E0为纯速度箱，4490项比较中28个bounds条件失败；其余算术比较吻合。读取生产DuoEnv.project调用确认原project_target_limits仍提供soft limits−issued target后，只修订本独立oracle并重跑：这28项不是runner缺陷，最终门使用原bounds严格相等，未放宽比较尺度。首轮命令输出保留在本会话；原始快照、candidate、生产源均未改变。

本审计核验的是18组真实selected projection输入/输出及其诊断算术，full9021 J仅producer检查，未全程离线重建。本两组E0没有exercise实际reference-limited bounds或joint组合。不独立重建几何/FK或豁免来源，不证明非线性PD兑现约束，不增加fresh样本、不读camera或holdout结局。后续完整结果与camera终检另审；旧报告、candidate/production/helper/tests均未修改。**production_promoted=false；hardware_approved=false；安全策略未批准。**
