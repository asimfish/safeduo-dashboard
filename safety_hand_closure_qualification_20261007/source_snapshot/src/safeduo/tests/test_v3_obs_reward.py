"""v3 P0/P2 回归测试（A5，Mac cpu 可跑）：

1. pair_obs_features：观测端 pair 特征化（去 raw pair_id 的出生饱和根治，
   v3_recipe_proposal 1a 节）。one-hot 正确性、padding 清零、幅值回到物理量级。
2. p 翻转罚符号机：duo_env._get_rewards 内联的三行状态机必须与
   eval/metrics._priority_flips（PRIORITY_FLIP 口径，死区 +-0.1）逐位一致
   ——duo_env 本身 Mac 上 import 不了（isaaclab），此处镜像其实现钉住口径；
   改 duo_env 的翻转行为必须同步这里。
"""

import pytest
import torch

from safeduo.eval.metrics import _priority_flips
from safeduo.safety.types import (
    CLASS_CROSS,
    CLASS_SELF,
    CLASS_TABLE,
    MAX_ACTIVE_PAIRS,
    PAIR_FEATURE_DIM,
    TOTAL_DOF,
    pair_obs_features,
)


def _pairs(n=2, m=4):
    ap = torch.zeros(n, m, 4)
    mask = torch.zeros(n, m, dtype=torch.bool)
    return ap, mask


def test_pair_features_shape_and_obs_dim_formula():
    n, m = 3, MAX_ACTIVE_PAIRS
    ap, mask = _pairs(n, m)
    out = pair_obs_features(ap, mask)
    assert out.shape == (n, m * PAIR_FEATURE_DIM)
    # duo_env.make_duo_env_cfg 的 coordinator obs_dim 公式（M=32 -> 275）
    obs_dim = 2 * TOTAL_DOF + m * PAIR_FEATURE_DIM + m + TOTAL_DOF + 5
    assert obs_dim == 275


def test_pair_features_onehot_and_physical_columns():
    ap, mask = _pairs(1, 4)
    rows = [(0.05, 0.3, CLASS_CROSS), (0.02, -0.1, CLASS_SELF),
            (-0.003, 0.7, CLASS_TABLE), (0.08, 0.0, CLASS_CROSS)]
    for i, (d, cv, cls) in enumerate(rows):
        ap[0, i, 0], ap[0, i, 1], ap[0, i, 2] = d, cv, cls
        ap[0, i, 3] = 1000.0 + i          # raw pair_id：不得进观测
        mask[0, i] = True
    out = pair_obs_features(ap, mask).view(4, PAIR_FEATURE_DIM)
    onehot_of = {CLASS_CROSS: (1, 0, 0), CLASS_SELF: (0, 1, 0),
                 CLASS_TABLE: (0, 0, 1)}
    for i, (d, cv, cls) in enumerate(rows):
        assert out[i, 0].item() == pytest.approx(d)
        assert out[i, 1].item() == pytest.approx(cv)
        assert tuple(out[i, 2:].tolist()) == onehot_of[cls]
    # 出生饱和根治的可测表述：pair_id 幅值（>=1000）不得泄漏进任何观测列
    assert out.abs().max().item() <= 1.0


def test_pair_features_padding_rows_all_zero():
    """padding 约定 pair_id=-1 / class_id=0：不乘 mask 会把 padding 行的
    one-hot 点亮成 cross 指示（32 行 padding = 常亮伪特征）。"""
    ap, mask = _pairs(2, 6)
    ap[..., 3] = -1.0                      # sphere_distance 的 padding 约定
    ap[0, 0] = torch.tensor([0.04, 0.2, CLASS_SELF, 7.0])
    mask[0, 0] = True
    out = pair_obs_features(ap, mask).view(2, 6, PAIR_FEATURE_DIM)
    assert (out[0, 1:] == 0).all() and (out[1] == 0).all()
    assert tuple(out[0, 0, 2:].tolist()) == (0, 1, 0)


def _env_flip_machine(p_trace: torch.Tensor) -> torch.Tensor:
    """duo_env._get_rewards 的 p 翻转状态机逐行镜像（那边 3 行 + reset 清零）。"""
    T, N = p_trace.shape
    sign_prev = torch.zeros(N)
    flips = torch.zeros(T, N, dtype=torch.bool)
    for t in range(T):
        p = p_trace[t]
        sgn = torch.where(p > 0.1, 1.0, torch.where(p < -0.1, -1.0, 0.0))
        flips[t] = (sgn != 0) & (sign_prev != 0) & (sgn != sign_prev)
        sign_prev = torch.where(sgn != 0, sgn, sign_prev)
    return flips


def test_flip_machine_matches_metrics_priority_flip():
    g = torch.Generator().manual_seed(11)
    p = (torch.randn(400, 16, generator=g) * 0.5).clamp(-1, 1)
    assert torch.equal(_env_flip_machine(p), _priority_flips(p, deadband=0.1))


def test_flip_machine_deadband_and_first_excursion():
    # 死区内震荡不计；首次越区不计（无前一符号）；跨死区翻转计一次
    p = torch.tensor([[0.05], [-0.08], [0.5], [0.09], [-0.3], [-0.9], [0.2]])
    flips = _env_flip_machine(p)
    assert flips.sum().item() == 2
    assert flips[4, 0].item() and flips[6, 0].item()
    assert torch.equal(flips, _priority_flips(p, deadband=0.1))
