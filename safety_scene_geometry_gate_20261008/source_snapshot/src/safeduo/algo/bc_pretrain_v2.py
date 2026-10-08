"""BC pretrain v2: obs normalization + Beta-NLL heads + Gaussian-compat export.

Fixes the two measured failure modes of the W2 `bc_pretrain` pipeline (A3 W3
sec.4 + sec.8, artifacts/warmstart/server_assembly_20260811_214843):

  1. UNNORMALIZED 243-dim obs: feature magnitudes span orders of magnitude
     (joint angles ~1 rad, margins ~0.1 m, deltas ~0.01), the trunk grows
     large features and passively saturates whatever sits behind tanh/sigmoid
     -> alpha-head MSE 0.299 WORSE than the 0.219 mean baseline on the
     aligned-pose dataset, p head 100% saturated at +1.
  2. tanh/sigmoid + MSE parameterization: 97% zero p labels give no restoring
     gradient once the shared trunk drifts (stable pathology, reproduced
     across batch/lr/seed).

  v2 pipeline:
  - ObsNormalizer: dataset mean/std, serialized into the checkpoint. The
    exported compat network FOLDS it into its first Linear layer, so PPO
    consumes RAW observations -- consistent with train_ppo, which runs
    WITHOUT rsl_rl empirical normalization (checked make_runner_cfg).
  - CoordinatorBetaMLP: the CoordinatorMLP ELU trunk (state-dict keys
    body.0/body.2 kept bit-compatible with A3's train/bc_init.py transplant
    contract) + a 10-param Beta head over (alpha_1..4, p'), p' = (p+1)/2.
    Same bounded-support rationale as algo/beta_actor (W3 double_hand port,
    same softplus + min_concentration parameterization): the Beta NLL has
    well-defined gradients at the support edges where MSE+squash saturates,
    and a future Beta-actor PPO (c2_lagrangian_v1) can warm-start from the
    Beta head directly.
  - Weighted NLL: the p dimension is upweighted on conflict-labeled samples
    (p* != 0) to counter the 97/3 imbalance; alpha dims stay uniform.
  - export_mlp_compat: distills the trained Beta means back into the LEGACY
    CoordinatorMLP format (logit/atanh least squares on trunk features +
    normalizer folded into body.0) and stores it under "state_dict", so
    A3's bc_init.py consumes a v2 checkpoint UNCHANGED, now with a real
    (non-degenerate) p row -- --bc_p_neutral should no longer be needed.

Acceptance gates (the work order pins these before handoff, computed on a
holdout split and stored in the checkpoint):
  alpha_mse < 0.75 * mean-baseline MSE
  p direction accuracy on conflict-labeled samples > 0.80
  (the run-level no-freeze gate lives in algo/bc_smoke_check.py)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Beta

LABEL_EPS = 1e-3      # Beta support squeeze: raw' = raw*(1-2e)+e
MIN_CONCENTRATION = 1.0


class ObsNormalizer:
    """Static dataset mean/std; serializable; foldable into a Linear layer."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-3)   # constant columns stay sane

    @classmethod
    def fit(cls, obs: torch.Tensor) -> "ObsNormalizer":
        return cls(obs.mean(dim=0), obs.std(dim=0))

    def apply(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean.to(obs)) / self.std.to(obs)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "ObsNormalizer":
        return cls(sd["mean"], sd["std"])


class CoordinatorBetaMLP(nn.Module):
    """CoordinatorMLP trunk (ELU, keys body.0/body.2) + Beta parameter head.

    Deterministic outputs = Beta means: alpha in (0,1)^4, p in (-1,1).
    """

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim, self.hidden = obs_dim, hidden
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.head_beta = nn.Linear(hidden, 10)
        nn.init.orthogonal_(self.head_beta.weight, gain=0.01)
        nn.init.zeros_(self.head_beta.bias)

    def concentrations(self, obs_n: torch.Tensor) -> tuple:
        z = self.body(obs_n)
        c1_raw, c0_raw = self.head_beta(z).chunk(2, dim=-1)
        c1 = torch.nn.functional.softplus(c1_raw) + MIN_CONCENTRATION
        c0 = torch.nn.functional.softplus(c0_raw) + MIN_CONCENTRATION
        return c1, c0

    def forward(self, obs_n: torch.Tensor) -> tuple:
        """-> (alpha_mean (N,4), p_mean (N,)) deterministic heads."""
        c1, c0 = self.concentrations(obs_n)
        mean = c1 / (c1 + c0)
        return mean[:, :4], mean[:, 4] * 2.0 - 1.0


def labels_to_raw(alpha_star: torch.Tensor, p_star: torch.Tensor,
                  eps: float = LABEL_EPS) -> torch.Tensor:
    """(alpha*, p*) -> squeezed (N,5) Beta-support labels in (0,1)."""
    raw = torch.cat([alpha_star, (p_star.unsqueeze(-1) + 1.0) * 0.5], dim=-1)
    return raw.clamp(0.0, 1.0) * (1.0 - 2.0 * eps) + eps


def beta_nll(c1: torch.Tensor, c0: torch.Tensor, raw: torch.Tensor,
             p_conflict_weight: torch.Tensor) -> torch.Tensor:
    """Per-dim Beta NLL; p dim (4) weighted per sample, alpha dims uniform."""
    nll = -Beta(c1, c0, validate_args=False).log_prob(raw)   # (N, 5)
    w = torch.ones_like(nll)
    w[:, 4] = p_conflict_weight
    return (nll * w).sum(dim=-1).mean()


@dataclass
class BCV2Config:
    epochs: int = 40
    batch_size: int = 2048
    hidden: int = 256
    lr: float = 3e-4
    p_conflict_weight: float = 4.0   # extra weight on p dim where p* != 0
    # conflict-labeled samples are ~3% of real datasets; pure loss weighting
    # leaves the p head at the Beta-NLL "safe" optimum (uniform, mean 0.5)
    # because sparse two-sided gradients cancel in expectation while the
    # trunk chases the alpha task. Guaranteeing a conflict quota per batch
    # keeps the direction signal dense enough to shape the trunk features.
    oversample_conflict: float = 0.25
    holdout_frac: float = 0.1
    seed: int = 0
    normalize: bool = True           # ablation switch


def _split(n: int, holdout_frac: float, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    k = max(1, int(n * holdout_frac))
    return perm[k:], perm[:k]


def evaluate_bc(model: CoordinatorBetaMLP, normalizer: "ObsNormalizer | None",
                obs: torch.Tensor, alpha_star: torch.Tensor,
                p_star: torch.Tensor, device: str = "cpu") -> dict:
    """Acceptance metrics on a (holdout) split."""
    model.eval()
    with torch.no_grad():
        x = obs.to(device)
        if normalizer is not None:
            x = normalizer.apply(x)
        a_hat, p_hat = model(x)
        a_hat, p_hat = a_hat.cpu(), p_hat.cpu()
    mse_alpha = (a_hat - alpha_star).pow(2).mean().item()
    base_alpha = (alpha_star.mean(dim=0, keepdim=True)
                  - alpha_star).pow(2).mean().item()
    conflict = p_star.abs() > 0.5
    if conflict.any():
        dir_acc = (torch.sign(p_hat[conflict])
                   == torch.sign(p_star[conflict])).float().mean().item()
    else:
        dir_acc = float("nan")
    mse_p = (p_hat - p_star).pow(2).mean().item()
    base_p = (p_star.mean() - p_star).pow(2).mean().item()
    sat = (p_hat.abs() > 0.99).float().mean().item()
    out = {
        "mse_alpha": mse_alpha, "mse_alpha_baseline": base_alpha,
        "alpha_gate_pass": bool(mse_alpha < 0.75 * base_alpha),
        "p_dir_acc_conflict": dir_acc,
        "p_dir_gate_pass": bool(dir_acc > 0.80) if conflict.any() else False,
        "n_conflict_holdout": int(conflict.sum().item()),
        "mse_p": mse_p, "mse_p_baseline": base_p,
        "p_sat_frac": sat,
        "alpha_pred_mean": a_hat.mean().item(),
        "alpha_pred_std": a_hat.std().item(),
    }
    out["acceptance_pass"] = bool(out["alpha_gate_pass"]
                                  and out["p_dir_gate_pass"])
    return out


def bc_pretrain_v2(dataset: dict, cfg: "BCV2Config | None" = None,
                   device: str = "cpu") -> tuple:
    """-> (model, normalizer|None, report dict with losses + acceptance)."""
    cfg = cfg or BCV2Config()
    torch.manual_seed(cfg.seed)
    obs = dataset["obs"].float()
    a_star, p_star = dataset["alpha_star"].float(), dataset["p_star"].float()
    tr, ho = _split(obs.shape[0], cfg.holdout_frac, cfg.seed)
    normalizer = ObsNormalizer.fit(obs[tr]) if cfg.normalize else None
    x_tr = normalizer.apply(obs[tr]) if normalizer else obs[tr]
    raw_tr = labels_to_raw(a_star[tr], p_star[tr])
    w_tr = 1.0 + cfg.p_conflict_weight * (p_star[tr].abs() > 0.5).float()
    model = CoordinatorBetaMLP(obs.shape[1], cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    x_tr, raw_tr, w_tr = x_tr.to(device), raw_tr.to(device), w_tr.to(device)
    n = x_tr.shape[0]
    conflict_pool = (p_star[tr].abs() > 0.5).nonzero(as_tuple=True)[0].to(device)
    quota = int(cfg.batch_size * cfg.oversample_conflict) \
        if len(conflict_pool) else 0
    g_dev = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    losses = []
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=device, generator=g_dev)
        ep, nb = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if quota and len(idx) == cfg.batch_size:
                extra = conflict_pool[torch.randint(
                    len(conflict_pool), (quota,), device=device,
                    generator=g_dev)]
                idx = torch.cat([idx[:-quota], extra])
            c1, c0 = model.concentrations(x_tr[idx])
            loss = beta_nll(c1, c0, raw_tr[idx], w_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep += loss.item()
            nb += 1
        losses.append(ep / max(nb, 1))
    report = {
        "losses": losses,
        "holdout": evaluate_bc(model, normalizer, obs[ho], a_star[ho],
                               p_star[ho], device=device),
        "n_train": int(len(tr)), "n_holdout": int(len(ho)),
        "normalized": bool(normalizer is not None),
    }
    return model, normalizer, report


# ------------------------------------------------------- compat export

def export_mlp_compat(model: CoordinatorBetaMLP,
                      normalizer: "ObsNormalizer | None",
                      obs_sample: torch.Tensor,
                      ridge: float = 1e-4,
                      p_dir_weight: float = 20.0) -> tuple:
    """Distill the Beta means into the legacy CoordinatorMLP format.

    Returns (state_dict, fit_report). The trunk copies over (same keys); the
    normalizer is folded into body.0 so the exported net eats RAW obs; the
    heads are ridge least-squares fits on trunk features:
        head_alpha: sigmoid(W z + b) ~ alpha_mean  (fit in logit space)
        head_p:     tanh(W z + b)    ~ p_mean      (fit in atanh space,
                    DIRECTED samples weighted x p_dir_weight)
    The p fit must be direction-weighted: ~97% of samples sit at p ~= 0, so an
    unweighted fit shrinks the sparse directed samples toward zero and flips
    marginal signs -- measured 08-12 on the 51k server dataset (p_dir_agree
    0.944 < 0.95 gate; the local 7.2k smoke sat at 1.0 and hid it). A sign
    flip on a conflict sample is exactly the r1-class pathology the smoke
    gate exists to catch, so the weighting targets directed fidelity and
    accepts mild nonzero drift on the p ~= 0 mass (PPO exploration absorbs
    that). A3's train/bc_init.py then applies its own tanh(x/2)~x/2
    transplant on top, unchanged.
    """
    model.eval()
    with torch.no_grad():
        x_n = normalizer.apply(obs_sample) if normalizer else obs_sample
        z = model.body(x_n)                            # (B, H)
        a_mean, p_mean = model(x_n)
        y_a = torch.logit(a_mean.clamp(1e-4, 1 - 1e-4))
        y_p = torch.atanh(p_mean.clamp(-0.999, 0.999)).unsqueeze(-1)
        zb = torch.cat([z, torch.ones(z.shape[0], 1)], dim=-1)  # (B, H+1)
        # ridge scaled to the feature magnitude + lstsq fallback: unnormalized
        # trunks (ablation path) produce huge/collinear features that make the
        # plain gram solve singular (hit 08-12 on the v2_nonorm ablation)
        gram = zb.T @ zb
        scale = gram.diagonal().mean().clamp_min(1.0)
        gram = gram + ridge * scale * torch.eye(zb.shape[1])
        w_dir = torch.ones(zb.shape[0], 1)
        w_dir[p_mean.abs() > 0.1] = p_dir_weight
        zw = zb * w_dir
        gram_p = zw.T @ zb
        scale_p = gram_p.diagonal().mean().clamp_min(1.0)
        gram_p = gram_p + ridge * scale_p * torch.eye(zb.shape[1])
        try:
            sol_a = torch.linalg.solve(gram, zb.T @ y_a)     # (H+1, 4)
            sol_p = torch.linalg.solve(gram_p, zw.T @ y_p)   # (H+1, 1)
        except torch.linalg.LinAlgError:
            sol_a = torch.linalg.lstsq(zb, y_a).solution
            ws = w_dir.sqrt()
            sol_p = torch.linalg.lstsq(zb * ws, y_p * ws).solution
        w_a, b_a = sol_a[:-1].T.contiguous(), sol_a[-1].contiguous()
        w_p, b_p = sol_p[:-1].T.contiguous(), sol_p[-1].contiguous()
        # fold normalization into the first linear layer: x_n = (x - mu)/sd
        w0 = model.body[0].weight.clone()
        b0 = model.body[0].bias.clone()
        if normalizer is not None:
            w0 = w0 / normalizer.std.unsqueeze(0)
            b0 = b0 - (w0 @ normalizer.mean)
        sd = {
            "body.0.weight": w0, "body.0.bias": b0,
            "body.2.weight": model.body[2].weight.clone(),
            "body.2.bias": model.body[2].bias.clone(),
            "head_alpha.weight": w_a, "head_alpha.bias": b_a,
            "head_p.weight": w_p, "head_p.bias": b_p,
        }
        # distillation quality on the fit sample (raw-obs path end to end)
        a_c = torch.sigmoid(zb[:, :-1] @ w_a.T + b_a)
        p_c = torch.tanh((zb[:, :-1] @ w_p.T + b_p).squeeze(-1))
        # direction agreement is only meaningful where the BC policy actually
        # has a direction; near-zero p_mean is sign noise by construction
        directed = p_mean.abs() > 0.1
        agree = (torch.sign(p_c[directed]) == torch.sign(p_mean[directed])) \
            .float().mean().item() if directed.any() else 1.0
        fit = {
            "alpha_distill_mae": (a_c - a_mean).abs().mean().item(),
            "alpha_distill_max": (a_c - a_mean).abs().max().item(),
            "p_distill_mae": (p_c - p_mean).abs().mean().item(),
            "p_dir_agree": agree,
            "n_directed": int(directed.sum().item()),
        }
    return sd, fit


def save_checkpoint_v2(path, model: CoordinatorBetaMLP,
                       normalizer: "ObsNormalizer | None",
                       obs_sample: torch.Tensor, report: dict,
                       obs_layout: str = "duo_env") -> dict:
    """v2 checkpoint: "state_dict" = the distilled CoordinatorMLP view (so
    train/bc_init.py loads it UNCHANGED); the Beta original rides along for
    the future Beta-actor PPO."""
    compat_sd, fit = export_mlp_compat(model, normalizer, obs_sample)
    ck = {
        "format": "bc_v2",
        "state_dict": compat_sd,                  # bc_init.py contract
        "beta_state_dict": model.state_dict(),
        "normalizer": normalizer.state_dict() if normalizer else None,
        "normalizer_folded_into_compat": normalizer is not None,
        "obs_dim": model.obs_dim, "hidden": model.hidden,
        "obs_layout": obs_layout,
        "acceptance": report.get("holdout", {}),
        "distill_fit": fit,
    }
    torch.save(ck, path)
    return ck
