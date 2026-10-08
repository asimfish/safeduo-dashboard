"""Warm-start early-PPO protection (W6-2b): critic warmup under a frozen actor.

A4's second BC-init failure autopsy (STATUS_A W4 sec.2): the BC net passed
every holdout gate yet full-throttled the live env from step one; with a
RANDOM critic, the first 1-2 PPO iterations then applied violent updates on
garbage advantages and threw the policy into the freeze attractor
(alpha -> 0, p -> +1, subsequently reinforced -- measured shard3/6
alpha == 0.000, p_pinned == 1.0). The live-behavior gate
(bc_smoke_check --live-rollout) now catches the bad checkpoint BEFORE launch;
this module protects the launch itself.

Work-order candidates, and why critic-warmup-frozen-actor is the one
implemented (lowest cost + cleanly ablatable):

  lr warmup            fights schedule="adaptive": rsl_rl recomputes the lr
                       from the measured KL EVERY learning epoch and writes
                       it into the optimizer param groups, clobbering any
                       external multiplier mid-update. Making a warmup stick
                       would mean patching inside the epoch loop.
  KL early stop        rsl_rl's update loop has no per-epoch hook; a real
                       trust-region stop means forking OnPolicyRunner.
  critic warmup        freeze the policy branch (actor MLP + noise std) via
  (frozen actor)       requires_grad for the first N updates: the critic
                       learns V under the FIXED BC policy + exploration
                       noise, so the first policy gradient ever applied is
                       computed from informed advantages. requires_grad is
                       honored wherever the update lives (clip_grad_norm_
                       and Adam both skip grad-less params) -- zero rsl_rl
                       internals touched. n_iters=0 is a bit-exact no-op,
                       which IS the ablation arm.

The one real interaction: while the actor is frozen the measured KL is ~0,
so the adaptive schedule multiplies the lr x1.5 per epoch up to its cap --
left alone, the unfrozen actor's FIRST update would fire at the cap (the
exact violence this module exists to prevent). The attach-time lr is
therefore restored at release (pinned in tests).

Wiring (A-line, one call after runner construction in train_ppo.py; alg is
runner.alg, whose policy attr is `policy` on current rsl_rl and
`actor_critic` on older versions -- both handled):

    if args.bc_init and args.bc_protect_iters > 0:
        from safeduo.algo.warmstart_protection import attach_warmstart_protection
        meta = attach_warmstart_protection(runner.alg,
                                           n_iters=args.bc_protect_iters)
        print("BC_PROTECT " + json.dumps(meta), flush=True)

Recommended n_iters=10: at 24 steps/env x 4096 envs that is ~1M env-steps of
pure value learning before the first policy gradient (the measured failure
window was 1-2 iters; 10 gives 5x margin at ~2-3 min wall time), registered
as c4_bc_protect_ctrl in experiments.csv.
"""

from __future__ import annotations

import torch


def policy_of(alg):
    """rsl_rl PPO policy attr across versions (`policy` new, `actor_critic`
    old) -- same fallback train_ppo.py itself uses for bc_init."""
    pol = getattr(alg, "policy", None)
    if pol is None:
        pol = alg.actor_critic
    return pol


def policy_branch_params(policy) -> list:
    """(name, param) of everything that is NOT the critic: actor MLP plus the
    exploration noise parameter (std/log_std -- freezing it keeps exploration
    at the init scale instead of letting entropy terms shrink it while the
    actor cannot compensate)."""
    return [(n, p) for n, p in policy.named_parameters()
            if not n.startswith("critic")]


def attach_warmstart_protection(alg, n_iters: int,
                                restore_lr: "float | None" = None) -> dict:
    """Freeze the policy branch for the first `n_iters` calls of alg.update().

    Wraps alg.update in place. Release (at the START of update n_iters, i.e.
    updates 0..n_iters-1 run frozen): re-enable grads and restore the
    attach-time lr on both alg.learning_rate and every optimizer param group
    (the adaptive-KL schedule inflates lr while KL ~= 0, see module
    docstring). n_iters <= 0 attaches nothing -- the ablation arm.
    Returns a meta dict for the launch log."""
    if n_iters <= 0:
        return {"enabled": False, "n_iters": int(n_iters)}
    pol = policy_of(alg)
    frozen = policy_branch_params(pol)
    if not frozen:
        raise ValueError("policy has no non-critic parameters to freeze")
    for _, p in frozen:
        p.requires_grad_(False)
    if restore_lr is None:
        restore_lr = float(alg.optimizer.param_groups[0]["lr"])
    state = {"updates": 0, "released": False}
    orig_update = alg.update

    def protected_update(*args, **kwargs):
        if not state["released"] and state["updates"] >= n_iters:
            for _, p in frozen:
                p.requires_grad_(True)
            if hasattr(alg, "learning_rate"):
                alg.learning_rate = restore_lr
            for g in alg.optimizer.param_groups:
                g["lr"] = restore_lr
            state["released"] = True
        out = orig_update(*args, **kwargs)
        state["updates"] += 1
        return out

    alg.update = protected_update
    return {
        "enabled": True,
        "n_iters": int(n_iters),
        "restore_lr": restore_lr,
        "n_frozen_params": len(frozen),
        "frozen_prefixes": sorted({n.split(".")[0] for n, _ in frozen}),
        "_state": state,          # exposed for tests/monitoring
    }


@torch.no_grad()
def snapshot_params(policy) -> dict:
    """name -> cloned tensor, for freeze verification in tests/monitors."""
    return {n: p.detach().clone() for n, p in policy.named_parameters()}
