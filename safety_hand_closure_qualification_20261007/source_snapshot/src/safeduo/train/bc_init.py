"""BC warm-start checkpoint -> rsl_rl actor MLP weight transplant (pure torch).

C2's bc_warmstart_run.py saves a CoordinatorMLP (algo/warmstart_oracle):
  body:  Linear(obs, H) ELU Linear(H, H) ELU
  heads: alpha = sigmoid(Linear(H, 4)),  p = tanh(Linear(H, 1))

The rsl_rl actor (actor_hidden_dims=[H, H]) is
  Linear(obs, H) ELU Linear(H, H) ELU Linear(H, 5)
whose raw output a is mapped by duo_env to alpha = (clamp(a,-1,1)+1)/2 and
p = clamp(a4,-1,1).

Trunk transfers exactly (same shapes). The final layer cannot match sigmoid
with an affine map; we use the first-order transplant

  alpha: want a = 2*sigmoid(x) - 1 = tanh(x/2)  ~=  x/2   (exact at x=0,
         env clamp saturates at |x|>=2 where tanh saturates smoothly)
  p:     want a = tanh(y)                        ~=  y

i.e. final.weight[:4] = head_alpha/2, final.weight[4] = head_p. Sign and
ordering of the BC policy are preserved everywhere, magnitudes only distort
mid-band (<=24% at the clamp knee); PPO fine-tunes from there. Keep the run's
init_noise_std as configured -- the transplant only sets the mean network.
"""

from __future__ import annotations

import torch


def load_bc_actor_init(actor: torch.nn.Module, ckpt_path: str,
                       p_neutral: bool = False) -> dict:
    """Copy a CoordinatorMLP checkpoint into an rsl_rl actor Sequential.

    actor must contain exactly 3 Linear layers shaped
    (H, obs), (H, H), (5, H) with H == ckpt hidden. Returns meta for logging.

    p_neutral: ZERO the p row (weights and bias). Use when the BC p head is
    degenerate -- observed 2026-08-12: tanh+MSE passive saturation collapsed
    head_p to constant +1 on the server dataset (97% zero labels give no
    restoring gradient once the shared trunk drifts); the alpha head remained
    genuinely predictive (mse 0.154 < 0.220 mean baseline), so trunk+alpha
    transplant keeps the value without baking a stuck priority.

    Zeroing (not "keep the actor's own init") is required because the BC
    trunk's feature magnitudes are large: a random p row projects them into
    a nearly constant large pre-clamp value -> p pinned at +-1 (measured on
    the aborted a5_v2 r1: p == +1.000 for every logged step). Zero row gives
    raw a4 == 0 -> env p == 0 truly neutral; PPO noise breaks symmetry.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    hidden, obs_dim = int(ck["hidden"]), int(ck["obs_dim"])
    linears = [m for m in actor.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3, (
        f"actor has {len(linears)} Linear layers, transplant needs trunk[H,H]+head "
        f"(actor_hidden_dims=[{hidden},{hidden}])")
    l0, l1, lf = linears
    assert tuple(l0.weight.shape) == (hidden, obs_dim), (
        f"actor first layer {tuple(l0.weight.shape)} != BC ({hidden},{obs_dim})"
        " -- obs layout drifted?")
    assert tuple(l1.weight.shape) == (hidden, hidden)
    assert tuple(lf.weight.shape) == (5, hidden), (
        f"actor final layer {tuple(lf.weight.shape)} != (5,{hidden})")
    with torch.no_grad():
        l0.weight.copy_(sd["body.0.weight"])
        l0.bias.copy_(sd["body.0.bias"])
        l1.weight.copy_(sd["body.2.weight"])
        l1.bias.copy_(sd["body.2.bias"])
        # alpha head: a = 2*sigmoid(x)-1 = tanh(x/2) ~= x/2
        lf.weight[:4].copy_(sd["head_alpha.weight"] * 0.5)
        lf.bias[:4].copy_(sd["head_alpha.bias"] * 0.5)
        if p_neutral:
            lf.weight[4:5].zero_()
            lf.bias[4:5].zero_()
        else:
            # p head: a = tanh(y) ~= y
            lf.weight[4:5].copy_(sd["head_p.weight"])
            lf.bias[4:5].copy_(sd["head_p.bias"])
    return {"obs_dim": obs_dim, "hidden": hidden, "p_neutral": p_neutral,
            "obs_layout": ck.get("obs_layout", "?"), "ckpt": str(ckpt_path)}
