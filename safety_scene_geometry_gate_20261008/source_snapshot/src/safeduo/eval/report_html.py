"""V3 offline replay report (minimal W2 version): margin/alpha/p curves with
event markers for the worst episodes + the metrics table, in one static HTML.

The full V3 (viewport overlay mp4 links, per-scenario drill-down) lands with
Isaac assets; this skeleton already covers the review-critical part: a human
can open one file and see WHERE interventions/oscillations/stalls happened.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from safeduo.delta._contract_stub import ARM_KEYS
from safeduo.eval.metrics import EpisodeBatch


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _episode_figure(batch: EpisodeBatch, records, env: int, title: str) -> str:
    t = torch.arange(batch.T) * batch.dt
    df = records[records["env_id"] == env]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    ax = axes[0]
    for key, color in (("min_margin_cross", "tab:red"),
                       ("min_margin_self", "tab:orange"),
                       ("min_margin_table", "tab:brown")):
        ax.plot(df["t"], df[key], color=color, lw=1.1, label=key[11:])
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(0.05, color="gray", ls="--", lw=0.8, label="d_soft")
    tube = batch.in_tube[:, env]
    ax.fill_between(t, *ax.get_ylim(), where=tube.numpy(), alpha=0.12,
                    color="red", label="conflict tube")
    ax.set_ylabel("margin (m)")
    ax.legend(fontsize=7, ncol=5)
    ax.set_title(title)
    ax = axes[1]
    for i, a in enumerate(ARM_KEYS):
        ax.plot(t, batch.alpha[:, env, i], lw=1.0, label=f"alpha {a}")
    for tt in df.loc[df["backstop_trigger"], "t"]:
        ax.axvline(tt, color="purple", alpha=0.3, lw=0.8)
    ax.set_ylabel("alpha")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=4)
    ax = axes[2]
    ax.plot(t, batch.priority_p[:, env], color="tab:green", lw=1.2)
    for tt in df.loc[df["priority_flip"], "t"]:
        ax.axvline(tt, color="red", alpha=0.5, lw=0.8)
    ax.set_ylabel("priority p")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("t (s)")
    return _fig_to_b64(fig)


def render_report(batch: EpisodeBatch, records, metrics: dict,
                  out_html: "Path | str", title: str = "SafeDuo eval",
                  top_k_episodes: int = 3) -> Path:
    dwell = batch.in_tube.float().sum(dim=0)
    worst = dwell.argsort(descending=True)[:min(top_k_episodes, batch.N)]
    figs = [
        _episode_figure(batch, records, int(e),
                        f"episode {int(e)} (tube dwell {dwell[e] * batch.dt:.2f}s)")
        for e in worst
    ]
    metric_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())
    img_html = "\n".join(f'<img src="data:image/png;base64,{b}"/><hr/>'
                         for b in figs)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>body{{font-family:sans-serif;margin:24px;max-width:1000px}}
table{{border-collapse:collapse}}td{{border:1px solid #ccc;padding:3px 8px;
font-size:13px}}img{{max-width:100%}}</style></head><body>
<h1>{title}</h1>
<h2>Metrics</h2><table>{metric_rows}</table>
<h2>Worst episodes (by conflict-tube dwell)</h2>
{img_html}
</body></html>"""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
