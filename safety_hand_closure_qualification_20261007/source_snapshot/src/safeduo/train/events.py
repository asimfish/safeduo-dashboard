"""事件 parquet 落盘（ACCEPTANCE §1 step_record 子集，W2 版）。

前 K 个 env 全事件流 + 全体聚合计数；每 flush_every 步写一个分片文件。
依赖 pyarrow（服务器 safeduo env 安装）。
"""

from __future__ import annotations

from pathlib import Path

import torch


class EventWriter:
    def __init__(self, out_dir: str, n_log_envs: int = 64, flush_every: int = 512):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.k = n_log_envs
        self.flush_every = flush_every
        self.buf: list = []
        self.shard = 0
        self.step = 0

    def add(self, t: int, fields: dict) -> None:
        """fields: name -> (N,) 或 (N,C) tensor；只留前 K env，搬 CPU。"""
        rec = {"t": t}
        for name, v in fields.items():
            v = v[: self.k].detach().float().cpu()
            rec[name] = v
        self.buf.append(rec)
        self.step += 1
        if self.step % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        cols: dict = {"t": [], "env": []}
        first = self.buf[0]
        names = [k for k in first if k != "t"]
        for n in names:
            width = first[n].shape[-1] if first[n].dim() > 1 else 1
            for c in range(width):
                cols[f"{n}_{c}" if width > 1 else n] = []
        for rec in self.buf:
            k = rec[names[0]].shape[0]
            cols["t"].extend([rec["t"]] * k)
            cols["env"].extend(range(k))
            for n in names:
                v = rec[n]
                if v.dim() > 1:
                    for c in range(v.shape[-1]):
                        cols[f"{n}_{c}"].extend(v[:, c].tolist())
                else:
                    cols[n].extend(v.tolist())
        pq.write_table(pa.table(cols), self.dir / f"events_{self.shard:05d}.parquet")
        self.buf.clear()
        self.shard += 1
