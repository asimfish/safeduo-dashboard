"""Aggregate pair-stratified episodes without hiding unexposed attempts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from safeduo.eval.coverage_design import DIRECTIONS, DISTANCE_BANDS, PAIR_NAMES


def _cp95_upper(k: int, n: int) -> float | None:
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    # Same one-sided Clopper-Pearson calculation used by random_battery, kept
    # local so this analysis remains usable without importing IsaacLab.
    import math

    alpha = 0.05
    def cdf(p: float) -> float:
        terms = []
        for i in range(k + 1):
            log_c = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            terms.append(log_c + i * math.log(p) + (n - i) * math.log1p(-p))
        peak = max(terms)
        return math.exp(peak) * sum(math.exp(x - peak) for x in terms)
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) * 0.5
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def aggregate(episodes: list[dict], near_m: float = 0.005) -> dict:
    """Return attempt/exposure/near/violation counts by experimental cell."""
    rows = []
    by_cell: dict[tuple, dict] = defaultdict(lambda: {
        "attempts": 0, "exposed": 0, "near": 0, "violations": 0,
    })
    for row in episodes:
        pair = row.get("pair")
        if pair not in PAIR_NAMES:
            continue
        pair_index = PAIR_NAMES.index(pair)
        margins = row.get("min_pair_margin_m") or []
        warns = row.get("pair_warn_steps") or []
        observed_margin = float(margins[pair_index]) if len(margins) > pair_index else None
        exposed = bool(len(warns) > pair_index and int(warns[pair_index]) > 0)
        near = bool(observed_margin is not None and observed_margin < near_m)
        key = (str(row.get("method")), pair, str(row.get("distance_band")),
               str(row.get("direction")), float(row.get("amp")))
        cell = by_cell[key]
        cell["attempts"] += 1
        cell["exposed"] += int(exposed)
        cell["near"] += int(near)
        cell["violations"] += int(bool(row.get("violation")))
    for key, counts in sorted(by_cell.items()):
        method, pair, band, direction, amp = key
        n = counts["attempts"]
        e = counts["exposed"]
        v = counts["violations"]
        rows.append({
            "method": method, "pair": pair, "distance_band": band,
            "direction": direction, "amp": amp, **counts,
            "exposure_rate": e / n if n else None,
            "near_rate": counts["near"] / n if n else None,
            "violation_rate": v / n if n else None,
            "violation_cp95_upper": _cp95_upper(v, n),
            "violation_cp95_upper_exposed": _cp95_upper(v, e) if e else None,
        })
    methods = sorted({row["method"] for row in rows})
    by_method = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        n = sum(row["attempts"] for row in subset)
        e = sum(row["exposed"] for row in subset)
        v = sum(row["violations"] for row in subset)
        by_method[method] = {
            "attempts": n, "exposed": e,
            "near": sum(row["near"] for row in subset), "violations": v,
            "exposure_rate": e / n if n else None,
            "violation_rate": v / n if n else None,
            "violation_cp95_upper": _cp95_upper(v, n),
            "violation_cp95_upper_exposed": _cp95_upper(v, e) if e else None,
            "pair_exposed": {pair: sum(row["exposed"] for row in subset if row["pair"] == pair)
                             for pair in PAIR_NAMES},
        }
    expected = {(method, pair, band, direction, amp)
                for method in methods
                for pair in PAIR_NAMES
                for band, _ in DISTANCE_BANDS
                for direction in DIRECTIONS
                for amp in sorted({key[-1] for key in by_cell})}
    missing = [
        {"method": method, "pair": pair, "distance_band": band,
         "direction": direction, "amp": amp}
        for method, pair, band, direction, amp in sorted(expected)
        if (method, pair, band, direction, amp) not in by_cell
    ]
    return {"schema": "safeduo.coverage_matrix.v1", "near_threshold_m": near_m,
            "episodes": len(episodes), "by_method": by_method,
            "rows": rows, "missing_cells": missing,
            "complete": not missing}


def write_outputs(result: dict, out: str | Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage_matrix.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["method", "pair", "distance_band", "direction", "amp", "attempts",
              "exposed", "near", "violations", "exposure_rate", "near_rate",
              "violation_rate", "violation_cp95_upper", "violation_cp95_upper_exposed"]
    with (out / "coverage_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in result["rows"])
    lines = ["# Pair-stratified coverage matrix", "", f"- Episodes: {result['episodes']}",
             f"- Complete pair × band × direction × amplitude design: **{result['complete']}**", "",
             "| method | pair | band | direction | amp | attempts | exposed | near | violations |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append("| {method} | {pair} | {distance_band} | {direction} | {amp:g} | {attempts} | {exposed} | {near} | {violations} |".format(**row))
    lines += ["", "## Method totals", "", "| method | attempts | exposed | near | violations | CP95 upper |", "|---|---:|---:|---:|---:|---:|"]
    for method, row in result["by_method"].items():
        upper = "n/a" if row["violation_cp95_upper"] is None else f"{row['violation_cp95_upper']:.4f}"
        lines.append(f"| {method} | {row['attempts']} | {row['exposed']} | {row['near']} | {row['violations']} | {upper} |")
    if result["missing_cells"]:
        lines += ["", "## Missing cells", ""] + [f"- `{cell}`" for cell in result["missing_cells"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--near-mm", type=float, default=5.0)
    args = parser.parse_args()
    episodes = json.loads(Path(args.episodes).read_text(encoding="utf-8"))
    result = aggregate(episodes, near_m=args.near_mm * 1e-3)
    write_outputs(result, args.out)
    print(json.dumps(result["by_method"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
