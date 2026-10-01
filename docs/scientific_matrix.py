"""Aggregate random-battery results into a reproducible research matrix.

This module deliberately uses only the Python standard library.  It treats each
``random_battery`` summary as an immutable experiment record, validates the
metadata before pooling results, and writes machine-readable CSV/JSON plus a
short Markdown report.  It is an analysis tool; it never reruns a simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PAIR_NAMES = ("F_L-F_R", "F_L-U_L", "F_L-U_R", "F_R-U_L", "F_R-U_R", "U_L-U_R")


def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> float | None:
    """Return the two-sided Wilson 95% upper bound for a binomial rate."""
    if n <= 0:
        return None
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return min(1.0, centre + radius)


def _numbers(value: Any) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(float(x) for x in value)
    return (float(value),)


def _seed_tuple(args: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in args.get("seeds", ()))


def _window_rows(summary: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    args = summary.get("args", {})
    rows: list[dict[str, Any]] = []
    for window in summary.get("windows", []):
        row = {
            "source": str(source),
            "seed": window.get("seed"),
            "flow": window.get("flow"),
            "amp": window.get("amp"),
            "episodes": int(window.get("episodes", 0)),
            "violations": int(window.get("violation_episodes", 0)),
            "violation_rate": window.get("violation_episode_rate"),
            "cp95_upper": window.get("violation_cp95_upper"),
        }
        for key in ("ckpt", "env_yaml", "num_envs", "duration_s", "theta", "k_expose"):
            row[key] = args.get(key)
        rows.append(row)
    return rows


def load_summary(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict) or not isinstance(summary.get("args"), dict):
        raise ValueError(f"invalid random_battery summary: {source}")
    return summary, _window_rows(summary, source)


def _metadata(summary: dict[str, Any]) -> tuple[Any, ...]:
    args = summary["args"]
    return (
        str(args.get("ckpt", "")),
        str(args.get("env_yaml", "")),
        int(args.get("num_envs", 0)),
        float(args.get("duration_s", 0.0)),
        str(args.get("theta", "")),
        int(args.get("k_expose", 0)),
    )


def aggregate(paths: Iterable[str | Path], *, allow_mixed: bool = False) -> dict[str, Any]:
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("at least one summary is required")
    summaries: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    for path in paths:
        summary, rows = load_summary(path)
        summaries.append((Path(path).resolve(), summary, rows))
    metadata = {_metadata(summary) for _, summary, _ in summaries}
    warnings: list[str] = []
    if len(metadata) > 1:
        message = "input summaries have incompatible simulator metadata; pooled rates are not comparable"
        if not allow_mixed:
            raise ValueError(message + " (use --allow-mixed only for an explicitly stratified report)")
        warnings.append(message)

    rows = [row for _, _, local_rows in summaries for row in local_rows]
    episodes = sum(int(row["episodes"]) for row in rows)
    violations = sum(int(row["violations"]) for row in rows)
    by_flow: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "violations": 0})
    by_seed: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "violations": 0})
    by_amp: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "violations": 0})
    for row in rows:
        for bucket, key in ((by_flow, str(row["flow"])), (by_seed, str(row["seed"])), (by_amp, str(row["amp"]))):
            bucket[key]["episodes"] += int(row["episodes"])
            bucket[key]["violations"] += int(row["violations"])
    pair_exposure: dict[str, int] = defaultdict(int)
    for _, summary, _ in summaries:
        for pair, count in (summary.get("aggregate", {}).get("pair_exposure", {}).get("episodes_exposed_by_pair", {}) or {}).items():
            pair_exposure[pair] += int(count)
    for pair in PAIR_NAMES:
        pair_exposure.setdefault(pair, 0)

    def rates(bucket: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
        result = {}
        for key, values in sorted(bucket.items()):
            n, k = values["episodes"], values["violations"]
            result[key] = {
                "episodes": n,
                "violations": k,
                "rate": (k / n) if n else None,
                "cp95_upper": wilson_upper(k, n),
            }
        return result

    return {
        "schema": "safeduo.scientific_matrix.v1",
        "inputs": [str(path) for path, _, _ in summaries],
        "metadata": [list(item) for item in sorted(metadata)],
        "warnings": warnings,
        "episodes": episodes,
        "violations": violations,
        "violation_rate": (violations / episodes) if episodes else None,
        "cp95_upper": wilson_upper(violations, episodes),
        "by_flow": rates(by_flow),
        "by_seed": rates(by_seed),
        "by_amp": rates(by_amp),
        "pair_exposure_episodes": dict(sorted(pair_exposure.items())),
        "rows": rows,
    }


def _missing_cells(result: dict[str, Any], seeds: set[int], flows: set[str], amps: set[float]) -> list[dict[str, Any]]:
    observed = {
        (int(row["seed"]), str(row["flow"]), float(row["amp"]))
        for row in result["rows"]
        if row["seed"] is not None and row["flow"] is not None and row["amp"] is not None
    }
    missing = []
    for seed in sorted(seeds):
        for flow in sorted(flows):
            for amp in sorted(amps):
                if (seed, flow, amp) not in observed:
                    missing.append({"seed": seed, "flow": flow, "amp": amp})
    return missing


def write_outputs(result: dict[str, Any], out: str | Path, *, expected_seeds: set[int] | None = None,
                  expected_flows: set[str] | None = None, expected_amps: set[float] | None = None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    missing = _missing_cells(result, expected_seeds or set(), expected_flows or set(), expected_amps or set())
    result["design"] = {
        "expected_seeds": sorted(expected_seeds or set()),
        "expected_flows": sorted(expected_flows or set()),
        "expected_amps": sorted(expected_amps or set()),
        "missing_cells": missing,
        "complete": not missing,
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["source", "seed", "flow", "amp", "episodes", "violations", "violation_rate", "cp95_upper", "ckpt", "env_yaml", "num_envs", "duration_s", "theta", "k_expose"]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in result["rows"])
    lines = [
        "# Scientific evaluation matrix",
        "",
        f"- Inputs: {len(result['inputs'])} summary files",
        f"- Episodes: {result['episodes']}; violations: {result['violations']}",
        f"- Episode violation rate: {result['violation_rate']:.4f}" if result["violation_rate"] is not None else "- Episode violation rate: n/a",
        f"- Wilson 95% upper bound: {result['cp95_upper']:.4f}" if result["cp95_upper"] is not None else "- Wilson 95% upper bound: n/a",
        "",
        "## Stratified results",
        "",
        "| factor | cell | episodes | violations | rate | Wilson 95% upper |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for factor, values in (("flow", result["by_flow"]), ("seed", result["by_seed"]), ("amp", result["by_amp"])):
        for cell, value in values.items():
            lines.append(f"| {factor} | {cell} | {value['episodes']} | {value['violations']} | {value['rate']:.4f} | {value['cp95_upper']:.4f} |" if value["rate"] is not None else f"| {factor} | {cell} | 0 | 0 | n/a | n/a |")
    lines += ["", "## Six arm-pair exposure", "", "| pair | exposed episodes |", "|---|---:|"]
    for pair in PAIR_NAMES:
        lines.append(f"| {pair} | {result['pair_exposure_episodes'].get(pair, 0)} |")
    if result["design"]["expected_seeds"]:
        lines += ["", "## Design completeness", "", f"- Complete: **{result['design']['complete']}**"]
        if missing:
            lines.append("- Missing cells:")
            lines.extend(f"  - seed={cell['seed']}, flow={cell['flow']}, amp={cell['amp']}" for cell in missing)
    if result["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in result["warnings"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="random_battery summary.json files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--expected-flows", nargs="*", default=[])
    parser.add_argument("--expected-amps", nargs="*", type=float, default=[])
    parser.add_argument("--allow-mixed", action="store_true")
    args = parser.parse_args()
    result = aggregate(args.inputs, allow_mixed=args.allow_mixed)
    write_outputs(result, args.out, expected_seeds=set(args.expected_seeds), expected_flows=set(args.expected_flows), expected_amps=set(args.expected_amps))
    print(json.dumps({key: result[key] for key in ("episodes", "violations", "violation_rate", "cp95_upper", "warnings")}, indent=2))


if __name__ == "__main__":
    main()
