"""Persist evidence when an experimental release observer/selector aborts.

The context must end before consuming release events. Healthy calls do not write
files or change the selector. This is an evidence recorder, not an actuator stop.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from contextlib import contextmanager
from pathlib import Path


def _json_state(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite": "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"}
    if isinstance(value, dict):
        return {key: _json_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_state(item) for item in value]
    return value


def _atomic_write(path: Path, text: str):
    temporary = path.with_name(path.name + ".abort.tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def release_evidence_guard(*, out_dir, stage, step, event_clock_s,
                           raw_state, dense_rows):
    """Retain the valid prefix and failing raw observation, then re-raise.

    If evidence storage fails, that error remains visible with the original
    failure as its exception context. Non-finite raw values get explicit JSON
    markers; they are never silently replaced by zeros or valid measurements.
    """
    try:
        yield
    except Exception as error:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        prefix = "".join(json.dumps(row, allow_nan=False) + "\n" for row in dense_rows)
        evidence = {
            "schema_version": 1,
            "status": "INVALID",
            "stage": stage,
            "step": step,
            "event_clock_s": event_clock_s,
            "raw_state": _json_state(raw_state),
            "dense_prefix_rows": len(dense_rows),
            "last_dense_step": dense_rows[-1]["step"] if dense_rows else None,
            "sampling": "completed valid post-step observations; failing step in raw_state",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_write(root / "release_dense.jsonl", prefix)
        _atomic_write(root / "abort_evidence.json", json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        raise
