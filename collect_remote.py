#!/usr/bin/env python3
"""Snapshot the SafeDuo GPU server for the dashboard. Runs ON the server (stdlib only), prints JSON.

Collected: per-GPU memory/util + compute processes (tagged safeduo / chembench / other, with a
human label parsed from the command line), tmux sessions, training runs (latest iter + key stats
from stats.jsonl), and four-gate evaluation directories (cells finished / in progress).
"""
import glob
import json
import os
import re
import subprocess
import time

HOME = os.path.expanduser("~")
ROOT = os.environ.get("SAFEDUO_ARTIFACTS", os.path.join(HOME, "safeduo", "artifacts"))
# colon-separated list: runs/ and clutch/ are scanned under every root (A100: code checkout + $HOME)
ROOTS = [r for r in ROOT.split(":") if r]
NODE = os.environ.get("SAFEDUO_NODE", "bjxy_5090")
GATE_SUBDIR = "clutch"
ACTIVE_S = 30 * 60  # a run/gate touched within 30 min counts as active


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except Exception:
        return ""


def label_proc(cmd):
    """Return (tag, label) for a GPU process command line."""
    if not cmd:
        return "other", "?"
    if "safeduo" in cmd:
        tag = "safeduo"
        if "train_lagrangian" in cmd or "train_sac" in cmd:
            m = re.search(r"--run_name\s+(\S+)", cmd)
            return tag, "train " + (m.group(1) if m else "?")
        if "clutch_eval" in cmd:
            m = re.search(r"--out\s+\S*/([^/\s]+)\s*$", cmd) or re.search(r"--out\s+\S*/([^/\s]+)", cmd)
            return tag, "gate " + (m.group(1) if m else "?")
        if "task_record" in cmd or "skill_record" in cmd or "record" in cmd:
            return tag, "record " + (re.search(r"--task\s+(\S+)", cmd).group(1) if re.search(r"--task\s+(\S+)", cmd) else "")
        m = re.search(r"-m\s+(safeduo\.[\w.]+)", cmd) or re.search(r"(\w+\.py)", cmd)
        return tag, m.group(1) if m else "safeduo"
    if "chembench" in cmd or "psilab" in cmd:
        return "chembench", "chembench"
    m = re.search(r"([\w\-]+\.py)", cmd)
    return "other", m.group(1) if m else cmd.split()[0].rsplit("/", 1)[-1]


def gpus():
    out = []
    q = sh("nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
    by_uuid = {}
    for line in q.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        g = {"idx": int(parts[0]), "name": parts[2].replace("NVIDIA GeForce ", ""), "used_mb": int(float(parts[3])),
             "total_mb": int(float(parts[4])), "util": int(float(parts[5])), "procs": []}
        by_uuid[parts[1]] = g
        out.append(g)
    apps = sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits")
    for line in apps.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or parts[0] not in by_uuid:
            continue
        pid = int(parts[1])
        tag, label = label_proc(cmdline(pid))
        by_uuid[parts[0]]["procs"].append({"pid": pid, "mem_mb": int(float(parts[2])), "tag": tag, "label": label})
    for g in out:
        g["procs"].sort(key=lambda p: -p["mem_mb"])
        g["mine"] = any(p["tag"] == "safeduo" for p in g["procs"])
    return out


def tmux():
    return [l.split(":")[0] for l in sh("tmux ls 2>/dev/null").splitlines() if ":" in l]


def read_last_json_line(path, max_bytes=20000):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - max_bytes))
            chunk = fh.read().decode("utf-8", "replace")
        for line in reversed(chunk.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def runs(max_runs=12):
    now = time.time()
    rows = []
    for d in [x for r in ROOTS for x in glob.glob(os.path.join(r, "runs", "*"))]:
        st = os.path.join(d, "stats.jsonl")
        if not os.path.isfile(st):
            continue
        mtime = os.path.getmtime(st)
        last = read_last_json_line(st) or {}
        ck = sorted(glob.glob(os.path.join(d, "model_*.pt")), key=os.path.getmtime)
        rows.append({
            "name": os.path.basename(d), "mtime": int(mtime), "active": now - mtime < ACTIVE_S,
            "iter": last.get("iter"), "alpha_exec": last.get("alpha_exec_mean"), "alpha": last.get("alpha_mean"),
            "costs": last.get("cost_means"), "lambdas": last.get("multipliers"), "hazard_rate": last.get("hazard_rate"),
            "entropy": last.get("entropy_ema"), "wall_s": last.get("wall_s"),
            "n_ckpt": len(ck), "has_last": os.path.isfile(os.path.join(d, "model_last.pt")),
        })
    rows.sort(key=lambda r: -r["mtime"])
    return rows[:max_runs]


def gates(max_dirs=16):
    now = time.time()
    rows = []
    for d in [x for r in ROOTS for x in glob.glob(os.path.join(r, "clutch", "*"))]:
        if not os.path.isdir(d):
            continue
        cells = sorted(glob.glob(os.path.join(d, "cell_*.json")))
        files = glob.glob(os.path.join(d, "*"))
        mtime = max([os.path.getmtime(f) for f in files] + [os.path.getmtime(d)])
        rows.append({"dir": os.path.basename(d), "n_cells": len(cells), "cells": [os.path.basename(c)[5:-5] for c in cells],
                     "mtime": int(mtime), "active": now - mtime < ACTIVE_S and not cells or now - mtime < 5 * 60})
    rows.sort(key=lambda r: -r["mtime"])
    return rows[:max_dirs]


def main():
    up = sh("uptime").strip()
    m = re.search(r"load average[s]?:\s*([\d.]+)", up)
    snap = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "node": NODE,
        "host": sh("hostname").strip(),
        "load": m.group(1) if m else None,
        "ncpu": os.cpu_count(),
        "uptime": up,
        "gpus": gpus(),
        "tmux": tmux(),
        "runs": runs(),
        "gates": gates(),
    }
    print(json.dumps(snap, ensure_ascii=False))


if __name__ == "__main__":
    main()
