#!/bin/bash
# Collect SafeDuo dashboard snapshots (bjxy_5090 GPU/runs/gates) and push to the `data` branch;
# mirror MASTER_REPORT.html into the Pages branch when it changed.
# Usage: update_dashboard.sh [--no-server]
set -u
DATA=~/Code/safeduo-dashboard-data
TOOLS=~/Code/safeduo-dashboard
SRC=/Users/liyufeng/Desktop/research/safeduo
REMOTE=bjxy_5090
A100=tianyiyun-30109
LOG=$TOOLS/update.log
ts() { date "+%m-%d %H:%M:%S"; }
cd "$DATA" || exit 1
LOCK="$DATA/.update.lock.d"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +20 2>/dev/null)" ]; then rmdir "$LOCK" 2>/dev/null; mkdir "$LOCK" 2>/dev/null || exit 0; else echo "$(ts) another update running, skip" >> "$LOG"; exit 0; fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
git pull -q --rebase origin data >/dev/null 2>&1 || git rebase --abort >/dev/null 2>&1
if [ "${1:-}" != "--no-server" ]; then
  if timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=20 $REMOTE 'python3 -' < "$TOOLS/collect_remote.py" > runs_5090.json.tmp 2>/dev/null && [ -s runs_5090.json.tmp ] && python3 -c "import json;json.load(open('runs_5090.json.tmp'))" 2>/dev/null; then
    mv runs_5090.json.tmp runs_5090.json; echo "$(ts) 5090 snapshot ok" >> "$LOG"
  else
    rm -f runs_5090.json.tmp; echo "$(ts) 5090 unreachable, keeping old snapshot" >> "$LOG"
  fi
  # A100 box (tianyiyun-30109, owner-designated; env on local NVMe behind a /dev/shm symlink): SafeDuo lives on the NFS under safeduo_a100/safeduo
  if timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=25 $A100 'SAFEDUO_NODE=tianyiyun-30109 SAFEDUO_ARTIFACTS=/home/dataset-assist-0/liyufeng/safeduo_a100/safeduo/artifacts python3 -' < "$TOOLS/collect_remote.py" > runs_a100.json.tmp 2>/dev/null && [ -s runs_a100.json.tmp ] && python3 -c "import json;json.load(open('runs_a100.json.tmp'))" 2>/dev/null; then
    mv runs_a100.json.tmp runs_a100.json; echo "$(ts) a100 snapshot ok" >> "$LOG"
  else
    rm -f runs_a100.json.tmp; echo "$(ts) a100 unreachable, keeping old snapshot" >> "$LOG"
  fi
  python3 - <<'PY'
import json, os, time
servers = []
for f in ("runs_5090.json", "runs_a100.json"):
    if os.path.exists(f):
        try:
            d = json.load(open(f)); d["snap_file"] = f; servers.append(d)
        except Exception:
            pass
primary = servers[0] if servers else {}
out = dict(primary); out["servers"] = servers; out["updated"] = max([s.get("updated", "") for s in servers] + [""]) or time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(out, open("runs.json", "w"), ensure_ascii=False, indent=1)
PY
  mkdir -p clutch_cells
  timeout 120 rsync -az --include='*/' --include='cell_*.json' --include='grid_summary.md' --exclude='*' $REMOTE:~/safeduo/artifacts/clutch/ clutch_cells/ 2>/dev/null && echo "$(ts) cells synced" >> "$LOG"
  python3 "$TOOLS/build_gates.py" clutch_cells gates.json >/dev/null 2>&1 || echo "$(ts) build_gates failed" >> "$LOG"
fi
python3 "$TOOLS/build_matrix.py" "$DATA" >/dev/null 2>&1 || echo "$(ts) build_matrix failed" >> "$LOG"
python3 -c "import json;[json.load(open(f)) for f in ('status.json','runs.json','gates.json','matrix.json','plan.json','roadmap.json','tasks.json')]" 2>/dev/null || { echo "$(ts) invalid json, abort" >> "$LOG"; exit 1; }
git add -A >/dev/null 2>&1
if ! git diff --cached --quiet; then
  git commit -q -m "data: $(date '+%Y-%m-%d %H:%M')" && (git push -q origin data 2>>"$LOG" || (git pull -q --rebase origin data && git push -q origin data 2>>"$LOG")) && echo "$(ts) data pushed" >> "$LOG"
else
  echo "$(ts) data no change" >> "$LOG"
fi
# videos: pull + transcode + push the media repo (independent of the data branch)
bash "$TOOLS/sync_media.sh" >/dev/null 2>&1
# MASTER_REPORT mirror (living doc maintained by the experiment line) into the Pages branch
# NOTE: ~/Desktop is TCC-protected; under launchd/cron this read fails (Operation not permitted) unless
# /bin/bash has Full Disk Access. Run this script interactively after editing MASTER to mirror it.
if cat "$SRC/paper/MASTER_REPORT.html" > "$TOOLS/.master.tmp" 2>/dev/null && [ -s "$TOOLS/.master.tmp" ]; then
  if ! cmp -s "$TOOLS/.master.tmp" "$TOOLS/MASTER_REPORT.html"; then
    mv "$TOOLS/.master.tmp" "$TOOLS/MASTER_REPORT.html"
    (cd "$TOOLS" && git add MASTER_REPORT.html && git commit -q -m "mirror: MASTER_REPORT $(date '+%m-%d %H:%M')" && git push -q origin main 2>>"$LOG") && echo "$(ts) MASTER mirrored" >> "$LOG"
  fi
  rm -f "$TOOLS/.master.tmp"
else
  rm -f "$TOOLS/.master.tmp"; echo "$(ts) MASTER source not readable here (TCC); run interactively to mirror" >> "$LOG"
fi
exit 0
