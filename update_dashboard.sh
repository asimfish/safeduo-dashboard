#!/bin/bash
# Collect SafeDuo dashboard snapshots (bjxy_5090 GPU/runs/gates) and push to the `data` branch;
# mirror MASTER_REPORT.html into the Pages branch when it changed.
# Usage: update_dashboard.sh [--no-server]
set -u
DATA=~/Code/safeduo-dashboard-data
TOOLS=~/Code/safeduo-dashboard
SRC=/Users/liyufeng/Desktop/research/safeduo
REMOTE=bjxy_5090
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
  if timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=20 $REMOTE 'python3 -' < "$TOOLS/collect_remote.py" > runs.json.tmp 2>/dev/null && [ -s runs.json.tmp ] && python3 -c "import json;json.load(open('runs.json.tmp'))" 2>/dev/null; then
    mv runs.json.tmp runs.json; echo "$(ts) server snapshot ok" >> "$LOG"
  else
    rm -f runs.json.tmp; echo "$(ts) server unreachable, keeping old runs.json" >> "$LOG"
  fi
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
# MASTER_REPORT mirror (living doc maintained by the experiment line) into the Pages branch
if [ -f "$SRC/paper/MASTER_REPORT.html" ] && ! cmp -s "$SRC/paper/MASTER_REPORT.html" "$TOOLS/MASTER_REPORT.html"; then
  cp "$SRC/paper/MASTER_REPORT.html" "$TOOLS/MASTER_REPORT.html"
  (cd "$TOOLS" && git add MASTER_REPORT.html && git commit -q -m "mirror: MASTER_REPORT $(date '+%m-%d %H:%M')" && git push -q origin main 2>>"$LOG") && echo "$(ts) MASTER mirrored" >> "$LOG"
fi
exit 0
