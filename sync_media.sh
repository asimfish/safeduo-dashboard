#!/bin/bash
# Pull new SafeDuo videos from the GPU box, transcode into ~/Code/safeduo-media (GitHub Pages), push.
# Sources: bjxy_5090:~/Code/safeduo_media (curated per-round clips) + ~/safeduo/artifacts/viz/a8_v4demo/*.mp4
# (raw recorder output, last 3 days). ~/Code is not TCC-protected, so this runs fine under launchd.
set -u
MEDIA=~/Code/safeduo-media
LOCAL=~/Code/safeduo_media
TOOLS=~/Code/safeduo-dashboard
REMOTE=bjxy_5090
LOG=$TOOLS/update.log
ts() { date "+%m-%d %H:%M:%S"; }
mkdir -p "$LOCAL/a8_v4demo"
timeout 300 rsync -az --include='*/' --include='*.mp4' --exclude='*' $REMOTE:~/Code/safeduo_media/ "$LOCAL/" 2>/dev/null && echo "$(ts) media rsync ok" >> "$LOG"
timeout 300 rsync -az --files-from=<(ssh -o BatchMode=yes -o ConnectTimeout=20 $REMOTE 'cd ~/safeduo/artifacts/viz/a8_v4demo && find . -maxdepth 1 -name "*.mp4" -mtime -3 -mmin +2 -printf "%f\n"' 2>/dev/null) $REMOTE:~/safeduo/artifacts/viz/a8_v4demo/ "$LOCAL/a8_v4demo/" 2>/dev/null
cd "$MEDIA" || exit 1
git pull -q --rebase origin main >/dev/null 2>&1 || git rebase --abort >/dev/null 2>&1
python3 "$TOOLS/build_media_index.py" "$MEDIA" "$LOCAL" >> "$LOG" 2>&1
git add -A >/dev/null 2>&1
if ! git diff --cached --quiet; then
  git commit -q -m "media: $(date '+%Y-%m-%d %H:%M')" && (git push -q origin main 2>>"$LOG" || (git pull -q --rebase origin main && git push -q origin main 2>>"$LOG")) && echo "$(ts) media pushed" >> "$LOG"
fi
exit 0
