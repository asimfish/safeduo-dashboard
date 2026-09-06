#!/usr/bin/env python3
"""Transcode new SafeDuo videos into the media repo and rebuild index.json.

Usage: build_media_index.py <media_repo_dir> <src_dir> [<src_dir> ...]
For every *.mp4 under the source dirs (recursive) that is not yet in the repo (by name + source size),
write videos/<name>.mp4 (H.264, <=960 px wide, CRF 29, faststart, no audio) and thumbs/<name>.jpg,
then rewrite index.json (newest first) with metadata parsed from the file name:
  r27_bottle7_s0_phys_camU.mp4 -> round r27, family bottle, mode s0 (gated), cam U
  s9task_pick_place_s0t35_a25.mp4 -> round s9, family pick_place, mode s0 theta .35 policy a25
  r34_beam1_raw_camX.mp4 -> round r34, family beam, mode raw
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

repo = Path(sys.argv[1])
srcs = [Path(p) for p in sys.argv[2:]]
(repo / "videos").mkdir(parents=True, exist_ok=True)
(repo / "thumbs").mkdir(parents=True, exist_ok=True)
idx_path = repo / "index.json"
index = {e["name"]: e for e in (json.load(open(idx_path)).get("videos", []) if idx_path.exists() else [])}

FAMILY_WORDS = ("bottle", "rod", "box", "baton", "tray", "relay", "grasp", "pickplace", "pick_place", "handover",
                "four_lift", "relay_chain", "pillow_sheath", "dual_pick_swap", "beam", "reagent", "beaker", "rack",
                "cube", "bar", "pair")


def parse_meta(name: str) -> dict:
    n = name.lower()
    m = re.match(r"^(r\d+|s9task|s9|pair|d5|loop\d|a\d+)", n)
    rnd = m.group(1) if m else "misc"
    fam = next((w for w in FAMILY_WORDS if w in n), "other")
    if fam == "pickplace":
        fam = "pick_place"
    mode = "gated" if re.search(r"_s0|gated|a2\d", n) else ("attach" if "attach" in n else "raw")
    cam = (re.search(r"cam([a-z])", n) or [None, None])[1]
    theta = (re.search(r"t(\d{2})", n) or [None, None])[1]
    policy = (re.search(r"(a\d\d[a-z]?)", n) or [None, None])[1]
    camu = cam.upper() if cam else None
    # wide overview cameras: W (R34 design-line overview), C (R26 oblique demo), and the pre-R26 default 3 m view
    view = "wide" if camu in (None, "W", "C") else "close"
    return {"round": rnd, "family": fam, "mode": mode, "cam": camu, "view": view,
            "theta": theta, "policy": policy}


def probe(path: Path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    try:
        d = json.loads(out)
        st = [s for s in d.get("streams", []) if s.get("width")]
        return float(d["format"]["duration"]), int(st[0]["width"]), int(st[0]["height"])
    except Exception:
        return None, None, None


def transcode(src: Path, dst: Path) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "29", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and dst.exists() and dst.stat().st_size > 1000


def thumb(src: Path, dst: Path, dur: float) -> None:
    t = max(0.5, (dur or 10.0) * 0.55)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                    "-vf", "scale=480:-2", "-q:v", "5", str(dst)], capture_output=True)


added = 0
for src_dir in srcs:
    if not src_dir.exists():
        continue
    for src in sorted(src_dir.rglob("*.mp4")):
        if "frames" in src.parts or src.stat().st_size < 50_000:
            continue
        name = src.stem
        stamp = f"{src.stat().st_size}"
        if name in index and index[name].get("src_size") == stamp and (repo / index[name]["file"]).exists():
            continue
        dur, w, h = probe(src)
        dst = repo / "videos" / f"{name}.mp4"
        if not transcode(src, dst):
            print("transcode failed:", src, file=sys.stderr)
            continue
        thumb(src, repo / "thumbs" / f"{name}.jpg", dur)
        meta = parse_meta(name)
        index[name] = {"name": name, "file": f"videos/{name}.mp4", "thumb": f"thumbs/{name}.jpg",
                       "src": str(src).replace(str(Path.home()), "~"), "src_size": stamp,
                       "mtime": int(src.stat().st_mtime), "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(src.stat().st_mtime)),
                       "dur_s": round(dur, 1) if dur else None, "res": f"{w}x{h}" if w else None,
                       "size_mb": round(dst.stat().st_size / 1e6, 2), **meta}
        added += 1
        print("added", name, meta)

for e in index.values():        # metadata is derived from the name: refresh it for old entries too
    e.update(parse_meta(e["name"]))
videos = sorted(index.values(), key=lambda e: -e["mtime"])
json.dump({"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(videos), "videos": videos},
          open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"index.json: {len(videos)} videos (+{added})")
