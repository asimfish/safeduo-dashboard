# SafeDuo empty-hand scene-contact qualification, 2026-10-07/08

This round measures native simulation contacts and hand joint paths in a four-arm scene. It does not run the neural policy or the full arm System0 projector. Objects are parked away and F hands remain open. It does not qualify loaded grasp/carry/place/release, full workspace safety, or real robots.

## Frozen experiments and data

- `development`:120 slots/118 distinct three-parameter target specifications, three static references,360 native cycles,720 U paths.75 full-limit grids,20 IID uniform,20 Latin hypercube and controls. Development records partner normal vectors and counts; no raw scalar point forces, so cannot rule out opposing-force cancellation.
- `validation`:12 frozen goals selected by the preregistered OS permutation and references2/8/20.72 requests; initial contact cache was incorrectly treated as current after initialization writes. All failed admission and physical states are retained, not relabelled as a preflight failure.
- `validation_v2`:same bank after contact-epoch fix. First real physics step remains scored. External NIST/table contact was outside the hand-internal gate: six recorded U paths exceed0.1N all-partner scalar force, one hand speed abort. The later profile34/U_L path stayed latched; its residual contact is not an independently executed profile34 failure.
- `validation_v3`:same bank after all unintended partner contact monitoring, five exact known-unsafe context refusals, and latched empty-hand neutral return.67 admitted paths and five context refusals are distinct outcomes.
- `validation_fresh_v3`:12 new OS-stratified trajectory references, each executing all12 frozen goals:144 cycles,288 attempted U paths,103680 native environment states. Frame selection is fixed in `REGISTRATION_VALIDATION_FRESH_V3.json`; failures are not replaced. The cycles and left/right hands are correlated and must not be treated as IID task trials.
- `paired_v3`:three same-initial-state old-goal bypass/guard pairs at2/8/20, plus one same-initial-state fixture-profile13 hold/return pair at20. Both fixture methods use the same v3 external contact detector and explicitly override target admission. They isolate recovery action, not the complete unchanged v2 algorithm. A recovered contact path still fails full-path safety.

## Criteria and oracles

Normal measurement for all post-development runs is per sensor/partner sum of absolute native point normal magnitudes. `analyze_v3.py` independently reconstructs this in float64 from original valid force/count/start arrays, and checks vector<=scalar with fixed rounding allowance. The scalar contact criterion is<=0.1N per measured sensor/partner; it is a simulation diagnostic, not a human-contact threshold. Native hand speed<=3rad/s; starting, closed and returned hand error<=0.02rad; target command error<=1e-6rad; matrix accounting<=0.001N; unchanged limits, finite data and complete receipts. Every physical state, including first post-init state, counts. No q/qd/object pose writes occur in loops.

The contact views cover measured hand partners, not a full all-body collision-distance or friction-wrench guarantee. Actual arms may move under native motors despite constant desired arm references. The final gate reports that motion and verifies constant arm desired targets and native/cache agreement.

Neutral return begins from the last desired hand target over one second, keeps the abort latch and requires six consecutive fresh safe/neutral/slow frames for recovery. It is restricted to empty hands with parked payloads; it is not loaded-object release or an immediate physical stop. Exact known bad reference/goal combinations are excluded; safe interpolation or novel-context prevention is not established.

## Replay

Keep the archived registration and receipt files byte-for-byte; hashes point to original local paths. Restore production Python source from `source_snapshot/src` and recreate original paths or explicitly document a path-mapped new invocation. The original Python environment was `/home/liyufeng/miniforge3/envs/safeduo`. Direct six USD root asset fingerprints are included. Nested remote textures/material dependencies were not all copied and some render warnings are retained. Without matching simulator/assets/environment, a replay is a new replication, not the same frozen run. Never overwrite an archived output directory.

For a new native invocation use the appropriate frozen generator, source path and registration, with a new output directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$artifact_dir/runtime_v3:/home/liyufeng/safeduo/src" \
LD_PRELOAD=/home/liyufeng/miniforge3/envs/safeduo/lib/libstdc++.so.6 \
OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/liyufeng/miniforge3/envs/safeduo/bin/python \
"$artifact_dir/runtime_v3/native_fresh_extended_v3.py" \
--registration "$artifact_dir/REGISTRATION_VALIDATION_FRESH_V3.json" \
--out "$replication_dir" --headless --device cuda:0 \
--kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'
```

`native_validation_v3.py` is for v3 regression; `native_paired_v3.py` is for the paired cohort. Native receipt/completeness/source SHA gates must pass; Kit process exit0 alone is insufficient. Original run folders and failures must be retained.

CPU assessment of the archived paths:

```bash
/usr/bin/python3 "$artifact_dir/analyze_v3.py" --name validation_v3
/usr/bin/python3 "$artifact_dir/analyze_v3.py" --name validation_fresh_v3
/usr/bin/python3 "$artifact_dir/analyze_v3.py" --name paired_v3
/usr/bin/python3 "$artifact_dir/final_native_gate.py"
```

These scorers read the archived local paths in their sources and write uniquely named result files. Run against a copy to avoid mutating published evidence. For development/V1/V2 retain the original vector scorer results plus independent scalar audits where raw points exist. Do not apply an unavailable scalar oracle retrospectively to development.

## Evidence and delivery

All raw NPZ chunks, initial state snapshots, contact identities, source code, original1280x720 PNGs and eight videos (H264 and VP9) are in the fixed media commit. `VIDEO_RECEIPT.json` binds every60-frame/10fps clip to original frame SHA and native state time. Images are unretouched. `DISPLAY_PAYLOAD.json.gz` stores original precision curves reduced to12-step maxima for display; raw NPZ files are the numerical evidence.

The main page carries only small HTML/JS/summary files to stay within GitHub Pages size limits. Media links bind to an immutable commit; `MEDIA_MANIFEST.json` lists byte sizes and SHA256. The download receipt verifies every dataset entry individually, not the entire history ZIP. Local browser checks inspect all curves/gallery frames with local archived media; public checks inspect all curves, all new reference environments and all eight videos using live URLs. Exact public files and successful Pages action are bound to the final main commit. Scientific limitations are in `ACCEPTANCE_MATRIX.json`; delivery verification does not change scientific verdicts.

A JSON metadata key-order rewrite occurred in `COVERAGE_REPORT.json` during the V1 workflow. Changed bytes, semantic equality and exact restoration are recorded in `SOURCE_METADATA_DRIFT.json`. Its caller is not proven; a same-name top-level `coverage.py` import is a hypothesis. Runtime v2/v3 modules are isolated from the artifact root to prevent that name collision. Physics code/assets/parameters were unchanged.
