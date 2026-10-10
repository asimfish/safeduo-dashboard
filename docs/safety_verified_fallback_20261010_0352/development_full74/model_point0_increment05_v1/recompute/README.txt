This is a portable CPU MODEL_ONLY recomputation bundle, not a simulator or admission tool.
Download every leaf listed in RECOMPUTE_BUNDLE_V1.json into one directory.
The FK, geometry, evidence IO and classification kernels are exact original source bytes.
io_v2.py is an explicitly separate portable dependency adapter; no physics math changed.
Requires Python 3 and NumPy. Do not open the formal holdout.

Verify downloaded bytes without evaluating any model:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -B recompute_model_only.py --verify-only

Recompute one registered 32-row block:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -B recompute_model_only.py --out NEW_DIR --first-block 0 --blocks 1

Recompute all8192 in256 serial32-row blocks (approximately591MB new gap outputs):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -B recompute_model_only.py --out NEW_DIR

Original model gaps stay on NAS with all256 block SHA ledgers. The report did not re-evaluate all gaps.
This recomputation cannot establish native/camera eligibility, drive peak, contact, global error bound or safety.
Partial recomputation never creates a full8192 schedule.
