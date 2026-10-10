#!/bin/bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: REPRODUCE_FAILED_DIAGNOSTIC_V6.sh NEW_OUTPUT_DIRECTORY_UNDER_FAILED_READER_ROOT' >&2
  exit 64
fi
exec env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' prlimit --as=1073741824:1073741824 --core=0:0 /home/liyufeng/miniforge3/envs/safeduo/bin/python -B /home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352/astra/full74_single_case_failed_reader_v6_v1/supervisor.py --mode failed-raw --binding /mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352/astra_full74_independent_reader_v1/single_case_camera_failed_reader_v6_v1/ACTUAL_FAILED_BINDING_V6.json --binding-sha256 548f31fa23aabca33fe14809e2b5b9e3d83a92cd2ca173f7b3a6dae4a2e473ae --out "$1"
