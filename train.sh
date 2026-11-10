set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
python tools/train.py \
  configs/recognition/mutual_kd/mutiview_mutual_kd_branch3.py \
  --seed "${SEED:-0}" \
  --deterministic \
  --work-dir "${WORK_DIR:-work_dirs/mv_mutual}" \
  "$@"
