set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python tools/test.py \
  "${CONFIG:-configs/recognition/mutual_kd/mutiview_mutual_kd_branch3_merged_logit_fusion_test.py}" \
  "${CHECKPOINT:?Set CHECKPOINT to the merged checkpoint}" \
  --work-dir "${WORK_DIR:-work_dirs/mv_mutual_test}" \
  "$@"
