set -euo pipefail

python tools/merge_free_multi_epoch_three_branch.py \
  --template-ckpt "${TEMPLATE_CKPT:?Set TEMPLATE_CKPT}" \
  --frame-anchor-ckpt "${FRAME_ANCHOR_CKPT:?Set FRAME_ANCHOR_CKPT}" \
  --stream-anchor-ckpt "${STREAM_ANCHOR_CKPT:?Set STREAM_ANCHOR_CKPT}" \
  --voxel-anchor-ckpt "${VOXEL_ANCHOR_CKPT:?Set VOXEL_ANCHOR_CKPT}" \
  --frame-specialist-ckpt "${FRAME_SPECIALIST_CKPT:?Set FRAME_SPECIALIST_CKPT}" \
  --stream-specialist-ckpt "${STREAM_SPECIALIST_CKPT:?Set STREAM_SPECIALIST_CKPT}" \
  --voxel-specialist-ckpt "${VOXEL_SPECIALIST_CKPT:?Set VOXEL_SPECIALIST_CKPT}" \
  --epoch-ckpts ${EPOCH_CKPTS:?Set EPOCH_CKPTS as a space-separated list} \
  --out "${OUT:-work_dirs/mv_mutual_merged/merged_three_branch.pth}" \
  --method free \
  --merge-groups head,fusion \
  --anchor-mode branch_anchor \
  --fft-mode flatten2d \
  --lambda-mode uniform \
  --expert-source low \
  --expert-rescale paper_abs \
  --cutoff-fusion "${CUTOFF_FUSION:-0.005}" \
  --cutoff-head "${CUTOFF_HEAD:-0.005}" \
  --expert-density-fusion "${EXPERT_DENSITY_FUSION:-0.005}" \
  --expert-density-head "${EXPERT_DENSITY_HEAD:-0.010}" \
  --expert-scale-fusion "${EXPERT_SCALE_FUSION:-0.10}" \
  --expert-scale-head "${EXPERT_SCALE_HEAD:-0.25}" \
  --expert-max-rescale 5.0 \
  --allow-missing \
  --verbose
