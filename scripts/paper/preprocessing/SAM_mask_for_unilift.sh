#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=256G
#SBATCH --output=logs/SAM_UNI_%A.out
#SBATCH --job-name=SAM_UNI


# Resolve grouped source paths for direct Bash runs and SLURM spool copies.
if [[ -n "${GAUSSIAN_GROUPING_ROOT:-}" && -f "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh" ]]; then
    source "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
else
    source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../env.sh"
fi

source ~/.bashrc
conda activate gaussian_grouping
source "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh"


python preprocessing/paper/SAM_mask_for_unilift.py \
  --img_root data/GSO_new/128x128/image \
  --out_root data/GSO_new/128x128/SAM_mask_Uni \
  --scene_start 1900 --scene_end 1999 \
  --sam_checkpoint third_party/deva/saves/sam_vit_h_4b8939.pth \
  --model_type vit_h \
  --pred_iou_thresh 0.7 \
  --stability_score_thresh 0.95 \
  --points_per_side 32 \
  --device cuda