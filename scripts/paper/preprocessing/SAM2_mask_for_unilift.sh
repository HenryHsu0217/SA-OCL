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
module load mamba
source activate sam2
source "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh"


python preprocessing/paper/SAM2_mask_for_unilift.py \
  --img_root data/GSO_new/128x128/image \
  --out_root data/GSO_new/128x128/SAM2_mask_Uni \
  --scene_start 1900 --scene_end 1999 \
