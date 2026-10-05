#!/bin/bash
#SBATCH --job-name=sam2_mask
#SBATCH --array=0-3
#SBATCH --output=logs/sam2/sam2_mask_%a_%A.out
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
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
START_SCENE=1900
SCENES_PER_JOB=25

JOB_START=$((START_SCENE + SLURM_ARRAY_TASK_ID * SCENES_PER_JOB))
JOB_END=$((JOB_START + SCENES_PER_JOB - 1))

for SCENE in $(seq ${JOB_START} ${JOB_END}); do
      python preprocessing/paper/SAM2/sam2_mask.py \
      --img_dir data/GSO_new/128x128/image \
      --out_dir data/GSO_new/128x128/SAM2_mask \
      --reprompt_every 15 \
      --ckpt preprocessing/paper/SAM2/weights/sam2.1_hiera_base_plus.pt \
      --cfg configs/sam2.1/sam2.1_hiera_b+.yaml --scene_id $SCENE
done


