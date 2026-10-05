#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --cpus-per-task=6
#SBATCH --mem=256G
#SBATCH --output=logs/METRIC_our_%A_%a.out
#SBATCH --job-name=METRIC_our
#SBATCH --array=0-9

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
module load triton/2025.1-gcc
module load cuda
module load gcc


START_SCENE=1900
SCENES_PER_JOB=10

JOB_START=$((START_SCENE + SLURM_ARRAY_TASK_ID * SCENES_PER_JOB))
JOB_END=$((JOB_START + SCENES_PER_JOB - 1))

for SCENE in $(seq ${JOB_START} ${JOB_END}); do
    echo "=== Scene ${SCENE} ==="
    python training/paper/train_GSO.py -s data/GSO_new -m output/COLMAP/GSO_new/${SCENE}_ours\
        --config_file config/gaussian_dataset/train.json --size 128 --scene_id ${SCENE} --train_split --eval --ours --config_file config/gaussian_dataset/train.json --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8 --checkpoint_path logs/octa/gso_new/best_model.pt --regloss3d --GSO

    python rendering/paper/render_ours_GSO.py -m output/COLMAP/GSO_new/${SCENE}_ours --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8 --checkpoint_path logs/octa/gso_new/best_model.pt --GSO --ours
    
    if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
        python evaluation/paper/metric.py \
            --model_path output/COLMAP/GSO_new/${SCENE}_ours\
            --output evaluation/reports/COLMAP/GSO_new/ours/ours_new_h200 --test --ours --save
    else
         python evaluation/paper/metric.py \
            --model_path output/COLMAP/GSO_new/${SCENE}_ours\
            --output evaluation/reports/COLMAP/GSO_new/ours/ours_new_h200 --test --ours 
    fi
done
