#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=256G
#SBATCH --output=logs/METRIC_Uni_sam2_%A_%a.out
#SBATCH --job-name=METRIC_Uni_sam2
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

START_SCENE=3100
SCENES_PER_JOB=10

JOB_START=$((START_SCENE + SLURM_ARRAY_TASK_ID * SCENES_PER_JOB))
JOB_END=$((JOB_START + SCENES_PER_JOB - 1))

echo "OCT Dataset"
for SCENE in $(seq ${JOB_START} ${JOB_END}); do
    echo "=== Scene ${SCENE} ==="
    python training/baselines/train_unified_lift.py -s data/OCTScenes_grouped -m output/COLMAP/OCTA/${SCENE}_Uni_sam2 \
        --config_file config/gaussian_dataset/train.json --size 128 --scene_id ${SCENE} --train_split --eval --OCT --sam2  --uniflift

    python rendering/baselines/render_unilift.py -m output/COLMAP/OCTA/${SCENE}_Uni_sam2  --num_classes 256 --OCT --sam2 --uniflift

    if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
        python evaluation/paper/metric.py \
            --model_path output/COLMAP/OCTA/${SCENE}_Uni_sam2   \
            --output evaluation/reports/COLMAP/OCTA/Uni/Uni_sam2 --test --save
    else
        python evaluation/paper/metric.py \
            --model_path output/COLMAP/OCTA/${SCENE}_Uni_sam2  \
            --output evaluation/reports/COLMAP/OCTA/Uni/Uni_sam2  --test
    fi
done

START_SCENE=1900
SCENES_PER_JOB=10
JOB_START=$((START_SCENE + SLURM_ARRAY_TASK_ID * SCENES_PER_JOB))
JOB_END=$((JOB_START + SCENES_PER_JOB - 1))
echo "GSO Dataset"
for SCENE in $(seq ${JOB_START} ${JOB_END}); do
    echo "=== Scene ${SCENE} ==="
    python training/baselines/train_unified_lift.py -s data/GSO_new -m output/COLMAP/GSO/${SCENE}_Uni_sam2  \
        --config_file config/gaussian_dataset/train.json --size 128 --scene_id ${SCENE} --train_split --eval --GSO --sam2  --uniflift

    python rendering/baselines/render_unilift.py -m output/COLMAP/GSO/${SCENE}_Uni_sam2 --num_classes 256 --GSO --sam2 --uniflift

    if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
        python evaluation/paper/metric.py \
            --model_path output/COLMAP/GSO/${SCENE}_Uni_sam2 \
            --output evaluation/reports/COLMAP/GSO/Uni/Uni_sam2 --test --save
    else
        python evaluation/paper/metric.py \
            --model_path output/COLMAP/GSO/${SCENE}_Uni_sam2  \
            --output evaluation/reports/COLMAP/GSO/Uni/Uni_sam2  --test
    fi
done