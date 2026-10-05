#!/bin/bash
#SBATCH --job-name=deva_mask
#SBATCH --array=0-3                   # 4 jobs (0, 1, 2, 3)
#SBATCH --output=logs/DEVA/deva_GSO_%a.out
#SBATCH --error=logs/DEVA/deva_GSO_%a.err
#SBATCH --time=04:00:00               # Increased time (25 scenes takes longer than 10)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h200-141g-short
#SBATCH --gres=gpu:1

# Resolve grouped source paths for direct Bash runs and SLURM spool copies.
if [[ -n "${GAUSSIAN_GROUPING_ROOT:-}" && -f "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh" ]]; then
    source "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
else
    source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../env.sh"
fi

set -euo pipefail

REPO_ROOT="${GAUSSIAN_GROUPING_ROOT}"
module load mamba
set +u
source ~/.bashrc
source "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh"
export TORCH_HOME="${REPO_ROOT}/cache/torch"
set -u
source activate gaussian_grouping
mkdir -p "${TORCH_HOME}"
echo "Using python: $(which python)"

# ---- user config ----
RES="128x128"
OCT_ROOT="${REPO_ROOT}/data/GSO_new"
DEVA_ROOT="${REPO_ROOT}/third_party/deva"
OUT_ROOT="${REPO_ROOT}/GSO/${RES}/DEVA_mask"

SIZE=128  # DEVA --size
CHUNK=4   # DEVA --chunk_size
TH=0.7    # --SAM_PRED_IOU_THRESHOLD

# ---- Range Calculation ----
# Global start scene
GLOBAL_START=1900
# How many scenes per single array job
SCENES_PER_JOB=25

# Calculate start and end for this specific array task
# Task 0: 1900 + (0*25) = 1900 to 1924
# ...
# Task 3: 1900 + (3*25) = 1975 to 1999
START_SCENE=$(( GLOBAL_START + (SLURM_ARRAY_TASK_ID * SCENES_PER_JOB) ))
END_SCENE=$(( START_SCENE + SCENES_PER_JOB - 1 ))
# ---------------------------

DATA_SOURCE_DIR="${OCT_ROOT}/${RES}/image"
mkdir -p logs
# Create a specific log folder for this batch range if needed, or keep generic
mkdir -p "${OUT_ROOT}"

# Temp workspace
TMPBASE="${SLURM_TMPDIR:-/tmp}"
WORKDIR="${TMPBASE}/oct_deva_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}"

echo "Job: ${SLURM_JOB_ID}, Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Processing Scenes: ${START_SCENE} to ${END_SCENE}"
echo "Workdir: ${WORKDIR}"

if [ ! -d "${DATA_SOURCE_DIR}" ]; then
  echo "ERROR: Source directory not found: ${DATA_SOURCE_DIR}"
  exit 2
fi

cd "${DEVA_ROOT}"

# Loop through the calculated range
for (( ID=START_SCENE; ID<=END_SCENE; ID++ )); do
  
  # Format ID as 4 digits (e.g., 1905)
  SCENE=$(printf "%04d" "${ID}")
  SOURCE_SCENE_PATH="${DATA_SOURCE_DIR}/${SCENE}"

  # Check if source exists
  if [ ! -d "${SOURCE_SCENE_PATH}" ]; then
    echo "Warning: Scene folder ${SCENE} not found in ${DATA_SOURCE_DIR}. Skipping."
    continue
  fi

  echo "---- Processing scene ${SCENE} ----"

  SCENE_TMP="${WORKDIR}/${SCENE}"
  mkdir -p "${SCENE_TMP}"

  # Copy images to temp to avoid modifying original data
  IMG_DIR="${SCENE_TMP}/${SCENE}"
  mkdir -p "${IMG_DIR}"
  cp "${SOURCE_SCENE_PATH}/"*.png "${IMG_DIR}/" 2>/dev/null || true

  # Check for files
  COUNT=$(ls "${IMG_DIR}"/*.png 2>/dev/null | wc -l)
  if [ "$COUNT" -eq 0 ]; then
     echo "Error: No PNG files found for scene ${SCENE}"
     rm -rf "${SCENE_TMP}"
     continue
  fi

  # Resize in place (on temp files)
  python -c "import cv2, glob, os; exts=('.jpg','.jpeg','.png'); [os.remove(f) if os.path.isfile(f) and not f.lower().endswith(exts) else (lambda i: cv2.imwrite(f, cv2.resize(i,(128,128))) if i is not None else None)(cv2.imread(f)) for f in glob.glob(os.path.join('${IMG_DIR}','*'))]"
  
  # Output folder
  SCENE_OUT="${SCENE_TMP}/deva_out"
  mkdir -p "${SCENE_OUT}"
  
  # Cleanup non-images
  find "${IMG_DIR}" -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
  find "${IMG_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
  
  # Run DEVA
  python demo/demo_automatic.py \
    --chunk_size "${CHUNK}" \
    --img_path "${IMG_DIR}" \
    --amp \
    --temporal_setting semionline \
    --size "${SIZE}" \
    --output "${SCENE_OUT}" \
    --use_short_id \
    --SAM_PRED_IOU_THRESHOLD "${TH}"

  # Copy results back to permanent storage
  DEST="${OUT_ROOT}/${SCENE}/"
  mkdir -p "${DEST}"
  
  if [ -d "${SCENE_OUT}/Annotations" ]; then
      cp -r "${SCENE_OUT}/Annotations/." "${DEST}/"
  else
      echo "Warning: DEVA generated no Annotations for scene ${SCENE}"
  fi

  # Cleanup temp for this scene
  rm -rf "${SCENE_TMP}"

done

rm -rf "${WORKDIR}"
echo "Done array task ${SLURM_ARRAY_TASK_ID}"