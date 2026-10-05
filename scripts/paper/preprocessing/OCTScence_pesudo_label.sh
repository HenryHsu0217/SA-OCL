#!/bin/bash
#SBATCH --job-name=deva_mask
#SBATCH --array=49
#SBATCH --output=logs/DEVA/deva_mask_640x480_%a.out
#SBATCH --error=logs/DEVA/deva_mask_640x480_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu-a100-80g
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
#START_SCENE=4983
#END_SCENE=4999

# ---- user config ----
RES="128x128"   # e.g., "480" or "full"
OCT_ROOT="${REPO_ROOT}/GSO"
DEVA_ROOT="${REPO_ROOT}/third_party/deva"
OUT_ROOT="${REPO_ROOT}/data/OCTScenes_grouped/${RES}/DEVA_mask"
SIZE=128  # DEVA --size
CHUNK=4   # DEVA --chunk_size
TH=0.7    # --SAM_PRED_IOU_THRESHOLD
# ---------------------

GROUP=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
GROUP_TAR="${OCT_ROOT}/${RES}/compressed/scenes_${GROUP}.tar"

mkdir -p logs
mkdir -p "${OUT_ROOT}/group_${GROUP}"

# Temp workspace (prefers SLURM_TMPDIR)
TMPBASE="${SLURM_TMPDIR:-/tmp}"
WORKDIR="${TMPBASE}/oct_deva_g${GROUP}_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}"

echo "Job: ${SLURM_JOB_ID}, Array task: ${GROUP}"
echo "Group tar: ${GROUP_TAR}"
echo "Workdir: ${WORKDIR}"
echo "Out: ${OUT_ROOT}/group_${GROUP}"

if [ ! -f "${GROUP_TAR}" ]; then
  echo "ERROR: Missing tar: ${GROUP_TAR}"
  exit 2
fi

# Go to DEVA repo
cd "${DEVA_ROOT}"

# If your cluster needs module loads / conda activate, do it here
# module load cuda/...
# source activate deva_env

# Detect scene folders inside the tar (assumes 10 scenes per group)
# This extracts top-level directories like 0000, 0001, ...
SCENES=$(tar -tf "${GROUP_TAR}" | awk -F/ 'NF>=2 {print $1}' | sort -u)
#SCENES=$(tar -tf "${GROUP_TAR}" | awk -F/ 'NF>=2 {print $1}' | sort -u -n)
echo "Scenes in group ${GROUP}:"
echo "${SCENES}"

for SCENE in ${SCENES}; do
  #if [ "${SCENE}" -lt "${START_SCENE}" ] || [ "${SCENE}" -gt "${END_SCENE}" ]; then
  #  continue
  #fi
  echo "---- Processing scene ${SCENE} (group ${GROUP}) ----"

  SCENE_TMP="${WORKDIR}/${SCENE}"
  mkdir -p "${SCENE_TMP}"

  # Extract only this scene folder (60 frames) into temp
  tar -xf "${GROUP_TAR}" -C "${SCENE_TMP}" "${SCENE}/"

  IMG_DIR="${SCENE_TMP}/${SCENE}"
  echo "Resizing images in ${IMG_DIR} to 256x256..."
  python -c "import cv2, glob, os; exts=('.jpg','.jpeg','.png'); [os.remove(f) if os.path.isfile(f) and not f.lower().endswith(exts) else (lambda i: cv2.imwrite(f, cv2.resize(i,(128,128))) if i is not None else None)(cv2.imread(f)) for f in glob.glob(os.path.join('${IMG_DIR}','*'))]"
  # Output for DEVA
  SCENE_OUT="${SCENE_TMP}/deva_out"
  mkdir -p "${SCENE_OUT}"
  find "${IMG_DIR}" -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
  find "${IMG_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
  find "${IMG_DIR}" -maxdepth 1 -type f ! \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -delete
  
  # Run DEVA (gray ID masks via --use_short_id)
  python demo/demo_automatic.py \
    --chunk_size "${CHUNK}" \
    --img_path "${IMG_DIR}" \
    --amp \
    --temporal_setting semionline \
    --size "${SIZE}" \
    --output "${SCENE_OUT}" \
    --use_short_id \
    --SAM_PRED_IOU_THRESHOLD "${TH}"

  # Copy results out
  # We store per scene under group folder:
  # OUT_ROOT/group_<g>/<scene>/object_mask/*.png
  DEST="${OUT_ROOT}/${SCENE}/"
  mkdir -p "${DEST}"
  cp -r "${SCENE_OUT}/Annotations/." "${DEST}/"

  # Cleanup temp for this scene to keep file count low
  rm -rf "${SCENE_TMP}"

done

# Cleanup group workdir
rm -rf "${WORKDIR}"

echo "Done group ${GROUP}"
