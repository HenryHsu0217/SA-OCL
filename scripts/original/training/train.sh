#!/bin/bash

# Check if the user provided an argument
# Resolve grouped source paths for direct Bash runs and SLURM spool copies.
if [[ -n "${GAUSSIAN_GROUPING_ROOT:-}" && -f "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh" ]]; then
    source "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
else
    source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../env.sh"
fi

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi


dataset_name="$1"
scale="$2"
dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# Gaussian Grouping training
python training/original/train.py    -s $dataset_folder -r ${scale}  -m output/${dataset_name} --config_file config/gaussian_dataset/train.json

# Segmentation rendering using trained model
python rendering/original/render.py -m output/${dataset_name} --num_classes 256
