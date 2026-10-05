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
    echo "Usage: $0 <output_folder> <config_file> "
    exit 1
fi


output_folder="$1"
config_file="$2"

if [ ! -d "$output_folder" ]; then
    echo "Error: Folder '$output_folder' does not exist."
    exit 2
fi



# Remove the selected object
python editing/original/edit_object_inpaint.py  -m ${output_folder} --config_file ${config_file}

