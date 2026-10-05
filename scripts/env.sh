#!/usr/bin/env bash
# Source from Bash to configure only project paths and the working directory.
GAUSSIAN_GROUPING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export GAUSSIAN_GROUPING_ROOT
export PYTHONPATH="${GAUSSIAN_GROUPING_ROOT}:${GAUSSIAN_GROUPING_ROOT}/third_party/deva:${GAUSSIAN_GROUPING_ROOT}/third_party/cuda/diff_gaussian_rasterization:${GAUSSIAN_GROUPING_ROOT}/third_party/cuda/diff_gaussian_rasterization_16${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${GAUSSIAN_GROUPING_ROOT}/cache/torch"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
cd -- "${GAUSSIAN_GROUPING_ROOT}"
