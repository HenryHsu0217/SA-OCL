#!/usr/bin/env bash
# Resolve grouped entrypoints without changing the Python implementation.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
if [[ "$#" -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: bash scripts/run.sh <path/to/script.py|script.sh> [arguments...]"
    echo "Example: bash scripts/run.sh training/paper/train_ours.py --help"
    exit 0
fi
case "$1" in
    /*) GG_ENTRYPOINT="$1" ;;
    *) GG_ENTRYPOINT="${GAUSSIAN_GROUPING_ROOT}/$1" ;;
esac
shift
if [[ ! -f "${GG_ENTRYPOINT}" ]]; then
    echo "Script does not exist: ${GG_ENTRYPOINT}" >&2
    exit 2
fi
case "${GG_ENTRYPOINT}" in
    *.py) exec "${GAUSSIAN_GROUPING_PYTHON:-python}" "${GG_ENTRYPOINT}" "$@" ;;
    *.sh) exec bash "${GG_ENTRYPOINT}" "$@" ;;
    *) echo "Expected a Python or shell script: ${GG_ENTRYPOINT}" >&2; exit 2 ;;
esac
