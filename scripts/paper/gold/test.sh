#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=256G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --job-name=test_GOCL_OCTA

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
python  GOLD_Code/test_ocl.py \
    --data_name  octa  \
    --data_path  data/OCTScenes_grouped/128x128/image \
    --model_name  gold_kl_sumrec_122i6e8b_0.001regbck_3496339326seed \
    --ocl_ckp_path logs/octa/gold_ocl/best_model.pt \
    --batch_size 4  --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8\
    --seed 3496339326 \


# python  GOLD_Code/test_gold.py \
#     --data_name  unique_gso  \
#     --data_path  /home/tlchen/conference/dataset/gso_multiview_unique.h5  \
#     --model_name  gold_kl_sumrec_122i6e8b_0.001regbck_3496339326seed \
#     --ocl_ckp_path logs/unique_gso/gold_kl_sumrec_122i6e8b_0.001regbck_3496339326seed/best_model.pt \
#     --gold_ckp_path logs/unique_gso/preocl_vqvae_singledis/best_model.pt \
#     --batch_size 4  --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8\
#     --seed 3496339326 \
