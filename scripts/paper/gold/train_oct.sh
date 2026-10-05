#!/bin/bash
#SBATCH --time=10:00:00

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=256G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --job-name=GOCL_GSO

# Resolve grouped source paths for direct Bash runs and SLURM spool copies.
if [[ -n "${GAUSSIAN_GROUPING_ROOT:-}" && -f "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh" ]]; then
    source "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
else
    source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../../env.sh"
fi

export CUDA_LAUNCH_BLOCKING=1

source ~/.bashrc
conda activate gaussian_grouping
source "${GAUSSIAN_GROUPING_ROOT}/scripts/env.sh"
# Train the Global Object-Centric Learning moudel of GOLD for feature reconstruction
torchrun --standalone --nproc_per_node 1 --master_port 20510 GOLD_Code/train_ocl.py \
    --data_name  octa  \
    --data_path  data/OCTScenes_grouped/128x128/image\
    --model_name  gold_origin_ocl_oct_continued\
    --batch_size 8  --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8\
    --seed 3496339326 --epochs 30 --num_cls 11\
    --reg_start 0.1 --reg_final 0.1 --reg_steps 697500 \
    --kld_start 1 --kld_final 1 --kld_start_steps 0  --kld_final_steps 80000\
    --tau_start 1.0 --tau_final 0.1 --tau_start_steps 100000  --tau_final_steps 300000 --beta 100 --num_iterations 2\
    --checkpoint_path logs/octa/gold_origin_ocl_cls_11_reg_01_iter_3_batch_8_declr4e4_20epoch/checkpoint.pt.tar
    
# Train the Image Encoder-Decoder module for image reconstruction
#python -m torch.distributed.launch --nproc_per_node 1  --master_port 20510 --use-env GOLD_Code/train_vqvae.py \
#    --data_name  gso  \
#    --data_path  /home/usr/dataset/gso.h5  \
#    --ocl_ckp_path logs/gso/gold_ocl/best_model.pt \
#    --model_name  gold_vqdec\
#    --batch_size 4 --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8\
#    --epochs 1000  \
#    --seed 3496339326 \
    # --checkpoint_path logs/gso/gold_vqdec/checkpoint.pt.tar

