#!/bin/bash
#SBATCH --partition=gpubase_h100_b1
#SBATCH --gres=gpu:h100:4
#SBATCH --account=aip-wenhu
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=eval_mmlupro_qwen3-4b_cft
#SBATCH --output=%j_eval_mmlupro_qwen3-4b_cft.log

nvidia-smi

bash eval_qwen3_4b_cft_rl_mmlu-pro.sh

