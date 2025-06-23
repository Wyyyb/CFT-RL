#!/bin/bash
#SBATCH --partition=gpubase_h100_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --account=aip-wenhu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=eval_general_reasoner_cft
#SBATCH --output=%j_eval_general_reasoner_cft.log

bash eval_qwen3_4b_base_rl.sh

