#!/bin/bash

# source /scratch/y726wang/CFT-RL/Qwen2.5-Math-Eval-0203/.venv/bin/activate
source /home/y726wang/projects/aip-wenhu/y726wang/miniconda3/bin/activate
conda activate eval_math

cd ../General-Reasoner
export CUDA_VISIBLE_DEVICES=0

python -m evaluation.eval_mmlupro \
    --model_path /scratch/y726wang/CFT-RL/verl-data/qwen3_4b_cft_ckpt40 \
    --output_file /scratch/y726wang/CFT-RL/verl-data/eval_results/output-mmlupro-qwen3-4b-cft_0624.json

