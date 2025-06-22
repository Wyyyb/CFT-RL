#!/bin/bash

# source /scratch/y726wang/CFT-RL/Qwen2.5-Math-Eval-0203/.venv/bin/activate
source /home/y726wang/projects/aip-wenhu/y726wang/miniconda3/bin/activate
conda activate eval_math

checkpoint_numbers=(20 40 60 80 100 120 140 160)
# checkpoint_numbers=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 124 128 132 136 140 144 148 152 156 160 164 168 172 176 180 184 188 192 196 200)

export CUDA_VISIBLE_DEVICES=0

# 串行处理每个检查点，每次都使用全部4张卡
for ckpt_num in "${checkpoint_numbers[@]}"; do
    summary_path="/scratch/y726wang/CFT-RL/eval_results_qwen3_4b_base_rl/summary.txt"
    saved_ckpt_path="/scratch/y726wang/CFT-RL/verl-data/save_checkpoints/cft-rl-base-0621_webinstruct-verified_Qwen3-4B-Base/global_step_${ckpt_num}/actor/"
    model_path="/scratch/y726wang/CFT-RL/verl-data/output_models/cft-rl_webinstruct-verified_Qwen3-4B-Base/ckpt-${ckpt_num}"
    output_path="/scratch/y726wang/CFT-RL/eval_results_qwen3_4b_base_rl/ckpt-${ckpt_num}/"
    INIT_MODEL_PATH="/scratch/y726wang/CFT-RL/verl-data/Qwen3-4B-Base"
    CHECKPOINT_PATH=${saved_ckpt_path}
    TARGET_DIR=${model_path}

    echo "Converting checkpoint ${ckpt_num}"
    cd /scratch/y726wang/CFT-RL/u_utils/
    bash convert_ckpt.sh ${INIT_MODEL_PATH} ${CHECKPOINT_PATH} ${TARGET_DIR}

    echo "Processing checkpoint ${ckpt_num}"

    cd /scratch/y726wang/CFT-RL/Qwen2.5-Math-Eval-0203/scripts
    mkdir -p $output_path

    bash evaluate_qwen3_no_think.sh $model_path $output_path $summary_path

    echo "Finished processing checkpoint ${ckpt_num}"
done

echo "All checkpoints processed successfully!"