#!/bin/bash

source /map-vepfs/miniconda3/bin/activate
conda activate yubo_eval

checkpoint_numbers=(10 20 30 40 50 60 70 80 90)
# checkpoint_numbers=(2 6 10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98 102 106 110 114 118 122 126 130 134 138 142 146 150 154 158 162 166 170 174 178 182 186 190 194 198)
 # 使用所有4张GPU
export CUDA_VISIBLE_DEVICES=0,5,6,7

# 串行处理每个检查点，每次都使用全部4张卡
for ckpt_num in "${checkpoint_numbers[@]}"; do
    summary_path="/map-vepfs/yubo/CFT-RL/eval_qwen3_4b_cft_0617/summary.txt"
    model_path="/map-vepfs/yubo/CFT-RL/output_models/qwen_3_4b_cft_0617/v2-20250617-213446/checkpoint-${ckpt_num}"
    output_path="/map-vepfs/yubo/CFT-RL/eval_qwen3_4b_cft_0617/ckpt_${ckpt_num}/"

    echo "Processing checkpoint ${ckpt_num}"

    cd /map-vepfs/yubo/CFT-RL/Qwen2.5-Math-Eval-0203/scripts
    mkdir -p $output_path

    bash evaluate_qwen3_no_think.sh $model_path $output_path $summary_path

    echo "Finished processing checkpoint ${ckpt_num}"
done

echo "All checkpoints processed successfully!"