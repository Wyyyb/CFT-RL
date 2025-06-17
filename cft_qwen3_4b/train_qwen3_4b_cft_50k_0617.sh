#!/usr/bin/env bash
set -x

source /map-vepfs/miniconda3/bin/activate
conda activate ys_swift

MODEL_PATH="/map-vepfs/yubo/models/Qwen3-4B-Base"

DATA_PATH="/map-vepfs/yubo/CFT-RL/local_data/webinstruct_cft_50k_swift_0617.json"

OUTPUT_DIR="/map-vepfs/yubo/CFT-RL/output_models/qwen_3_4b_cft_0617/"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=0

cd /map-vepfs/yubo/CriticCoT/ms-swift

torchrun \
    --nproc_per_node 1 \
    --standalone \
    swift/cli/sft.py\
    --use_hf True \
    \
    --model $MODEL_PATH \
    --train_type full \
    --torch_dtype bfloat16 \
    \
    --dataset $DATA_PATH \
    --split_dataset_ratio 0 \
    --dataset_num_proc 1 \
    --streaming False \
    --strict False \
    --deepspeed zero3 \
    --remove_unused_columns False \
    --dataloader_num_workers 1 \
    \
    --truncation_strategy delete \
    \
    --output_dir $OUTPUT_DIR \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --weight_decay 0.05 \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --report_to none \
    --logging_first_step True \
    --logging_steps 1 \
    \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 512 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_only_model True \
    --warmup_ratio 0.2 \
    --ddp_backend "nccl" \
    \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False\
    --attn_impl flash_attn \

    # --attn_impl flash_attn \

    # --save_strategy "epoch" \
    # --save_strategy "steps" \
    # --save_steps 109 \
    # --save_total_limit 5 \
    # --deepspeed zero3 \
    # --max_steps -1 \
    # --device_map auto \
    # --override_exist_file True \
    # --eval_strategy None \
    # --custom_dataset_in

