export INIT_MODEL_PATH="/scratch/y726wang/CFT-RL/verl-data/Qwen3-4B-Base"
export CHECKPOINT_PATH="/scratch/y726wang/CFT-RL/verl-data/save_checkpoints/cft-rl_webinstruct-verified_Qwen3-4B-Base/global_step_60/actor"
# export STEP="global_step_200"
export TARGET_DIR="/scratch/y726wang/CFT-RL/verl-data/output_models/cft-rl_webinstruct-verified_Qwen3-4B-Base_ckpt60"


# 函数：复制 tokenizer 文件
copy_tokenizer_files() {
    local ckpt_path=$1
    local init_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$init_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi
    # 创建目标路径，确保它存在
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    # 复制每个文件
    for filename in "${files_to_copy[@]}"; do
        src="$init_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}

# 调用复制 tokenizer 文件函数
copy_tokenizer_files "$TARGET_DIR" "$INIT_MODEL_PATH"

# 执行模型合并脚本
python model_merger.py --backend fsdp --hf_model_path $INIT_MODEL_PATH --local_dir $CHECKPOINT_PATH --target_dir $TARGET_DIR

echo "Model convert done for $TARGET_DIR"