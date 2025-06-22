#!/bin/bash

# 使用函数打印帮助信息
usage() {
    echo "Usage: $0 INIT_MODEL_PATH CHECKPOINT_PATH TARGET_DIR"
    echo ""
    echo "Arguments:"
    echo "  INIT_MODEL_PATH   Path to the initial model"
    echo "  CHECKPOINT_PATH   Path to the checkpoint directory"
    echo "  TARGET_DIR        Path to the target output directory"
    echo ""
    exit 1
}

# 检查是否提供了足够的参数
if [ "$#" -ne 3 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

# 按顺序读取参数
INIT_MODEL_PATH="$1"
CHECKPOINT_PATH="$2"
TARGET_DIR="$3"

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
python model_merger.py --backend fsdp --hf_model_path "$INIT_MODEL_PATH" --local_dir "$CHECKPOINT_PATH" --target_dir "$TARGET_DIR"

echo "Model convert done for $TARGET_DIR"