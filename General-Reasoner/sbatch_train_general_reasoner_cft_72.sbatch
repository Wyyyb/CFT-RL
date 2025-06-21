#!/bin/bash
#SBATCH --partition=gpubase_h100_b4
#SBATCH --gres=gpu:h100:8
#SBATCH --account=aip-wenhu
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=train_general_reasoner_cft_72
#SBATCH --output=%j_train_general_reasoner_cft_72.log

# 激活虚拟环境
source /scratch/y726wang/verl-tool/.venv/bin/activate

# 设置工作目录（根据您的实际目录调整）
export WORKING_DIR=$(pwd)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# 直接运行训练脚本，使用脚本中的默认参数
bash train_general_reasoner_cft.sh