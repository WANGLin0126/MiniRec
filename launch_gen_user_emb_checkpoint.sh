#!/bin/bash

# 多卡训练启动脚本 - 优化大数据集训练
# 使用方法: ./launch_distill.sh [num_gpus] [dataset_size]


echo "Starting multi-GPU training with $NUM_GPUS GPUs for $DATASET_SIZE dataset"

export TORCH_DISABLE_ADDR2LINE=1
# 设置NCCL环境变量
export NCCL_TIMEOUT=3600  # 30分钟超时
export NCCL_DEBUG=INFO
export TRANSFORMERS_VERBOSITY=info
export NCCL_IB_TIMEOUT=3600
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整
export CUDA_LAUNCH_BLOCKING=1

# 设置PyTorch环境变量
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# 根据数据集大小调整参数

# 小数据集参数
EVAL_BATCH_SIZE=1


echo "Using eval_batch_size=$EVAL_BATCH_SIZE"

# 启动训练
accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --mixed_precision=fp16 \
    gen_user_emb_checkpoint.py \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --use_vllm=False \
    # --main_process_port=29500 \

echo "Training completed!"

