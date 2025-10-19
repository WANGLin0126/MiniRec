#!/bin/bash
# ================== 基础环境设置 ==================
export TERM=xterm-256color
export LANG=en_US.UTF-8
# 设置 NCCL 调试环境变量
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_TENSOR_PARALLEL_SIZE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1   # 禁掉 InfiniBand，避免误选



NUM_PROCESSES=4
MAIN_PROCESS_PORT=20138


# ================== 配置 ==================
MODEL='gemma'   # 可选: 'qwen3b', 'qwen1.5b', 'gemma', 'deepseek'

# DATASET_CAT='Musical_Instruments'
DATASET_CAT='CDs_and_Vinyl'

DATASET_DIR="data/${DATASET_CAT}_0_2022-10-2023-10"

# ================== 运行多个任务 ==================
for n in 27498; do
    RUN_NAME="difficulty-${n}-${MODEL}-${DATASET_CAT}"
    echo "🚀 正在运行 run_name=${RUN_NAME}..."

    accelerate launch \
        --num_processes=$NUM_PROCESSES --config_file=accelerates/deepspeed_config.yaml \
        --multi_gpu \
        --num_machines=1 \
        --mixed_precision=fp16 \
        --dynamo_backend=no \
        --main_process_port=$MAIN_PROCESS_PORT \
        gen_difficulty.py \
        --model=$MODEL \
        --dataset_dir=$DATASET_DIR \
        --dataset_category=$DATASET_CAT \
        --n_train=$n \
        --batch_size=8 \
        --max_new_tokens=256 \
        --group_size=4 \
        --run_name=$RUN_NAME \
        --use_vllm=False \
        --checkpoint_path="/storage_fast/lwang/SeqRecDistill/RRec/checkpoints/checkpoint-256-gemma-CDs_and_Vinyl-1/checkpoint-28"

    echo "✅ 已完成 ${RUN_NAME}"
    echo "-------------------------------------------"
done

echo "🎉 所有 forward 任务已执行完毕！"