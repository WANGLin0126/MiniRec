#!/bin/bash
# ----------------------------
export TERM=xterm-256color
export LANG=en_US.UTF-8
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_TENSOR_PARALLEL_SIZE=1

NUM_PROCESSES=3
MAIN_PROCESS_PORT=20138

# 'qwen3b', 'gemma'
MODEL='gemma'
# ----------------------------

# Musical_Instruments
DATASET_CAT='CDs_and_Vinyl'

DATASET_DIR='data/'$DATASET_CAT'_0_2022-10-2023-10'


# ---------------------------- multi-runs ----------------------------


for n in 1024; do
    for round in 1 2 3; do
        RUN_NAME="minirec-${n}-${MODEL}-${DATASET_CAT}-${round}"
        echo "🚀 Running run_name=${RUN_NAME}..."
        
        accelerate launch --num_processes=$NUM_PROCESSES --config_file=accelerates/deepspeed_config.yaml \
            --main_process_port=$MAIN_PROCESS_PORT train.py \
            --model=$MODEL \
            --dataset_dir=$DATASET_DIR \
            --dataset_category=$DATASET_CAT \
            --train_batch_size=4 \
            --eval_batch_size=8 \
            --max_new_tokens=256 \
            --warmup_steps=32 \
            --seed=42 \
            --num_train_epochs=10 \
            --run_name=$RUN_NAME \
            --group_size=4 \
            --use_vllm=True \
            --shuffle=False \
            --resume_from_checkpoint=False

        echo "✅ Done ${RUN_NAME}"
        echo "-------------------------------------------"
        echo "🔍 Extracting metrics..."
        python extract_rec_metrics_with_settings.py --run_name=$RUN_NAME
    done
done

echo "🎉 All tasks completed!"