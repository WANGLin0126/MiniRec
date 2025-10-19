#!/bin/bash



echo "Starting multi-GPU training with $NUM_GPUS GPUs for $DATASET_SIZE dataset"

export TORCH_DISABLE_ADDR2LINE=1
export NCCL_TIMEOUT=3600  # 30
export NCCL_DEBUG=INFO
export TRANSFORMERS_VERBOSITY=info
export NCCL_IB_TIMEOUT=3600
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3  
export CUDA_LAUNCH_BLOCKING=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

EVAL_BATCH_SIZE=1


echo "Using eval_batch_size=$EVAL_BATCH_SIZE"

accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --mixed_precision=fp16 \
    gen_user_emb_checkpoint.py \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --use_vllm=False \
    # --main_process_port=29500 \

echo "Training completed!"

