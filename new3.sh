#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train_logs/output_train_%j.log
#SBATCH --error=train_logs/error_train_%j.log
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # 每节点只启1个 launcher
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=48:00:00
#SBATCH --account=project_465002133   # Project for billing 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uestc.longteng@gmail.com

SIF=/scratch/project_465002213/images/pytorch270/   # ★ 用 .sif 文件

module use /appl/local/training/modules/AI-20241126/
module load singularity-userfilesystems

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE) # so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# NCCL environment variables for multi-node training
# Increase timeout for large-scale multi-node setups
export NCCL_DEBUG=WARN
# Use new PyTorch environment variable (replaces deprecated NCCL_ASYNC_ERROR_HANDLING)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
# For InfiniBand networks (adjust if using Ethernet)
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# Network interface (adjust based on your cluster)
# export NCCL_SOCKET_IFNAME=ib0
# Additional stability settings for multi-node
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# For debugging GPU mapping issues, uncomment if needed:
# export NCCL_DEBUG=INFO

# Training Environment variables
# Bypass the access limit of huggingface
# export HF_ENDPOINT=https://hf-api.gitee.com
# export HF_HOME=~/.cache/gitee-ai

# Training Environment variables
# export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export DATASET_NAME="./local_datasets/keremberke/pokemon-classification_latents"
# export DATASET_NAME="./local_datasets/Donghyun99/CUB-200-2011_latents"
# export MODEL_NAME="sd-legacy/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export DATASET_NAME="./local_datasets/Donghyun99/Stanford-Cars_latents"
export OUTPUT_DIR="./output/finetune/lora/${MODEL_NAME}/${DATASET_NAME}"

# -------------------------------
# Launch distributed job
# -------------------------------
srun \
singularity exec \
    --env HF_HOME=/scratch/project_465002213/.cache/huggingface \
    --env NCCL_DEBUG=$NCCL_DEBUG \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING \
    --env NCCL_TIMEOUT=$NCCL_TIMEOUT \
    --env NCCL_IB_DISABLE=$NCCL_IB_DISABLE \
    --env NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /users/longteng,/scratch/project_465002213,/scratch/project_465001640 \
    $SIF bash -c "\
        echo \"============================\" && \
        echo \"[Node \${SLURM_NODEID}] Host: \${SLURMD_NODENAME}\" && \
        echo \"[Node \${SLURM_NODEID}] Running pre-check /users/longteng/run-pytorch-cmd.sh\" && \
        date && \
        export MACHINE_RANK=\${SLURM_NODEID} && \
        export NCCL_DEBUG=WARN && \
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && \
        export NCCL_TIMEOUT=3600 && \
        bash /users/longteng/run-pytorch-cmd.sh \
        accelerate launch --multi_gpu \
            --mixed_precision=no \
            --dynamo_backend=no \
            --num_processes $NUM_PROCESSES \
            --num_machines $NNODES \
            --machine_rank \${SLURM_NODEID} \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MASTER_PORT \
            teng_dit.py \
                --pretrained_model_name_or_path=$MODEL_NAME \
                --dataset_name=$DATASET_NAME \
                --max_train_samples=18000 \
                --dataloader_num_workers=1 \
                --resolution=256 --center_crop --random_flip \
                --resolution_latent=32 \
                --train_batch_size=16 \
                --gradient_accumulation_steps=1 \
                --mixed_precision="bf16" \
                --max_train_steps=400000 \
                --learning_rate=1e-4 \
                --snr_gamma=5.0 \
                --max_grad_norm=1 \
                --lr_scheduler="cosine" --lr_warmup_steps=0 \
                --output_dir=${OUTPUT_DIR} \
                --checkpointing_steps=1000 \
                --num_validation_images=16 \
                --validation_epochs=100 \
                --guidance_scale=4 \
                --sample_steps=250 \
                --emb_type="hyp" \
                --seed=42
                "