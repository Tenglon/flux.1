#!/bin/bash
#SBATCH --job-name=train_pokemon_lora
#SBATCH --output=train_logs/output_pokemon_lora_%j.log
#SBATCH --error=train_logs/error_pokemon_lora_%j.log
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --time=48:00:00  # Adjust time limit as needed

# Load necessary modules (adjust according to your system)
# module purge
# module load cuda
# module load anaconda  # or whatever environment management you use
ml NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1

# Activate your conda environment if needed
# source activate your_env_name

######################
### Set enviroment ###
######################
source activate flux
export GPUS_PER_NODE=4
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
export HYDRA_FULL_ERROR=1

# Add NCCL configuration
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_GID_INDEX=3 #https://github.com/NVIDIA/nccl/issues/426
# export NCCL_P2P_DISABLE=1 #from Johannes

# Get the node names
export MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

MACHINE_RANK=${SLURM_NODEID:-0}

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $MACHINE_RANK \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    --multi_gpu \
    "

export LAUNCHER2="accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend=no \
    "

# Training Environment variables
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="black-forest-labs/FLUX.1-schnell"
# export DATASET_NAME="wanghaofan/pokemon-wiki-captions"
export DATASET_NAME="keremberke/pokemon-classification"
# export DATASET_NAME="Donghyun99/CUB-200-2011"
# export DATASET_NAME="Donghyun99/Stanford-Cars"
export OUTPUT_DIR="./output/finetune/lora/${MODEL_NAME}/${DATASET_NAME}"

# Bypass the access limit of huggingface
# export HF_ENDPOINT=https://hf-api.gitee.com
# export HF_HOME=~/.cache/gitee-ai

export USE_SBATCH=0

if [ $USE_SBATCH -eq 1 ]
then
    srun $LAUNCHER2 \
    train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --dataloader_num_workers=8 \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --mixed_precision="fp16" \
    --use_lora \
    --max_train_steps=15000 \
    --learning_rate=1e-4 \
    --snr_gamma=5.0 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --output_dir=${OUTPUT_DIR} \
    --report_to=wandb \
    --checkpointing_steps=500 \
    --validation_prompt="" \
    --guidance_scale=7.5 \
    --seed=42
else
    $LAUNCHER2 \
    train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --dataloader_num_workers=8 \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=32 \
    --gradient_accumulation_steps=4 \
    --mixed_precision="fp16" \
    --use_lora \
    --max_train_steps=15000 \
    --learning_rate=1e-4 \
    --snr_gamma=5.0 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --output_dir=${OUTPUT_DIR} \
    --report_to=wandb \
    --checkpointing_steps=500 \
    --validation_prompt="a photo of a" \
    --guidance_scale=7.5 \
    --seed=42
fi