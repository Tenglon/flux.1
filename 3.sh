#!/bin/bash
#SBATCH --job-name=train_pokemon_lora
#SBATCH --output=train_logs/output_tiny%j.log
#SBATCH --error=train_logs/error_tiny%j.log
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=320GB
#SBATCH --time=120:00:00  # Adjust time limit as needed

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

echo $SLURM_NNODES, $((SLURM_NNODES * GPUS_PER_NODE)), $MACHINE_RANK

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
    --num_processes 2 \
    --main_process_port 29501 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend=no \
    "

# Training Environment variables
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="black-forest-labs/FLUX.1-schnell"
# export DATASET_NAME="./local_datasets/keremberke/pokemon-classification_latents"
# export DATASET_NAME="./local_datasets/Donghyun99/CUB-200-2011_latents"
export DATASET_NAME="./local_datasets/Donghyun99/Stanford-Cars_latents"
export OUTPUT_DIR="./output/finetune/lora/${MODEL_NAME}/${DATASET_NAME}"

# Bypass the access limit of huggingface
# export HF_ENDPOINT=https://hf-api.gitee.com
# export HF_HOME=~/.cache/gitee-ai

# export CUDA_VISIBLE_DEVICES=0,1

export USE_SBATCH=0

if [ $USE_SBATCH -eq 1 ]
then
    srun $LAUNCHER \
    teng_tiny.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --max_train_samples=16000 \
    --dataloader_num_workers=8 \
    --resolution=256 --center_crop --random_flip \
    --resolution_latent=32 \
    --train_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --mixed_precision="bf16" \
    --max_train_steps=400000 \
    --learning_rate=1e-4 \
    --snr_gamma=5.0 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --output_dir=${OUTPUT_DIR} \
    --checkpointing_steps=20000 \
    --validation_prompt="a photo of a" \
    --num_validation_images=8 \
    --validation_epochs=10 \
    --guidance_scale=4 \
    --emb_type="hyp" \
    --seed=42
else
    $LAUNCHER2 \
    teng_tiny.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --max_train_samples=16000 \
    --dataloader_num_workers=8 \
    --resolution=256 --center_crop --random_flip \
    --resolution_latent=32 \
    --train_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --mixed_precision="bf16" \
    --max_train_steps=400000 \
    --learning_rate=1e-4 \
    --snr_gamma=5.0 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --output_dir=${OUTPUT_DIR} \
    --checkpointing_steps=20000 \
    --validation_prompt="a photo of a" \
    --num_validation_images=8 \
    --validation_epochs=5 \
    --guidance_scale=4 \
    --emb_type="hyp" \
    --seed=42
fi