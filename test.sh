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


export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export MODEL_NAME="sd-legacy/stable-diffusion-v1-5"
export DATASET_NAME="./local_datasets/Donghyun99/Stanford-Cars_latents"
export OUTPUT_DIR="./output/finetune/lora/${MODEL_NAME}/${DATASET_NAME}"

# python \
accelerate launch --multi-gpu --num_processes 8 --mixed_precision=bf16 \
teng_tiny.py                --pretrained_model_name_or_path=$MODEL_NAME \
                --dataset_name=$DATASET_NAME \
                --max_train_samples=18000 \
                --dataloader_num_workers=8 \
                --resolution=256 --center_crop --random_flip \
                --resolution_latent=32 \
                --train_batch_size=32 \
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
                --validation_epochs=10 \
                --guidance_scale=4 \
                --sample_steps=250 \
                --emb_type="hyp" \
                --seed=42