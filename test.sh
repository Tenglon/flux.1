


export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="sd-legacy/stable-diffusion-v1-5"
export DATASET_NAME="./local_datasets/Donghyun99/Stanford-Cars_latents"
# export DATASET_NAME="./local_datasets/Donghyun99/CUB-200-2011_latents"
export OUTPUT_DIR="./output/finetune/lora/${MODEL_NAME}/${DATASET_NAME}_hyp"

# python \
accelerate launch --multi-gpu --num_processes 4 --mixed_precision=bf16 --main_process_port 29501 \
teng_dit.py                --pretrained_model_name_or_path=$MODEL_NAME \
                --dataset_name=$DATASET_NAME \
                --max_train_samples=18000 \
                --dataloader_num_workers=0 \
                --resolution=256 --center_crop --random_flip \
                --resolution_latent=32 \
                --train_batch_size=16 \
                --gradient_accumulation_steps=1 \
                --mixed_precision="bf16" \
                --learning_rate=1e-4 \
                --max_train_steps=400000 \
                --max_grad_norm=1 \
                --lr_scheduler="cosine" --lr_warmup_steps=0 \
                --output_dir=${OUTPUT_DIR} \
                --checkpointing_steps=30000 \
                --num_validation_images=16 \
                --validation_epochs=20 \
                --guidance_scale=4 \
                --sample_steps=50 \
                --emb_type="hyp" \
                --seed=42