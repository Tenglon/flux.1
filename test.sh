export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export DATASET_NAME="./local_datasets/Donghyun99/Stanford-Cars_latents"
export OUTPUT_DIR="./output/finetune/lora/${MODEL_NAME}/${DATASET_NAME}"

python teng_dit.py                --pretrained_model_name_or_path=$MODEL_NAME \
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
                --validation_epochs=10 \
                --guidance_scale=4 \
                --sample_steps=250 \
                --emb_type="hyp" \
                --seed=42