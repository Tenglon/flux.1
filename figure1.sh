export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export STORE_PATH="./local_datasets/keremberke/pokemon-classification_latents/checkpoint-6000-oh/unet"
# export STORE_PATH="./local_datasets/keremberke/pokemon-classification_latents/checkpoint-6000-hyp/unet"
# export STORE_PATH1="./local_datasets/Donghyun99/CUB-200-2011_latents/checkpoint-16000-oh-7ppiw6jk"
# export STORE_PATH2="./local_datasets/Donghyun99/CUB-200-2011_latents/checkpoint-16000-hyp-ct72p7rf"
export STORE_PATH1="./local_datasets/Donghyun99/Stanford-Cars_latents/checkpoint-14000-oh-6qq5yxdm"
export STORE_PATH2="./local_datasets/Donghyun99/Stanford-Cars_latents/checkpoint-14000-hyp-lu0zk4jn"

# export STORE_PATH2="./local_datasets/Donghyun99/CUB-200-2011_latents/checkpoint-400000"
# export STORE_PATH2="./local_datasets/Donghyun99/CUB-200-2011_latents/checkpoint-260000"

export OUTPUT_DIR1="./output/finetune/lora/${MODEL_NAME}/${STORE_PATH1}"
export OUTPUT_DIR2="./output/finetune/lora/${MODEL_NAME}/${STORE_PATH2}"

# export DATASET_NAME="./local_datasets/keremberke/pokemon-classification_latents"
# export DATASET_NAME="./local_datasets/Donghyun99/CUB-200-2011_latents"
export DATASET_NAME="./local_datasets/Donghyun99/Stanford-Cars_latents"

python figure1_interpolation.py \
    --model_path ${OUTPUT_DIR1} \
    --pretrained_model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --num_validation_images 11 \
    --guidance_scale 4 \
    --emb_type oh \
    --sample_steps 250 \
    --seed 0

python figure1_interpolation.py \
    --model_path ${OUTPUT_DIR2} \
    --pretrained_model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --num_validation_images 11 \
    --guidance_scale 4 \
    --emb_type hyp \
    --sample_steps 250 \
    --seed 0