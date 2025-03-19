from datasets import load_dataset, concatenate_datasets, load_from_disk
import os
import torch
from torchvision import transforms
from utils import tokenize_captions, tokenize_captions2
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL

class args:
    pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
    revision = None
    # dataset_name = "keremberke/pokemon-classification" # 7201, 110 类
    # dataset_name = "Donghyun99/CUB-200-2011" # 11788, 200 classes
    dataset_name = "Donghyun99/Stanford-Cars" # 16185, 196 classes
    dataset_config_name = None
    cache_dir = None
    train_data_dir = None
    center_crop = True
    random_flip = True
    resolution = 512
    train_batch_size = 4
    dataloader_num_workers = 1 # BUG: can only be 1 at the moment
    class_embedding_dim = 768


DATASET_NAME_MAPPING = {
    "keremberke/pokemon-classification": ("image", "labels"),
    "Donghyun99/CUB-200-2011": ("image", "label"),
    "Donghyun99/Stanford-Cars": ("image", "label"),
}

PROMPT_TEMPLATE = 'A detailed and realistic depiction of a {}'

# Preprocessing the datasets.
train_transforms = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
)
vae.to(device)

def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    pixel_values = [train_transforms(image) for image in images]
    # new_examples["input_ids"] = tokenize_captions2(tokenizer, examples, class_column, PROMPT_TEMPLATE)
    # new_examples["class_labels"] = examples[class_column]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float().to(device)
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    examples["latents"] = latents
    return examples



if __name__ == "__main__":

    if args.dataset_name == "keremberke/pokemon-classification":
        dataset = load_dataset(args.dataset_name, 'full', cache_dir=args.cache_dir, data_dir=args.train_data_dir)
        image_column = 'image'
        class_column = 'labels'
    else:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir)
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)

        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        class_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    
    class_set = dataset['train'].features[class_column].names
    # print(class_set)
    class_embeddings = torch.randn(len(class_set), args.class_embedding_dim)

    if args.dataset_name == "keremberke/pokemon-classification":
        full_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    else:
        full_dataset = concatenate_datasets([dataset['train'], dataset['test']])

    # train_dataset = full_dataset.with_transform(preprocess_train)
    train_dataset = full_dataset.map(preprocess_train, batched=True, batch_size=args.train_batch_size, num_proc=args.dataloader_num_workers, desc="处理数据集中...")
    train_dataset.save_to_disk(args.dataset_name + "_latents.hf")

    # load dataset
    load_dataset = load_from_disk(args.dataset_name + "_latents.hf")
    print(load_dataset)