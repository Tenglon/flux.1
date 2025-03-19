# Bypass the access limit of huggingface
# export HF_ENDPOINT=https://hf-api.gitee.com
# export HF_HOME=~/.cache/gitee-ai

from datasets import load_dataset, concatenate_datasets, load_from_disk
import os
import torch
from torchvision import transforms
from utils import tokenize_captions, tokenize_captions2
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL
# os.environ["HF_ENDPOINT"] = "https://hf-api.gitee.com"
# os.environ["HF_HOME"] = "~/.cache/gitee-ai"



class args:
    pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
    revision = None
    # dataset_name = "keremberke/pokemon-classification" # 7201, 110 类
    # dataset_name = "Donghyun99/CUB-200-2011" # 11788, 200 classes
    # dataset_name = "Donghyun99/Stanford-Cars" # 16185, 196 classes
    dataset_name = "./keremberke/pokemon-classification_latents.hf"
    # dataset_name = "./Donghyun99/CUB-200-2011_latents.hf"
    # dataset_name = "./Donghyun99/Stanford-Cars_latents.hf"
    dataset_config_name = None
    cache_dir = None
    train_data_dir = None
    center_crop = True
    random_flip = True
    resolution = 512
    train_batch_size = 4
    dataloader_num_workers = 0
    class_embedding_dim = 768


DATASET_NAME_MAPPING = {
    # "lambdalabs/naruto-blip-captions": ("image", "text"),
    # "wanghaofan/pokemon-wiki-captions": ("image", "text_en", "name_en", "text_zh", "name_zh"),
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


tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
)


def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    # examples["input_ids"] = tokenize_captions2(tokenizer, examples, class_column, PROMPT_TEMPLATE)
    examples["class_labels"] = examples[class_column]
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    latents_list = [torch.tensor(example['latents']) for example in examples]
    latents = torch.stack(latents_list)
    # input_ids = torch.stack([example["input_ids"] for example in examples])
    class_labels = torch.tensor([example["class_labels"] for example in examples])
    cond_embeddings = class_embeddings[class_labels]
    return {"pixel_values": pixel_values, "class_labels": class_labels, "cond_embeddings": cond_embeddings, "latents": latents}


if __name__ == "__main__":


    dataset = load_from_disk(args.dataset_name)
    image_column = 'image'
    class_column = 'labels'
    latents_column = 'latents'
    
    class_set = dataset.features[class_column].names
    print(class_set)
    class_embeddings = torch.randn(len(class_set), args.class_embedding_dim)

    train_dataset = dataset.with_transform(preprocess_train)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    for batch in train_dataloader:
        imgs, class_labels, cond_embeddings, latents = batch['pixel_values'], batch['class_labels'], batch['cond_embeddings'], batch['latents']
        break
