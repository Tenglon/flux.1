from datasets import load_dataset, concatenate_datasets, load_from_disk
import os
import torch
from torchvision import transforms
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL
from datasets import DatasetDict, Dataset
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--dataset_name", type=str, default="keremberke/pokemon-classification")
# parser.add_argument("--dataset_name", type=str, default="Donghyun99/Stanford-Cars")
# parser.add_argument("--dataset_name", type=str, default="Donghyun99/CUB-200-2011")
parser.add_argument("--revision", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--train_data_dir", type=str, default=None)
parser.add_argument("--center_crop", type=bool, default=True)
parser.add_argument("--random_flip", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--dataloader_num_workers", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="./local_datasets")
args = parser.parse_args()


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

    # Step 0: 加载数据集
    if args.dataset_name == "keremberke/pokemon-classification":
        dataset = load_dataset(args.dataset_name, 'full', cache_dir=args.cache_dir, data_dir=args.train_data_dir)
        dataset = dataset.rename_column("labels", "label")
    else:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir)
    
    # Step 1: 获取类别
    image_column = 'image'
    class_column = 'label'
    
    class_set = dataset['train'].features[class_column].names
    # print(class_set)

    # Step 2: 合并数据集
    if args.dataset_name == "keremberke/pokemon-classification":
        full_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    else:
        full_dataset = concatenate_datasets([dataset['train'], dataset['test']])

    # Step 3: 处理类别级数据集
    if args.dataset_name == "Donghyun99/CUB-200-2011":
        embs_prefix = './embs/cub_flat_reduced_768D'
    elif args.dataset_name == "Donghyun99/Stanford-Cars":
        embs_prefix = './embs/cars_reduced_768D'
    elif args.dataset_name == "keremberke/pokemon-classification":
        embs_prefix = './embs/pokemon_768D'

    hyp_emb_path = embs_prefix + '_c0.1.pth'
    sph_emb_path = embs_prefix + '_sph.pth'

    class_level_dataset_hyp = Dataset.from_dict(torch.load(hyp_emb_path))
    class_level_dataset_sph = Dataset.from_dict(torch.load(sph_emb_path))
    class_level_dataset_oh = dict()
    class_level_dataset_oh['objects']    = class_level_dataset_hyp['objects']
    class_level_dataset_oh['embeddings'] = torch.eye(len(class_level_dataset_hyp['objects']))
    class_level_dataset_oh = Dataset.from_dict(class_level_dataset_oh)

    # Step 4: 处理样本级数据集
    dataset = full_dataset.map(preprocess_train, batched=True, batch_size=args.train_batch_size, num_proc=args.dataloader_num_workers, desc="处理数据集中...")

    # step5: 创建数据集
    dataset_dict = DatasetDict({
        'sample_level': dataset,  # n_samples rows
        'class_level_hyp': class_level_dataset_hyp,        # n_classes rows
        'class_level_sph': class_level_dataset_sph,        # n_classes rows
        'class_level_oh': class_level_dataset_oh        # n_classes rows
    })

    dataset_dict.save_to_disk(os.path.join(args.output_dir, args.dataset_name + "_latents"))
    # load dataset
    load_dataset = load_from_disk(os.path.join(args.output_dir, args.dataset_name + "_latents"))
    print(load_dataset)