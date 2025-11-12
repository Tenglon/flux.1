from datasets import load_dataset, concatenate_datasets, load_from_disk
import os
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL
from datasets import DatasetDict, Dataset, Features, Array3D, Image, ClassLabel
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
import PIL
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
# parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
# parser.add_argument("--pretrained_model_name_or_path", type=str, default="facebook/DiT-XL-2-256")
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
parser.add_argument("--dataloader_num_workers", type=int, default=4)
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

# 自定义 PyTorch Dataset 类
class ImageDataset(TorchDataset):
    def __init__(self, hf_dataset, image_column, class_column, transform):
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.class_column = class_column
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item[self.image_column]
        image = image.convert("RGB")
        pixel_values = self.transform(image)
        label = item[self.class_column]
        return {
            "idx": idx,
            "pixel_values": pixel_values,
            "label": label,
        }



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

    # Step 4: 处理样本级数据集 - 使用 DataLoader + Accelerator 多 GPU 并行
    accelerator = Accelerator()
    
    # 初始化 VAE（每个进程都需要加载）
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    # 手动将 VAE 移动到对应设备（不使用 prepare，因为这是推理模型）
    vae.to(accelerator.device)
    vae.eval()
    
    # 创建 PyTorch Dataset 和 DataLoader
    image_dataset = ImageDataset(full_dataset, image_column, class_column, train_transforms)
    dataloader = DataLoader(
        image_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    
    # 只对 DataLoader 使用 accelerator 准备（VAE 已经手动移动到设备）
    dataloader = accelerator.prepare(dataloader)
    
    # 处理数据 - 每个进程处理分配给它的数据
    all_latents = {}
    all_images = {}
    all_labels = {}
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(dataloader), desc="处理数据集中...")
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values_batch = batch["pixel_values"]
            labels_batch = batch["label"]
            # 确保数据在正确的设备上
            pixel_values_batch = pixel_values_batch.to(accelerator.device)
            pixel_values_batch = pixel_values_batch.to(memory_format=torch.contiguous_format).float()
            indices = batch["idx"]
            
            # 编码为 latent
            latents = vae.encode(pixel_values_batch).latent_dist.sample()
            latents = latents * vae.config.scaling_factor 
            
            # 收集每个样本的 latent、pixel_values 和 label（移到 CPU 并转换为 numpy）
            for i, idx in enumerate(indices):
                idx = idx.item()
                all_latents[idx] = latents[i].cpu().numpy()
                # 将 CHW (3x256x256) 转换为 HWC (256x256x3)，并将值范围从 [-1,1] 转换回 [0,1]
                img_tensor = pixel_values_batch[i].cpu().permute(1, 2, 0)  # CHW -> HWC
                img_tensor = (img_tensor + 1) / 2  # [-1,1] -> [0,1]
                img_tensor = torch.clamp(img_tensor, 0, 1)  # 确保值在 [0,1] 范围内
                img_array = (img_tensor.numpy() * 255).astype('uint8')  # 转换为 uint8 [0,255]
                all_images[idx] = PIL.Image.fromarray(img_array)
                all_labels[idx] = labels_batch[i].item() if torch.is_tensor(labels_batch[i]) else labels_batch[i]
            
            if accelerator.is_main_process:
                progress_bar.update(1)
    
    if accelerator.is_main_process:
        progress_bar.close()
    
    # 同步所有进程
    accelerator.wait_for_everyone()
    
    # 收集所有进程的结果到主进程
    from accelerate.utils import gather_object
    all_gathered_latents = gather_object([all_latents])
    all_gathered_images = gather_object([all_images])
    all_gathered_labels = gather_object([all_labels])
    
    # 在主进程上合并所有数据
    if accelerator.is_main_process:
        # 合并所有进程收集的数据
        merged_latents = {}
        merged_images = {}
        merged_labels = {}
        
        for proc_latents in tqdm(all_gathered_latents, desc="合并 latent"):
            if proc_latents:  # 确保不是 None
                merged_latents.update(proc_latents)
        
        for proc_images in tqdm(all_gathered_images, desc="合并 images"):
            if proc_images:  # 确保不是 None
                merged_images.update(proc_images)
        
        for proc_labels in tqdm(all_gathered_labels, desc="合并 labels"):
            if proc_labels:  # 确保不是 None
                merged_labels.update(proc_labels)
        
        # 使用生成器在线构建数据集，避免一次性加载所有数据到内存
        # 优化：移除 .tolist()（numpy数组直接支持，避免慢速转换）
        progress_bar = tqdm(total=len(full_dataset), desc="构建数据集")
        def data_generator():
            for i in range(len(full_dataset)):
                yield {
                    "latents": merged_latents[i],  # 直接使用numpy数组，不需要tolist()，datasets库原生支持
                    "images": merged_images[i],     # PIL Image对象，datasets库支持
                    "label": merged_labels[i]
                }
                progress_bar.update(1)

        progress_bar.close()
        
        # 获取latents的shape以确定Array3D的维度
        sample_latent = merged_latents[0]
        latent_shape = sample_latent.shape  # 应该是 (channels, height, width)
        
        # 创建Features对象，明确指定label列为ClassLabel类型
        # 使用full_dataset中的ClassLabel特征来保持一致性
        features = Features({
            "latents": Array3D(shape=latent_shape, dtype="float32"),
            "images": Image(),
            "label": full_dataset.features[class_column]  # 直接使用full_dataset的ClassLabel特征
        })
        
        # 使用单进程构建，避免多进程开销和同步问题
        # 移除 keep_in_memory=True 以减少内存压力，让系统自动管理内存
        dataset = Dataset.from_generator(
            data_generator, 
            features=features,  # 指定features以确保label列是ClassLabel类型
            num_proc=args.dataloader_num_workers,  # 使用单进程避免多进程同步开销和卡顿
            keep_in_memory=False  # 不全部加载到内存，减少内存压力和GC停顿
        )
    
    # step5: 创建数据集
        dataset_dict = DatasetDict({
            'sample_level': dataset,  # n_samples rows
            'class_level_hyp': class_level_dataset_hyp,        # n_classes rows
            'class_level_sph': class_level_dataset_sph,        # n_classes rows
            'class_level_oh': class_level_dataset_oh        # n_classes rows
        })

        dataset_dict.save_to_disk(os.path.join(args.output_dir, args.dataset_name + "_latents"))
        # load dataset
        loaded_dataset = load_from_disk(os.path.join(args.output_dir, args.dataset_name + "_latents"))

        print(loaded_dataset)