import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import random
import numpy as np
from tqdm import tqdm
from hier_util import HierUtil
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from accelerate import Accelerator
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser(description="Generate interpolated images between two classes")
    parser.add_argument("--model_path", type=str, default="/projects/0/prjs0774/flux.1/output/finetune/lora/runwayml/stable-diffusion-v1-5/local_datasets/keremberke/pokemon-classification_latents/checkpoint-6000-oh", 
                        help="Path to the model checkpoint")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='runwayml/stable-diffusion-v1-5', help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--dataset_name", type=str, default="./local_datasets/keremberke/pokemon-classification_latents",
                        help="Dataset name to determine which class set to use")
    parser.add_argument("--num_validation_images", type=int, default=8, 
                        help="Number of interpolation steps to generate")
    parser.add_argument("--guidance_scale", type=float, default=4,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--emb_type", type=str, default='hyp',
                        help="Embedding type to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="interpolation_results",
                        help="Directory to save the generated images")
    return parser.parse_args()


import torch

def mobius_add(x, y, c=1.0):
    xy = (2 * c * (x * y).sum(dim=-1, keepdim=True))
    x2 = c * (x * x).sum(dim=-1, keepdim=True)
    y2 = c * (y * y).sum(dim=-1, keepdim=True)
    denominator = 1 + xy + y2 * x2
    return ((1 + xy + y2) * x + (1 - x2) * y) / denominator.clamp_min(1e-5)

def mobius_scalar_mul(r, x, c=1.0):
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    tanh_rcx = torch.tanh(r * torch.atanh(torch.sqrt(c) * norm_x))
    return tanh_rcx * x / (norm_x * torch.sqrt(c))

def geodesic_linspace(x, y, n_points, c=0.1):
    t = torch.linspace(0, 1, n_points, device=x.device, dtype=x.dtype).unsqueeze(-1)
    neg_x = -x
    diff = mobius_add(neg_x, y, c)
    points = mobius_add(x, mobius_scalar_mul(t, diff, c), c)
    return points


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Initialize accelerator for loading state
    accelerator = Accelerator()
    accelerator.load_state(args.model_path)
    pipeline.unet = accelerator.unwrap_model(pipeline.unet)
    
    # Get class set based on dataset name
    dataset_mapping = {
        "./local_datasets/keremberke/pokemon-classification_latents": HierUtil.get_selected_pokemon,
        "./local_datasets/Donghyun99/CUB-200-2011_latents": HierUtil.get_selected_birds,
        "./local_datasets/Donghyun99/Stanford-Cars_latents": HierUtil.get_selected_cars,
    }
    
    get_class_set_fn = dataset_mapping.get(args.dataset_name)
    if not get_class_set_fn:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    
    selected_class_set = get_class_set_fn()
    
    # Choose two random classes
    class1, class2 = random.sample(selected_class_set, 2)
    print(f"Interpolating between: {class1} and {class2}")
    
    # Load class embeddings
    base_dir = '/projects/0/prjs0774/flux.1/output/finetune/lora/runwayml/stable-diffusion-v1-5/local_datasets/'
    dataset_path = Path(base_dir) / args.dataset_name / "checkpoint-6000-oh"
    
    # Load the dataset to get class embeddings
    dataset = load_from_disk(args.dataset_name)
    
    # Get class indices
    # class_set_plain = dataset['sample_level'].features['label'].names
    class_set_embs = dataset['class_level_oh']['objects']
    if args.emb_type == 'oh':
        class_embeddings = torch.tensor(dataset['class_level_oh']['embeddings'])
    elif args.emb_type == 'sph':
        class_embeddings = torch.tensor(dataset['class_level_sph']['embeddings'])
    elif args.emb_type == 'hyp':
        class_embeddings = torch.tensor(dataset['class_level_hyp']['embeddings'])

    # Handle special case for Stanford-Cars dataset
    if args.dataset_name == "./local_datasets/Donghyun99/Stanford-Cars_latents":
        class1_idx = class_set_embs.index(class1.replace("_", " "))
        class2_idx = class_set_embs.index(class2.replace("_", " "))
    else:
        class1_idx = class_set_embs.index(class1)
        class2_idx = class_set_embs.index(class2)
    
    # Get the embeddings for the two classes
    class1_embedding = class_embeddings[class1_idx]
    class2_embedding = class_embeddings[class2_idx]
    
    # Generate interpolated images
    images = []
    # Create interpolation weights in a vectorized way
    alphas = torch.linspace(0, 1, args.num_validation_images)
    
    # Vectorized interpolation between the two class embeddings
    interpolated_embeddings = (1 - alphas[:, None]) * class1_embedding + alphas[:, None] * class2_embedding
    
    # Add sequence length dimension and move to cuda
    cond_embeddings = interpolated_embeddings.unsqueeze(1).to("cuda")
    negative_prompt_embeds = torch.zeros_like(cond_embeddings)
        
    # Generate image
    with torch.no_grad():
        # import pdb
        # pdb.set_trace()
        latents = pipeline(
            guidance_scale=args.guidance_scale,
            prompt_embeds=cond_embeddings,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=50,
            output_type="latent",
        )[0]
        
        # Decode the latents using VAE
        images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    
    # Create output directory if it doesn't exist
    output_dir = Path(f"interpolation_results/{args.dataset_name}/{class1}_to_{class2}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert images to a grid using make_grid
    # First ensure all images are in the right format for make_grid (B, C, H, W)
    image_tensors = []
    for img in images:
        # Normalize from [-1, 1] to [0, 1] range
        img_tensor = (img + 1) / 2
        img_tensor = torch.clamp(img_tensor, 0, 1)
        image_tensors.append(img_tensor)
    
    # Stack all images into a batch
    image_batch = torch.stack(image_tensors)
    
    # Create a grid of images
    grid = make_grid(image_batch, nrow=len(image_tensors), padding=2).float()
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Create a figure to display the grid
    # Create a figure to display the grid in one row
    fig, ax = plt.subplots(figsize=(len(image_tensors) * 3, 3))
    
    # Display the entire grid as one image
    ax.imshow(grid_np)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"interpolation_row_{args.emb_type}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated {len(images)} interpolated images between {class1} and {class2}")
    print(f"Images saved to {output_dir}")

if __name__ == "__main__":
    main()





