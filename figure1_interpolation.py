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

def parse_args():
    parser = argparse.ArgumentParser(description="Generate interpolated images between two classes")
    parser.add_argument("--model_path", type=str, default="/projects/0/prjs0774/flux.1/output/finetune/lora/runwayml/stable-diffusion-v1-5/local_datasets/keremberke/pokemon-classification_latents/checkpoint-20000", 
                        help="Path to the model checkpoint")
    parser.add_argument("--dataset_name", type=str, default="keremberke/pokemon-classification_latents",
                        help="Dataset name to determine which class set to use")
    parser.add_argument("--num_validation_images", type=int, default=8, 
                        help="Number of interpolation steps to generate")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="interpolation_results",
                        help="Directory to save the generated images")
    return parser.parse_args()

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
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    import pdb
    pdb.set_trace()
    
    # Get class set based on dataset name
    dataset_mapping = {
        "keremberke/pokemon-classification_latents": HierUtil.get_selected_pokemon,
        "Donghyun99/CUB-200-2011_latents": HierUtil.get_selected_birds,
        "Donghyun99/Stanford-Cars_latents": HierUtil.get_selected_cars,
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
    dataset_path = Path(base_dir) / args.dataset_name / "checkpoint-20000"
    
    # Load the dataset to get class embeddings
    from datasets import load_from_disk
    dataset = load_from_disk(str(Path(base_dir) / args.dataset_name.split("/")[-1] + "_latents"))
    
    # Get class indices
    class_set_plain = dataset['sample_level'].features['label'].names
    
    # Handle special case for Stanford-Cars dataset
    if args.dataset_name == "Donghyun99/Stanford-Cars_latents":
        class1_idx = class_set_plain.index(class1.replace("_", " "))
        class2_idx = class_set_plain.index(class2.replace("_", " "))
    else:
        class1_idx = class_set_plain.index(class1)
        class2_idx = class_set_plain.index(class2)
    
    # Get embeddings
    class_embeddings = torch.tensor(dataset['class_level_oh']['embeddings'])
    
    # Get the embeddings for the two classes
    class1_embedding = class_embeddings[class1_idx]
    class2_embedding = class_embeddings[class2_idx]
    
    # Generate interpolated images
    images = []
    for i in tqdm(range(args.num_validation_images)):
        # Calculate interpolation weight
        alpha = i / (args.num_validation_images - 1) if args.num_validation_images > 1 else 0
        
        # Interpolate between the two class embeddings
        interpolated_embedding = (1 - alpha) * class1_embedding + alpha * class2_embedding
        
        # Add batch dimension and sequence length dimension
        cond_embeddings = interpolated_embedding.unsqueeze(0).unsqueeze(0).to("cuda")
        
        # Generate image
        with torch.no_grad():
            latents = pipeline(
                guidance_scale=args.guidance_scale,
                encoder_hidden_states=cond_embeddings,
                num_inference_steps=50,
                output_type="latent",
            ).images[0]
            
            # Decode the latents using VAE
            latents = latents.unsqueeze(0)  # Add batch dimension
            images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0][0]

    
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
    grid = make_grid(image_batch, nrow=len(image_tensors), padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Create a figure to display the grid
    # Create a figure to display the grid in one row
    fig, ax = plt.subplots(figsize=(len(image_tensors) * 3, 3))
    
    # Display the entire grid as one image
    ax.imshow(grid_np)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "interpolation_row.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated {len(images)} interpolated images between {class1} and {class2}")
    print(f"Images saved to {output_dir}")

if __name__ == "__main__":
    main()





