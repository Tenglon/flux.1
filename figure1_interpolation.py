import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, concatenate_datasets, load_from_disk
from packaging import version

from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr, cast_training_params, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from utils import ckpt_limit, enable_xformers, log_validation, tokenize_captions2
from hier_util import HierUtil
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random
import numpy as np

def registor_new_accelerate(args, accelerator, ema_model):
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


def set_mixed_precision(args, accelerator):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    return weight_dtype


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
    parser.add_argument("--sample_steps", type=int, default=250,
                        help="Number of sampling steps")
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear",
                        help="Beta schedule for DDPM")
    parser.add_argument("--prediction_type", type=str, default="epsilon",
                        help="Prediction type for DDPM")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate for the optimizer.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_train_steps", type=int, default=20000, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use Exponential Moving Average for the final model weights.")
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"], help="Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai) for experiment tracking and logging of model metrics and model checkpoints")

    return parser.parse_args()


def mobius_add(x, y, c=1.0):
    xy = (2 * c * (x * y).sum(dim=-1, keepdim=True))
    x2 = c * (x * x).sum(dim=-1, keepdim=True)
    y2 = c * (y * y).sum(dim=-1, keepdim=True)
    denominator = 1 + xy + y2 * x2
    return ((1 + xy + y2) * x + (1 - x2) * y) / denominator.clamp_min(1e-5)

def mobius_scalar_mul(r, x, c=1.0):
    c = torch.tensor(c, device=x.device, dtype=x.dtype)
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-5)
    tanh_rcx = torch.tanh(r * torch.atanh(torch.sqrt(c) * norm_x))
    return tanh_rcx * x / (norm_x * torch.sqrt(c))

def geodesic_linspace(x, y, n_points, c=0.1):
    c = torch.tensor(c, device=x.device, dtype=x.dtype)
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

    if args.dataset_name == "./local_datasets/Donghyun99/CUB-200-2011_latents":
        class_set_embs = [cls.replace(cls[:4], "") if cls[:3].isdigit() and cls[3] == "." else cls for cls in class_set_embs]

    
    # Load model
    unet = UNet2DConditionModel(
        sample_size=64,  # args.resolution_latent from your training
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        only_cross_attention=False,
        cross_attention_dim=class_embeddings.shape[1],  # Adjust to match your class_embeddings.shape[1]
    )

    ema_model = EMAModel(
            unet.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
        )
    

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model 
    

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
    )
    # `accelerate` 0.16.0 will have better support for customized saving
    registor_new_accelerate(args, accelerator, ema_model)

    weight_dtype = set_mixed_precision(args, accelerator)
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    ema_model.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare the model with accelerator
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.sample_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )

    # Load the saved state
    accelerator.load_state(args.model_path)
    # Unwrap the model
    unwrapped_unet = accelerator.unwrap_model(unet)
    
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
    class1 = 'AM_General_Hummer_SUV_2000'
    # class1 = 'BMW_X3_SUV_2012'
    class2 = 'BMW_X3_SUV_2012'
    # class2 = 'Jeep_Patriot_SUV_2012'
    # class1 = 'American Goldfinch'
    # class2 = 'House Sparrow'
    # Rose_breasted_Grosbeak and White_crowned_Sparrow
    # class1 = 'Rose_breasted_Grosbeak'
    # class2 = 'White_crowned_Sparrow'
    print(f"Interpolating between: {class1} and {class2}")

    # Handle special case for Stanford-Cars dataset
    if args.dataset_name == "./local_datasets/Donghyun99/Stanford-Cars_latents":
        class1_idx = class_set_embs.index(class1.replace(" ", " "))
        class2_idx = class_set_embs.index(class2.replace(" ", " "))
    else:
        class1_idx = class_set_embs.index(class1)
        class2_idx = class_set_embs.index(class2)
    
    # Get the embeddings for the two classes
    class1_embedding = class_embeddings[class1_idx]
    class2_embedding = class_embeddings[class2_idx]
    
    # Generate interpolated images
    images = []
    # Create interpolation weights in a vectorized way
    if args.emb_type == 'oh':
        alphas = torch.linspace(0, 1, args.num_validation_images)
        # Vectorized interpolation between the two class embeddings
        interpolated_embeddings = (1 - alphas[:, None]) * class1_embedding + alphas[:, None] * class2_embedding
    elif args.emb_type == 'hyp':
        interpolated_embeddings = geodesic_linspace(class1_embedding, class2_embedding, args.num_validation_images, c=0.1)
    
    # Add sequence length dimension and move to cuda
    cond_embeddings = interpolated_embeddings.unsqueeze(1).to("cuda")
    negative_prompt_embeds = torch.zeros_like(cond_embeddings)
        
    # Generate image
    with torch.no_grad():

        generator = torch.Generator(device="cuda")
        if args.seed is not None:
            generator = generator.manual_seed(args.seed)

        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            unet=unet,
            safety_checker=None
        ).to("cuda")
        latents = pipeline(
            prompt = None, 
            guidance_scale=args.guidance_scale,
            num_inference_steps=250,
            prompt_embeds=cond_embeddings,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            output_type="latent"
        )[0]

        # import pdb
        # pdb.set_trace()
        
        # Decode the latents using VAE
        images = pipeline.vae.decode(latents.to(torch.bfloat16) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    
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

    # remove file if exists
    save_path = output_dir / f"interpolation_row_{args.emb_type}.png"
    if os.path.exists(save_path):
        os.remove(save_path)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated {len(images)} interpolated images between {class1} and {class2}")
    print(f"Images saved to {output_dir}")

if __name__ == "__main__":
    main()





