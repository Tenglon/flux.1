"""Simplified DiT training script using DiTPipeline and DiT-S/2 transformer.

This module mirrors the command line interface of ``teng_dit.py`` so that
existing automation continues to work, while trimming down the implementation
to the essentials required to finetune the DiT transformer.
"""

import argparse
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_from_disk
from diffusers import DiTPipeline, DiTTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.nn import functional as F
from PIL import Image
from tqdm.auto import tqdm
from hier_util import HierUtil


logger = get_logger(__name__, log_level="INFO")


def build_dit_s_2_transformer(sample_size):
    """Create a DiT-S/2 transformer (patch_size=2, S-sized model)."""
    dit_s_2_config = dict(
        sample_size=sample_size,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_attention_heads=6,
        attention_head_dim=64,
        num_layers=12,
        attention_bias=True,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
    )
    return DiTTransformer2DModel(**dit_s_2_config)


def parse_args():
    """Parse command line arguments.

    The signature matches ``teng_dit.py`` so that external callers can switch
    between the two scripts without any adjustments.
    """

    parser = argparse.ArgumentParser(description="Minimal SD3 DiT fine-tuning script")
    parser.add_argument("--dataset_name", type=str, default=None, required=True)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=None,
        help="Optional initialization weights for the DiT transformer.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="dit-output")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--resolution_latent", type=int, default=64)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=None, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=0.75)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="wandb")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--guidance_dropout_prob", type=float, default=0.1)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--emb_type", type=str, default="oh")
    parser.add_argument("--validation_epochs", type=int, default=10)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Either --dataset_name or --train_data_dir must be provided.")

    return args


# Prompt templates for different datasets
PROMPT_MAPPING = {
    "./local_datasets/keremberke/pokemon-classification_latents": "A photo of a {}.",
    "./local_datasets/Donghyun99/CUB-200-2011_latents": "A photo of a {}.",
    "./local_datasets/Donghyun99/Stanford-Cars_latents": "A photo of a {}.",
}

selected_hier_mapping = {
    "./local_datasets/keremberke/pokemon-classification_latents": HierUtil.get_selected_pokemon(),
    "./local_datasets/Donghyun99/CUB-200-2011_latents": HierUtil.get_selected_birds(),
    "./local_datasets/Donghyun99/Stanford-Cars_latents": HierUtil.get_selected_cars(),
}


def _load_dataset(args):
    """Load dataset from disk, consistent with teng_dit.py."""
    dataset = load_from_disk(args.dataset_name)
    return dataset


def _prepare_dataset(dataset, args):
    """Prepare dataset consistent with teng_dit.py."""
    image_column = 'images'
    class_column = 'label'
    latents_column = 'latents'

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["class_labels"] = examples[class_column]
        examples["latents"] = examples[latents_column]
        return examples

    train_dataset = dataset['sample_level'].with_transform(preprocess_train)
    return train_dataset


def _is_wandb_enabled(accelerator):
    trackers = getattr(accelerator, "trackers", [])
    for tracker in trackers:
        name = getattr(tracker, "name", "").lower()
        if "wandb" in name:
            return True
        if tracker.__class__.__name__.lower().startswith("wandb"):
            return True
    return False


def _prepare_wandb_images(image_batch, captions):
    """Convert a batch of image tensors into wandb.Image objects with captions."""
    import wandb  # Imported lazily to avoid dependency when not logging.
    
    if image_batch is None or len(image_batch) == 0:
        return []

    images = (image_batch / 2 + 0.5).clamp(0, 1).cpu()

    wandb_images = []
    for img_tensor, caption in zip(images, captions):
        if img_tensor.shape[0] == 3:
            img_np = img_tensor.permute(1, 2, 0).numpy()
        else:
            img_np = img_tensor.numpy()
        wandb_images.append(wandb.Image(img_np, caption=caption))

    return wandb_images


def _tensor_batch_to_wandb_images(tensor_batch, captions):
    import wandb  # Imported lazily to avoid dependency when not logging.

    if tensor_batch is None:
        return []

    images = tensor_batch.detach().cpu()
    images = (images / 2 + 0.5).clamp(0, 1)

    wandb_images = []
    for image, caption in zip(images, captions):
        wandb_images.append(wandb.Image(to_pil_image(image), caption=caption))

    return wandb_images


def _pil_images_to_tensor_batch(pil_images):
    tensor_images = []
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    for image in pil_images:
        tensor_images.append(normalize(image.convert("RGB")))
    if not tensor_images:
        return None
    return torch.stack(tensor_images)


def _run_wandb_validation(
    accelerator,
    args,
    pipeline,
    transformer,
    vae,
    latents_scale,
    train_dataset,
    ema_model,
    weight_dtype,
    global_step,
    class_set_plain,
):
    if not accelerator.is_main_process or not _is_wandb_enabled(accelerator):
        return

    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb is not installed; skipping validation logging.")
        return

    num_images = min(args.num_validation_images, len(train_dataset))
    if num_images == 0:
        return

    samples = []
    for idx in range(num_images):
        samples.append(train_dataset[idx])

    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    class_labels = torch.tensor([sample["class_labels"] for sample in samples])
    class_names = [class_set_plain[label] for label in class_labels]

    pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    if ema_model is not None:
        ema_model.store(unwrapped_transformer.parameters())
        ema_model.copy_to(unwrapped_transformer.parameters())

    pipeline.transformer = unwrapped_transformer
    pipeline.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    was_training = unwrapped_transformer.training
    unwrapped_transformer.eval()

    # Add scale_model_input method to FlowMatchEulerDiscreteScheduler if it doesn't exist
    # Flow matching schedulers don't need to scale the input, so we just return it as-is
    # IMPORTANT: In Flow Matching, model output is velocity field v, which must be used with scheduler.step(v, t, x_t)
    # Do NOT treat the output as epsilon or x0, and do NOT implement any custom scale_model_input logic
    if isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
        if not hasattr(pipeline.scheduler, 'scale_model_input'):
            def scale_model_input(sample: torch.Tensor, timestep) -> torch.Tensor:
                """Compatibility method for FlowMatchEulerDiscreteScheduler - returns input unchanged.
                
                Flow Matching does not require input scaling. The model output (velocity field v)
                must be fed directly to scheduler.step(v, t, x_t) without any transformation.
                """
                return sample
            pipeline.scheduler.scale_model_input = scale_model_input

    try:
        # DiTPipeline uses class_labels (integers), not prompts
        class_label_ids = class_labels.tolist()

        with torch.no_grad():
            # VAE encoding: input is [-1, 1], multiply by scaling_factor
            original_latents = vae.encode(pixel_values).latent_dist.sample() * latents_scale
            
            # VAE decoding: divide by scaling_factor before decode
            original_latents_restored = vae.decode(
                original_latents / latents_scale,
                return_dict=False,
                generator=None,
            )[0]

        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + global_step)

        with torch.no_grad():
            # DiTPipeline directly output images rather than latents
            generated_images, generated_latents = pipeline(class_labels = class_label_ids, 
                                guidance_scale=args.guidance_scale, 
                                generator=generator, 
                                num_inference_steps=args.sample_steps,
                                output_type="numpy")
        generated_images = torch.from_numpy(generated_images[0])

        logs = {
            "validation/original_image":  _prepare_wandb_images(pixel_values.cpu().detach(), class_names),
            "validation/original_latents_restored_image": _prepare_wandb_images(original_latents_restored.cpu().detach(), class_names),
            "validation/generated_image": _prepare_wandb_images(generated_images, class_names),
            "validation/original_latents_histogram": wandb.Histogram(original_latents.detach().float().cpu().numpy().flatten()),
            "validation/generated_latents_histogram": wandb.Histogram(generated_latents.detach().float().cpu().numpy().flatten()),
        }

        accelerator.log(logs, step=global_step)
    finally:
        if ema_model is not None:
            ema_model.restore(unwrapped_transformer.parameters())

        if was_training:
            unwrapped_transformer.train()


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        mixed_precision=args.mixed_precision,
        log_with=args.logger if args.logger != "none" else None,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE from pretrained path
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )

    # Initialize scheduler with minimal settings for validation
    # use_dynamic_shifting=False and shift=0 for minimal configuration
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=args.sample_steps,
        shift=1.0,  # Minimal setting: shift=0
        use_dynamic_shifting=False,  # Disable dynamic shifting
    )

    # Create DiT-S/2 transformer
    transformer_sample_size = args.resolution_latent
    logger.info(
        "Initializing a DiT-S/2 sized transformer from scratch (sample_size=%s, patch_size=2).",
        transformer_sample_size,
    )
    transformer = build_dit_s_2_transformer(sample_size=transformer_sample_size)

    if args.transformer_model_name_or_path:
        logger.info(f"Loading transformer from {args.transformer_model_name_or_path}")
        transformer = DiTTransformer2DModel.from_pretrained(
            args.transformer_model_name_or_path,
            cache_dir=args.cache_dir,
        )

    # Create DiTPipeline for validation
    pipeline = DiTPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=noise_scheduler,
    )

    # Freeze VAE
    vae.requires_grad_(False)

    # Load dataset consistent with teng_dit.py
    dataset = _load_dataset(args)
    image_column = 'images'
    class_column = 'label'
    latents_column = 'latents'

    # Get prompt template
    PROMPT_TEMPLATE = PROMPT_MAPPING.get(args.dataset_name, "A photo of a {}.")
    
    # Get class set for embeddings
    class_set_emb = dataset['class_level_hyp']['objects']
    if args.dataset_name == "./local_datasets/Donghyun99/CUB-200-2011_latents":
        class_set_emb = [cls.replace(cls[:4], "") if cls[:3].isdigit() and cls[3] == "." else cls for cls in class_set_emb]

    class_set_plain = dataset['sample_level'].features[class_column].names
    selected_class_set = selected_hier_mapping.get(args.dataset_name, [])
    
    if args.dataset_name == "./local_datasets/Donghyun99/Stanford-Cars_latents":
        selected_class_labels = [class_set_plain.index(cls.replace("_", " ")) for cls in selected_class_set]
    else:
        selected_class_labels = [class_set_plain.index(cls) for cls in selected_class_set] if selected_class_set else None

    # Prepare dataset
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['sample_level'] = dataset['sample_level'].shuffle(seed=args.seed)
            if selected_class_labels is not None:
                idx = torch.tensor([(label in selected_class_labels) for label in dataset['sample_level']['label']])
                idx = torch.where(idx)[0]
                if len(idx) < args.max_train_samples:
                    repeat_factor = args.max_train_samples // len(idx) + 1
                    idx = idx.repeat(repeat_factor)[:args.max_train_samples]
                dataset['sample_level'] = dataset['sample_level'].select(idx)
            else:
                dataset['sample_level'] = dataset['sample_level'].select(range(min(len(dataset['sample_level']), args.max_train_samples)))
    
    train_dataset = _prepare_dataset(dataset, args)
    
    logger.info(f"Dataset size: {train_dataset.num_rows}")
    train_labels = [class_set_plain[label] for label in dataset['sample_level']['label']]
    unique_train_labels = list(set(train_labels))
    logger.info(f"Label of the samples: {unique_train_labels}")

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        latents_list = [torch.tensor(example['latents']) for example in examples]
        latents = torch.stack(latents_list)
        class_labels = torch.tensor([example["class_labels"] for example in examples])
        
        return {"pixel_values": pixel_values, "class_labels": class_labels, "latents": latents}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
        prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    pipeline.to(accelerator.device)
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    latents_scale = getattr(vae.config, "scaling_factor", math.sqrt(2.0))

    if args.use_ema:
        ema_model = EMAModel(transformer.parameters(), inv_gamma=args.ema_inv_gamma, power=args.ema_power)
    else:
        ema_model = None

    # Calculate number of epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_dataset.num_rows}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Initialize wandb trackers
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    global_step = 0
    transformer.train()

    validation_interval = max(args.validation_epochs, 0)

    # Create progress bar
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_main_process, mininterval=1)

    # Train by epochs
    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Use precomputed latents from dataset (already scaled)
                latents = batch["latents"].to(dtype=weight_dtype, device=accelerator.device, non_blocking=True)

                noise = torch.randn_like(latents)
                # Sample timesteps from scheduler's timesteps array
                scheduler_timesteps = noise_scheduler.timesteps.to(device=latents.device)
                indices = torch.randint(0, len(scheduler_timesteps), (latents.shape[0],), device=latents.device)
                timesteps = scheduler_timesteps[indices]
                # Flow Matching: x_t = σ * x_0 + (1 - σ) * x_1
                # where x_0 is noise and x_1 is latents (real data)
                noisy_latents = noise_scheduler.scale_noise(latents, timesteps, noise)

                # Get class labels
                class_labels = batch["class_labels"].to(device=latents.device)
                # Apply guidance dropout
                if torch.rand(1).item() < args.guidance_dropout_prob:
                    class_labels = torch.zeros_like(class_labels)

                # DiT transformer uses class_labels directly, not text embeddings
                # In Flow Matching, model output is the velocity field v
                model_output = transformer(
                    noisy_latents,
                    timestep=timesteps,
                    class_labels=class_labels,
                ).sample

                # For Flow Matching: target is the velocity field v = x_1 - x_0
                # where x_1 is latents (real data) and x_0 is noise
                # IMPORTANT: The model output v must be used with scheduler.step(v, t, x_t) during inference
                # Do NOT treat v as epsilon or x0, and do NOT apply any diffusion formulas manually
                if isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
                    target = latents - noise  # velocity field: v = x_1 - x_0
                else:
                    raise ValueError(f"Unsupported noise scheduler type: {type(noise_scheduler)}")
                loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if ema_model is not None:
                    ema_model.step(transformer.parameters())

            if accelerator.sync_gradients:
                train_loss += loss.detach().item()
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if validation_interval and global_step % validation_interval == 0:
                    _run_wandb_validation(
                        accelerator,
                        args,
                        pipeline,
                        transformer,
                        vae,
                        latents_scale,
                        train_dataset,
                        ema_model,
                        weight_dtype,
                        global_step,
                        class_set_plain,
                    )

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_path = Path(args.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if ema_model is not None:
            ema_model.store(transformer.parameters())
            ema_model.copy_to(transformer.parameters())
        transformer.save_pretrained(save_path / "transformer")
        if ema_model is not None:
            ema_model.restore(transformer.parameters())


if __name__ == "__main__":
    main()

