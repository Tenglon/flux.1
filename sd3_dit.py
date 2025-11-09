"""Simplified DiT training script tailored for SD3 checkpoints.

This module mirrors the command line interface of ``teng_dit.py`` so that
existing automation continues to work, while trimming down the implementation
to the essentials required to finetune the SD3 DiT transformer.
"""

import argparse
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image


logger = get_logger(__name__, log_level="INFO")


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
    parser.add_argument("--prediction_type", type=str, default="epsilon")
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


def _load_dataset(args):
    """Load a huggingface dataset or local folder."""

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
        )

    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split")

    return dataset["train"]


def _image_transforms(args):
    interpolation = transforms.InterpolationMode.BILINEAR
    crop = transforms.CenterCrop if args.center_crop else transforms.RandomCrop

    transform_list = [transforms.Resize(args.resolution, interpolation=interpolation)]
    if args.center_crop:
        transform_list.append(crop(args.resolution))
    if args.random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.extend([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    return transforms.Compose(transform_list)


def _prepare_dataset(raw_dataset, args):
    """Convert raw dataset samples into tensors expected by the model."""

    image_column = "image"
    if image_column not in raw_dataset.column_names:
        for name in raw_dataset.column_names:
            sample = raw_dataset[0][name]
            if isinstance(sample, Image.Image):
                image_column = name
                break

    caption_column = "text"
    if caption_column not in raw_dataset.column_names:
        fallbacks = ["caption", "captions", "prompt"]
        for name in fallbacks:
            if name in raw_dataset.column_names:
                caption_column = name
                break
        else:
            for name in raw_dataset.column_names:
                sample = raw_dataset[0][name]
                if isinstance(sample, str):
                    caption_column = name
                    break

    transform = _image_transforms(args)

    def preprocess(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [transform(image) for image in images]

        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, tuple)):
                captions.append(caption[0])
            else:
                captions.append(str(caption))
        examples["captions"] = captions
        return examples

    with_transform = raw_dataset.with_transform(preprocess)
    with_transform.set_format(type="torch", columns=["pixel_values"], output_all_columns=True)
    return with_transform


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

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )
    transformer = pipeline.transformer

    if args.transformer_model_name_or_path:
        transformer = type(transformer).from_pretrained(args.transformer_model_name_or_path)

    vae = pipeline.vae
    noise_scheduler = pipeline.scheduler

    # Freeze components that are not trained
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipeline, name, None)
        if encoder is not None:
            encoder.requires_grad_(False)
    vae.requires_grad_(False)

    raw_train_dataset = _load_dataset(args)
    if args.max_train_samples is not None:
        raw_train_dataset = raw_train_dataset.select(
            range(min(len(raw_train_dataset), args.max_train_samples))
        )
    train_dataset = _prepare_dataset(raw_train_dataset, args)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        captions = [example["captions"] for example in examples]
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "captions": captions}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
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

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    latents_scale = getattr(vae.config, "scaling_factor", math.sqrt(2.0))

    if args.use_ema:
        ema_model = EMAModel(transformer.parameters(), inv_gamma=args.ema_inv_gamma, power=args.ema_power)
    else:
        ema_model = None

    global_step = 0
    transformer.train()

    for batch in train_dataloader:
        with accelerator.accumulate(transformer):
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype, device=accelerator.device)
            latents = vae.encode(pixel_values).latent_dist.sample() * latents_scale

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                prompt_embeds, pooled_embeds, _ = pipeline.encode_prompt(
                    batch["captions"],
                    device=accelerator.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

            model_output = transformer(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"pooled_projections": pooled_embeds},
            ).sample

            target = noise if args.prediction_type == "epsilon" else latents
            loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if ema_model is not None:
                ema_model.step(transformer.parameters())

        global_step += 1
        accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

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

