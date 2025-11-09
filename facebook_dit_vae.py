import argparse
import logging
import math
import os
from datetime import timedelta
from pathlib import Path

import datasets
import diffusers
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_from_disk
from diffusers import (
    AutoencoderKL,
    DiTPipeline,
    FlowMatchEulerDiscreteScheduler,
    DiTTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo
from torchvision import transforms
from tqdm.auto import tqdm

import wandb
from hier_util import HierUtil
from utils import ckpt_limit, enable_xformers, log_validation


logger = get_logger(__name__, log_level="INFO")


SELECTED_CLASSES = {
    "./local_datasets/keremberke/pokemon-classification_latents": HierUtil.get_selected_pokemon(),
    "./local_datasets/Donghyun99/CUB-200-2011_latents": HierUtil.get_selected_birds(),
    "./local_datasets/Donghyun99/Stanford-Cars_latents": HierUtil.get_selected_cars(),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DiT model that relies on the Facebook VAE.")
    parser.add_argument("--dataset_name", type=str, default=None, required=True)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument(
        "--dit_components_model_name_or_path",
        type=str,
        default="facebook/DiT-XL-2-512",
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default="facebook/DiT-B-2-256x256",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--resolution_latent", type=int, default=64)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=None, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample"])
    parser.add_argument("--sample_steps", type=int, default=1000)
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
    parser.add_argument("--num_validation_images", type=int, default=8)

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name or a training directory.")

    return args


def compute_snr_flow_matching(noise_scheduler, timesteps):
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=timesteps.dtype)
    if hasattr(noise_scheduler, "timesteps") and noise_scheduler.timesteps is not None:
        schedule_timesteps = noise_scheduler.timesteps.to(device=timesteps.device)
        indices = torch.bucketize(timesteps, schedule_timesteps, right=True).clamp(max=len(sigmas) - 1)
    else:
        num_sigmas = len(sigmas)
        indices = (timesteps / noise_scheduler.config.num_train_timesteps * (num_sigmas - 1)).long()
        indices = indices.clamp(0, num_sigmas - 1)
    sigma = sigmas[indices].float().clamp_(1e-8, 1 - 1e-8)
    snr = ((1.0 - sigma) / sigma) ** 2
    return snr


def compute_snr_compatible(noise_scheduler, timesteps):
    if isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
        return compute_snr_flow_matching(noise_scheduler, timesteps)
    return compute_snr(noise_scheduler, timesteps)


def build_dit_b_2_transformer(sample_size: int) -> DiTTransformer2DModel:
    config = dict(
        sample_size=sample_size,
        patch_size=4,
        in_channels=16,
        out_channels=16,
        num_attention_heads=6,
        attention_head_dim=64,
        num_layers=12,
        attention_bias=True,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
    )
    return DiTTransformer2DModel(**config)


def prepare_class_embeddings(args, dataset):
    class_set_emb = dataset["class_level_hyp"]["objects"]
    if args.dataset_name == "./local_datasets/Donghyun99/CUB-200-2011_latents":
        class_set_emb = [
            cls.replace(cls[:4], "") if cls[:3].isdigit() and cls[3] == "." else cls
            for cls in class_set_emb
        ]

    emb_map = {
        "oh": torch.tensor(dataset["class_level_oh"]["embeddings"]),
        "hyp": torch.tensor(dataset["class_level_hyp"]["embeddings"]),
        "sph": torch.tensor(dataset["class_level_sph"]["embeddings"]),
    }
    if args.emb_type not in emb_map:
        raise ValueError(f"Unknown embedding type: {args.emb_type}")
    class_embeddings = emb_map[args.emb_type]

    class_set_plain = dataset["sample_level"].features["label"].names
    selected_class_set = SELECTED_CLASSES[args.dataset_name]
    if args.dataset_name == "./local_datasets/Donghyun99/Stanford-Cars_latents":
        selected_labels = [class_set_plain.index(cls.replace("_", " ")) for cls in selected_class_set]
    else:
        selected_labels = [class_set_plain.index(cls) for cls in selected_class_set]

    return class_embeddings, class_set_plain, class_set_emb, selected_labels


def decode_latents(vae, latents):
    latents = latents.to(vae.dtype)
    images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return images


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def create_repository(args, accelerator):
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=args.hub_private_repo,
            )


def set_verbose_level(accelerator):
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def resume_training_if_needed(args, accelerator, num_update_steps_per_epoch):
    if not args.resume_from_checkpoint:
        return 0, 0

    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        candidates = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
        path = sorted(candidates, key=lambda x: int(x.split("-")[1]))[-1] if candidates else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        return 0, 0

    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(os.path.join(args.output_dir, path))
    global_step = int(path.split("-")[1])
    resume_global_step = global_step * args.gradient_accumulation_steps
    first_epoch = global_step // num_update_steps_per_epoch
    resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    return first_epoch, resume_step


def prepare_transforms(args):
    crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    flip = transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x)
    return transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            crop,
            flip,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def preprocess_train(examples, train_transforms, image_column, class_column, latents_column):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["class_labels"] = examples[class_column]
    examples["latents"] = examples[latents_column]
    return examples


def collate_fn_builder(class_embeddings, class_set_plain, class_set_emb):
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(
            memory_format=torch.contiguous_format
        )
        latents = torch.stack([torch.tensor(example["latents"]) for example in examples])
        class_labels = torch.tensor([example["class_labels"] for example in examples])
        class_names = [class_set_plain[label] for label in class_labels]
        class_names = [name.replace(" ", "_") for name in class_names]
        class_emb_idx = torch.tensor([class_set_emb.index(name) for name in class_names])
        cond_embeddings = class_embeddings[class_emb_idx]
        return {
            "pixel_values": pixel_values.float(),
            "class_labels": class_labels,
            "cond_embeddings": cond_embeddings,
            "latents": latents,
        }

    return collate_fn


def log_wandb_latents(tracker, tag_prefix, images, latents, captions, step):
    if tracker.name != "wandb":
        return
    images = images.detach().float().cpu()
    images = (images / 2 + 0.5).clamp(0, 1)
    wandb_images = []
    for img_tensor, caption in zip(images, captions):
        if img_tensor.shape[0] == 3:
            img = img_tensor.permute(1, 2, 0).numpy()
        else:
            img = img_tensor.numpy()
        wandb_images.append(wandb.Image(img, caption=caption))
    tracker.log(
        {
            f"{tag_prefix}/images": wandb_images,
            f"{tag_prefix}/latents": wandb.Histogram(latents.detach().float().cpu().numpy().flatten()),
        },
        step=step,
    )


def create_pipeline(args, transformer, vae, noise_scheduler, weight_dtype):
    pipeline = DiTPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.vae = vae
    pipeline.scheduler = noise_scheduler
    return pipeline


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_verbose_level(accelerator)

    if args.seed is not None:
        set_seed(args.seed)

    create_repository(args, accelerator)

    dataset = load_from_disk(args.dataset_name)
    image_column = "images"
    class_column = "label"
    latents_column = "latents"

    class_embeddings, class_set_plain, class_set_emb, selected_class_labels = prepare_class_embeddings(args, dataset)

    vae = AutoencoderKL.from_pretrained(
        args.dit_components_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    transformer_sample_size = args.resolution_latent
    logger.info("Initializing a DiT-S/4 sized transformer (sample_size=%s).", transformer_sample_size)
    transformer = build_dit_b_2_transformer(sample_size=transformer_sample_size)

    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            transformer.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=DiTTransformer2DModel,
            model_config=transformer.config,
        )

    transformer.requires_grad_(True)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    if ema_model is not None:
        ema_model.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers(transformer)

    try:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.dit_components_model_name_or_path,
            subfolder="scheduler",
            revision=args.revision,
            variant=args.variant,
        )
        logger.info("Loaded FlowMatch Euler scheduler from %s.", args.dit_components_model_name_or_path)
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "Falling back to a freshly initialized FlowMatchEulerDiscreteScheduler because loading from %s failed: %s",
            args.dit_components_model_name_or_path,
            error,
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=args.sample_steps,
            shift=1.0,
        )

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_transforms = prepare_transforms(args)

    def train_preprocess(examples):
        return preprocess_train(examples, train_transforms, image_column, class_column, latents_column)

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["sample_level"] = dataset["sample_level"].shuffle(seed=args.seed)
            idx = torch.tensor([(label in selected_class_labels) for label in dataset["sample_level"]["label"]])
            idx = torch.where(idx)[0]
            if len(idx) < args.max_train_samples:
                repeat_factor = args.max_train_samples // len(idx) + 1
                idx = idx.repeat(repeat_factor)[: args.max_train_samples]
            dataset["sample_level"] = dataset["sample_level"].select(idx)
        train_dataset = dataset["sample_level"].with_transform(train_preprocess)

    logger.info("Dataset size: %s", train_dataset.num_rows)
    train_labels = [class_set_plain[label] for label in dataset["sample_level"]["label"]]
    unique_train_labels = sorted(set(train_labels))
    logger.info("Label of the samples: %s", unique_train_labels)

    collate_fn = collate_fn_builder(class_embeddings, class_set_plain, class_set_emb)
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

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
    )

    if ema_model is None:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, ema_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer,
            ema_model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    logger.info(
        "RANK=%s LOCAL_RANK=%s DEVICE=%s WORLD_SIZE=%s",
        accelerator.process_index,
        accelerator.local_process_index,
        accelerator.device,
        accelerator.num_processes,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.splitext(os.path.basename(__file__))[0])

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    max_train_steps = args.max_train_steps
    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %s", dataset.num_rows)
    logger.info("  Num Epochs = %s", num_epochs)
    logger.info("  Instantaneous batch size per device = %s", args.train_batch_size)
    logger.info("  Total train batch size (parallel & accumulation) = %s", total_batch_size)
    logger.info("  Gradient Accumulation steps = %s", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %s", max_train_steps)

    global_step = 0
    train_loss = 0.0
    first_epoch, resume_step = resume_training_if_needed(args, accelerator, num_update_steps_per_epoch)

    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_main_process, mininterval=1)
    transformer.train()

    for epoch in range(first_epoch, num_epochs):
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                latents = batch["latents"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                class_labels = batch["class_labels"].to(accelerator.device, non_blocking=True)
                if torch.rand(1).item() < args.guidance_dropout_prob:
                    class_labels = torch.zeros_like(class_labels)

                noise = torch.randn(latents.shape, dtype=weight_dtype, device=latents.device)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    1,
                    noise_scheduler.config.num_train_timesteps + 1,
                    (bsz,),
                    device=latents.device,
                ).float()

                noisy_latents = noise_scheduler.scale_noise(latents, timesteps, noise)

                if isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
                    target = latents - noise
                else:
                    if args.prediction_type is not None:
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                model_pred = transformer(noisy_latents, timesteps, class_labels, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr_compatible(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)],
                        dim=1,
                    ).min(dim=1)[0]
                    if not isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
                        if noise_scheduler.config.prediction_type == "epsilon":
                            mse_loss_weights = mse_loss_weights / snr
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if ema_model is not None:
                    ema_model.step(transformer.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    if args.checkpoints_total_limit is not None:
                        ckpt_limit(args)
                    wandb_run_id = ""
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            wandb_run_id = tracker.run.id
                    save_path = os.path.join(
                        args.output_dir,
                        f"checkpoint-{global_step}-{args.emb_type}-{wandb_run_id}",
                    )
                    accelerator.save_state(save_path)
                    logger.info("Saved state to %s", save_path)

            progress_bar.set_postfix(step_loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

            if global_step >= args.max_train_steps:
                break

        if not accelerator.is_main_process or epoch % args.validation_epochs == 0:
            continue

        pipeline = create_pipeline(
            args,
            unwrap_model(accelerator, transformer),
            vae,
            noise_scheduler,
            weight_dtype,
        )
        pipeline.vae.to(accelerator.device, dtype=weight_dtype)

        with torch.no_grad():
            generated_latents, generated_labels = log_validation(
                pipeline,
                args,
                accelerator,
                epoch,
                class_embeddings=class_embeddings,
                class_set=class_set_emb,
                unique_train_labels=unique_train_labels,
            )
            generated_latents = generated_latents.to(pipeline.vae.dtype)
            generated_images = decode_latents(pipeline.vae, generated_latents)

            reference_latents = batch["latents"][: args.num_validation_images].to(pipeline.vae.dtype)
            original_images = decode_latents(pipeline.vae, reference_latents)

            original_labels = (
                batch["class_labels"][: args.num_validation_images].detach().cpu().tolist()
            )
            generated_captions = [class_set_emb[idx] for idx in generated_labels]
            original_captions = [class_set_plain[idx] for idx in original_labels]

            for tracker in accelerator.trackers:
                log_wandb_latents(
                    tracker,
                    "validation/original",
                    original_images,
                    reference_latents,
                    original_captions,
                    global_step,
                )
                log_wandb_latents(
                    tracker,
                    "validation/generated",
                    generated_images,
                    generated_latents,
                    generated_captions,
                    global_step,
                )

        del pipeline
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_fp32 = unwrap_model(accelerator, transformer.to(torch.float32))
        transformer_fp32.save_pretrained(
            os.path.join(args.output_dir, "transformer"),
            safe_serialization=True,
        )

        if args.validation_prompt is not None:
            pipeline = create_pipeline(
                args,
                transformer_fp32,
                vae,
                noise_scheduler,
                weight_dtype,
            )
            pipeline.vae.to(accelerator.device, dtype=weight_dtype)
            with torch.no_grad():
                generated_latents_list, generated_labels = log_validation(
                    pipeline,
                    args,
                    accelerator,
                    epoch,
                    class_embeddings=class_embeddings,
                    class_set=class_set_emb,
                    unique_train_labels=unique_train_labels,
                    is_final_validation=True,
                )
                generated_latents = generated_latents_list[0].to(pipeline.vae.dtype)
                generated_images = decode_latents(pipeline.vae, generated_latents)
                generated_captions = [class_set_emb[idx] for idx in generated_labels]

                for tracker in accelerator.trackers:
                    log_wandb_latents(
                        tracker,
                        "validation/final_generated",
                        generated_images,
                        generated_latents,
                        generated_captions,
                        global_step,
                    )
            del pipeline
            torch.cuda.empty_cache()

    accelerator.end_training()


if __name__ == "__main__":
    main()
