#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from utils import ckpt_limit, enable_xformers, tokenize_captions, tokenize_captions2, restum_from_ckpt

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# 
def save_model_card(repo_id: str, images: list = None, base_model: str = None, dataset_name: str = None, repo_folder: str = None):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""# LoRA text2image fine-tuning - {repo_id} These are LoRA adaption weights for {base_model}. 
    The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n {img_str} """

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(pipeline, args, accelerator, epoch, is_final_validation=False, class_embeddings=None, class_set=None):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    prompt_idxs = random.sample(range(len(class_set)), args.num_validation_images)

    # START: add class embeddings
    prompt_idxs = random.sample(range(len(class_set)), args.num_validation_images)
    class_labels = torch.tensor(prompt_idxs).to(accelerator.device)
    
    cond_embeddings = class_embeddings[prompt_idxs].to(accelerator.device)
    classidx2name = {i: name for i, name in enumerate(class_set)}
    prompt_labels = [classidx2name[i] for i in prompt_idxs]
    prompt_labels = [PROMPT_TEMPLATE.format(prompt_label) for prompt_label in prompt_labels]

    with autocast_ctx:
        generated = pipeline(prompt_labels, guidance_scale=args.guidance_scale, num_inference_steps=30, generator=generator, class_labels=class_labels)
        generated_images = generated.images
        images = generated_images
    # END: add class embeddings

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {class_set[prompt_idxs[i]]}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


def parse_args():
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")

    parser.add_argument("--dataset_name", type=str, required=True, help="The namqANhb#7hS!2EVA6e of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The config of the Dataset, leave as None if there's only one config.")
    parser.add_argument("--train_data_dir", type=str, default=None, help="A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified.")
    parser.add_argument("--image_column", type=str, default="image", help="The column of the dataset containing an image.")
    parser.add_argument("--caption_column", type=str, default="text", help="The column of the dataset containing a caption or a list of captions.")
    parser.add_argument("--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images that should be generated during validation with `validation_prompt`.")
    parser.add_argument("--validation_epochs", type=int, default=2, help="Run fine-tuning validation every X epochs. The validation process consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images`.")

    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned-lora", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", type=str, default=None, help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images, all train/validation dataset will be resized to this resolution")
    parser.add_argument("--center_crop", default=False, action="store_true", help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")
    parser.add_argument("--random_flip", action="store_true", help="whether to randomly flip images horizontally")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None, required=True, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help= 'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--prediction_type", type=str, default=None, help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Supported platforms are `tensorboard`, `wandb` and `comet_ml`. Use `all` to report to all integrations.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates. Resuming training using `--resume_from_checkpoint`.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."))
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA.")
    parser.add_argument("--rank", type=int, default=32, help=("The dimension of the LoRA update matrices."))

    parser.add_argument("--guidance_dropout_prob", type=float, default=0.1, help="The dropout probability of the guidance.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="The scale of the guidance.")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


args = parse_args()

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
    "wanghaofan/pokemon-wiki-captions": ("image", "text_en", "name_en"),
    "Donghyun99/Stanford-Cars": ("image", "label"),
}

PROMPT_MAPPING = {
    "keremberke/pokemon-classification": "A photo of a pokemon.",
    "Donghyun99/CUB-200-2011": "A photo of a bird.",
    "Donghyun99/Stanford-Cars": "A photo of a car.",
}
PROMPT_TEMPLATE = PROMPT_MAPPING[args.dataset_name]

def main():
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    ##### Start Inject class embeddings #####

    # unet._set_class_embedding(
    #     class_embed_type="simple_projection",
    #     act_fn=None, # Not important for identity embedding
    #     num_class_embeds=None, # Not important for identity embedding
    #     projection_class_embeddings_input_dim=768, # Not important for identity embedding
    #     time_embed_dim=unet.time_embedding.linear_1.out_features,
    #     timestep_input_dim=None, # Not important for identity embedding
    # )
    # print(unet.time_embedding.linear_1.weight.shape)

    # unet.class_embedding.weight = torch.nn.Parameter(torch.eye(unet.time_embedding.linear_1.out_features))

    ##### End Inject class embeddings #####
    
    # freeze parameters of models to save more memory
    if args.use_lora:
        unet.requires_grad_(False)
    else:
        unet.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    if args.use_lora:
        unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers(unet)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if not args.use_lora:
        optimizer = optimizer_cls(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
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
    classidx2name = {i: name for i, name in enumerate(class_set)}
    # class_embeddings = torch.randn(len(class_set), 1280) # here 1280 is the dimension of the time embedding, which = unet.time_embedding.linear_1.out_features
    class_embeddings = torch.zeros(len(class_set), 768)
    for i, name in enumerate(class_set):
        PROMPT_TEMPLATE_EMBEDDING = PROMPT_TEMPLATE + '{}'
        class_embeddings[i] = text_encoder(tokenizer(PROMPT_TEMPLATE_EMBEDDING.format(name), padding=True, return_tensors="pt").input_ids.to(accelerator.device))[1]

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

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        # examples["input_ids"] = tokenize_captions(tokenizer, examples, caption_column)
        examples["input_ids"] = tokenize_captions2(tokenizer, examples, class_column, PROMPT_TEMPLATE)
        examples["class_labels"] = examples[class_column]
        return examples

    with accelerator.main_process_first():
        if args.dataset_name == "keremberke/pokemon-classification":
            full_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        else:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            
        if args.max_train_samples is not None:
            full_dataset = full_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = full_dataset.with_transform(preprocess_train)       
        # train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        class_labels = torch.tensor([example["class_labels"] for example in examples])
        cond_embeddings = class_embeddings[class_labels]
        return {"pixel_values": pixel_values, "input_ids": input_ids, "class_labels": class_labels, "cond_embeddings": cond_embeddings}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes

    num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step, first_epoch = 0, 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        global_step, first_epoch = restum_from_ckpt(args, accelerator, num_update_steps_per_epoch)
    else:
        global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset != 0:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                ### Start Inject class embeddings ###
                cond_embeddings = batch["cond_embeddings"]
                if torch.rand(1).item() < args.guidance_dropout_prob:
                    cond_embeddings = torch.zeros_like(cond_embeddings)
                encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
                # class_labels = batch["class_labels"]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, class_labels = cond_embeddings, return_dict=False)[0]
                ### End Inject class embeddings ###

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            ckpt_limit(args)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet)

                        if args.use_lora:   
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet)
                            )
                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )
                        else:
                            # Save full model weights
                            pipeline = DiffusionPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                revision=args.revision,
                                variant=args.variant,
                                torch_dtype=torch.float32,
                            )
                            pipeline.unet = unwrapped_unet
                            pipeline.save_pretrained(args.output_dir)
                            logger.info(f"Full model saved to {args.output_dir}")

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    safety_checker=None
                )
                images = log_validation(pipeline, args, accelerator, epoch, class_embeddings=class_embeddings, class_set=class_set)

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = unwrap_model(unet)
        
        if args.use_lora:
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            StableDiffusionPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
        else:
            # Save full model weights
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=torch.float32,
            )
            pipeline.unet = unwrapped_unet
            pipeline.save_pretrained(args.output_dir)
            logger.info(f"Full model saved to {args.output_dir}")


        # Final inference
        # Load previous pipeline
        if args.validation_prompt is not None:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                safety_checker=None
            )

            if args.use_lora:
                # load attention processors for LoRA
                pipeline.load_lora_weights(args.output_dir)
            else:
                # Load the trained unet
                pipeline.unet = unwrapped_unet

            # run inference
            images = log_validation(pipeline, args, accelerator, epoch, class_embeddings=class_embeddings, class_set=class_set, is_final_validation=True)


    accelerator.end_training()







if __name__ == "__main__":
    main()