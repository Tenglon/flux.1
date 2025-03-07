
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


logger = get_logger(__name__, log_level="INFO")


def log_validation(pipeline, args, accelerator, epoch, is_final_validation=False, class_embeddings=None, class_set=None):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False, desc="Generating images", mininterval=0.5)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

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
                        # wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
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
    parser.add_argument("--num_validation_images", type=int, default=8, help="Number of images that should be generated during validation with `validation_prompt`.")
    parser.add_argument("--validation_epochs", type=int, default=2, help="Run fine-tuning validation every X epochs. The validation process consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images`.")

    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned-lora", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", type=str, default=None, help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--lora_saved_dir", type=str, default=None, help="The directory where the saved lora weights will be stored.")
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
    parser.add_argument("--logging_dir", type=str, default="logs_generate", help="log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
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



DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
    "wanghaofan/pokemon-wiki-captions": ("image", "text_en", "name_en"),
    "Donghyun99/Stanford-Cars": ("image", "label"),
}

PROMPT_TEMPLATE = '{}'

def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)


    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
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
    unet.requires_grad_(False)
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

    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unwrap_model(unet),
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        safety_checker=None
    )
    print(args.lora_saved_dir)
    pipeline.load_lora_weights(
        pretrained_model_name_or_path_or_dict=args.lora_saved_dir,
        weight_name = "pytorch_lora_weights.safetensors",
    )
    pipeline.to(accelerator.device)

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
    class_embeddings = torch.zeros(len(class_set), 768)
    for i, name in enumerate(class_set):
        class_embeddings[i] = text_encoder(tokenizer(PROMPT_TEMPLATE.format(name), padding=True, return_tensors="pt").input_ids.to(accelerator.device))[1]

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-generate", config=vars(args))

    if accelerator.is_main_process:
        # create pipeline
        images = log_validation(pipeline, args, accelerator, epoch = 0, class_embeddings=class_embeddings, class_set=class_set)

if __name__ == "__main__":
    main()