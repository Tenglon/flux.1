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
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    DiTPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler,
    DiTTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import wandb
from utils import ckpt_limit, enable_xformers, log_validation, tokenize_captions2
from hier_util import HierUtil


logger = get_logger(__name__, log_level="INFO")


def compute_snr_flow_matching(noise_scheduler, timesteps):
    """
    Computes SNR for flow matching schedulers.
    
    For flow matching: x_t = (1 - σ) * x_0 + σ * x_1
    Signal strength: (1 - σ)
    Noise strength: σ
    SNR = ((1 - σ) / σ)^2
    
    Args:
        noise_scheduler: Flow matching scheduler with sigmas attribute
        timesteps: Tensor of timesteps (in [0, num_train_timesteps])
        
    Returns:
        Tensor: SNR values for each timestep
    """
    # Get sigmas for the given timesteps
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=timesteps.dtype)
    
    # Convert timesteps to indices using the same logic as scale_noise
    # Use index_for_timestep if available, otherwise compute directly
    if hasattr(noise_scheduler, 'index_for_timestep'):
        # Use the scheduler's built-in method to find indices
        schedule_timesteps = noise_scheduler.timesteps.to(device=timesteps.device)
        indices = torch.tensor(
            [noise_scheduler.index_for_timestep(t.item(), schedule_timesteps) for t in timesteps],
            device=timesteps.device
        )
    elif hasattr(noise_scheduler, 'timesteps') and noise_scheduler.timesteps is not None:
        # Find indices for each timestep by finding closest match
        schedule_timesteps = noise_scheduler.timesteps.to(device=timesteps.device)
        timesteps_expanded = timesteps.unsqueeze(-1)  # [batch_size, 1]
        schedule_expanded = schedule_timesteps.unsqueeze(0)  # [1, num_sigmas]
        # Find closest timestep index for each input timestep
        distances = (schedule_expanded - timesteps_expanded).abs()  # [batch_size, num_sigmas]
        indices = distances.argmin(dim=1)  # [batch_size]
    else:
        # Fallback: convert timesteps directly to indices
        # timesteps are in [0, num_train_timesteps], sigmas has num_train_timesteps elements
        num_sigmas = len(sigmas)
        # Normalize timesteps to [0, 1] and scale to [0, num_sigmas-1]
        indices = (timesteps / noise_scheduler.config.num_train_timesteps * (num_sigmas - 1)).long()
        indices = torch.clamp(indices, 0, num_sigmas - 1)
    
    # Get sigma values for the timesteps
    sigma = sigmas[indices].float()
    
    # Ensure sigma is in valid range [0, 1] to avoid division by zero
    sigma = torch.clamp(sigma, min=1e-8, max=1.0 - 1e-8)
    
    # Compute SNR: SNR = ((1 - σ) / σ)^2
    one_minus_sigma = 1.0 - sigma
    snr = (one_minus_sigma / sigma) ** 2
    
    return snr


def compute_snr_compatible(noise_scheduler, timesteps):
    """
    Computes SNR compatible with both traditional diffusion and flow matching schedulers.
    
    Args:
        noise_scheduler: Either a traditional scheduler (with alphas_cumprod) or flow matching scheduler (with sigmas)
        timesteps: Tensor of timesteps
        
    Returns:
        Tensor: SNR values for each timestep
    """
    # Check if it's a flow matching scheduler
    if isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
        return compute_snr_flow_matching(noise_scheduler, timesteps)
    else:
        # Use the standard compute_snr for traditional schedulers
        return compute_snr(noise_scheduler, timesteps)


def registor_new_accelerate(args, accelerator, ema_model):
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "transformer_ema"), DiTTransformer2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = DiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


def _prepare_wandb_images(image_batch, captions):
    """Convert a batch of image tensors into wandb.Image objects with captions."""
    if image_batch is None or len(image_batch) == 0:
        return []

    images = image_batch.detach().float().cpu()
    images = (images / 2 + 0.5).clamp(0, 1)

    wandb_images = []
    for img_tensor, caption in zip(images, captions):
        if img_tensor.shape[0] == 3:
            img_np = img_tensor.permute(1, 2, 0).numpy()
        else:
            img_np = img_tensor.numpy()
        wandb_images.append(wandb.Image(img_np, caption=caption))

    return wandb_images


def enable_xformers(model):
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        model.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")


import torch, torch.nn as nn
from diffusers.models.normalization import AdaLayerNormZero
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings


def build_dit_b_2_transformer(sample_size):
    """Create a DiT transformer matching the Facebook DiT-B/2 architecture."""

    dit_b_2_config = dict(
        sample_size=sample_size,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_attention_heads=12,
        attention_head_dim=64,
        num_layers=12,
        attention_bias=True,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
    )

    dit_s_4_config = dict(
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

    return DiTTransformer2DModel(**dit_s_4_config)




def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training tiny model script.")
    parser.add_argument("--dataset_name", type=str, default=None, required=True, help="The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that HF Datasets can understand.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The config of the Dataset, leave as None if there's only one config.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default="facebook/DiT-B-2-256x256",
        help="Path to the DiT transformer weights to use instead of the SD3 DiT. Defaults to the 110M parameter DiT/B-2.",
    )
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")

    parser.add_argument("--train_data_dir", type=str, default=None, help="A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None, help="The directory where the downloaded models and datasets will be stored.")

    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    parser.add_argument("--resolution_latent", type=int, default=64, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    parser.add_argument("--center_crop", default=False, action="store_true", help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")

    parser.add_argument("--random_flip", default=False, action="store_true", help="whether to randomly flip images horizontally")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--max_train_steps", type=int, default=None, required=True, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="The scheduler type to use. Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use Exponential Moving Average for the final model weights.")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_private_repo", action="store_true", help="Whether or not to create a private repository.")
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"], help="Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai) for experiment tracking and logging of model metrics and model checkpoints")
    parser.add_argument("--logging_dir", type=str, default="logs", help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample"], help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.")
    parser.add_argument("--sample_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming using `--resume_from_checkpoint`.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    parser.add_argument("--guidance_dropout_prob", type=float, default=0.1, help="The dropout probability of the guidance.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="The scale of the guidance.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument("--emb_type", type=str, default="oh", help="The type of the embedding.")

    parser.add_argument("--validation_epochs", type=int, default=10, help="The number of epochs to validate the model.")
    parser.add_argument("--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference.")
    parser.add_argument("--num_validation_images", type=int, default=8, help="The number of images to validate the model.")



    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args

args = parse_args()


PROMPT_MAPPING = {
    "./local_datasets/keremberke/pokemon-classification_latents": "A photo of a pokemon.",
    "./local_datasets/Donghyun99/CUB-200-2011_latents": "A photo of a bird.",
    "./local_datasets/Donghyun99/Stanford-Cars_latents": "A photo of a car.",
}
PROMPT_TEMPLATE = PROMPT_MAPPING[args.dataset_name]

selected_hier_mapping = {
    "./local_datasets/keremberke/pokemon-classification_latents": HierUtil.get_selected_pokemon(),
    "./local_datasets/Donghyun99/CUB-200-2011_latents": HierUtil.get_selected_birds(),
    "./local_datasets/Donghyun99/Stanford-Cars_latents": HierUtil.get_selected_cars(),
}



def main():
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.cuda.is_available():                 # ROCm 也用 torch.cuda 接口
        torch.cuda.set_device(accelerator.local_process_index)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_verbose_level(accelerator)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation, create output_dir and push to hub
    create_repository(args, accelerator)

    dataset = load_from_disk(args.dataset_name)
    image_column = 'images'
    class_column = 'label'
    latents_column = 'latents'
    
    class_set_emb = dataset['class_level_hyp']['objects']
    if args.dataset_name == "./local_datasets/Donghyun99/CUB-200-2011_latents":
        class_set_emb = [cls.replace(cls[:4], "") if cls[:3].isdigit() and cls[3] == "." else cls for cls in class_set_emb]

    # class_embeddings = torch.randn(len(class_set_emb), 300)
    # for i, name in enumerate(class_set_emb):
        # PROMPT_TEMPLATE_EMBEDDING = PROMPT_TEMPLATE + '{}'
        # class_embeddings[i] = text_encoder(tokenizer(PROMPT_TEMPLATE_EMBEDDING.format(name), padding=True, return_tensors="pt").input_ids.to(accelerator.device))[1]
    if args.emb_type == "oh":
        class_embeddings = torch.tensor(dataset['class_level_oh']['embeddings'])
    elif args.emb_type == "hyp":
        class_embeddings = torch.tensor(dataset['class_level_hyp']['embeddings'])
    elif args.emb_type == "sph":
        class_embeddings = torch.tensor(dataset['class_level_sph']['embeddings'])
    else:
        raise ValueError(f"Unknown embedding type: {args.emb_type}")

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    if args.transformer_model_name_or_path:
        logger.info(
            "Argument --transformer_model_name_or_path=%s is ignored; using a freshly initialized DiT-B/2 instead.",
            args.transformer_model_name_or_path,
        )

    transformer_sample_size = args.resolution_latent
    logger.info(
        "Initializing a DiT-S/4 sized transformer from scratch (sample_size=%s, patch_size=4).",
        transformer_sample_size,
    )
    transformer = build_dit_b_2_transformer(sample_size=transformer_sample_size)


    ema_model = EMAModel(
            transformer.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=DiTTransformer2DModel,
            model_config=transformer.config,
        )
    
    # `accelerate` 0.16.0 will have better support for customized saving
    registor_new_accelerate(args, accelerator, ema_model)

    # freeze parameters of models to save more memory

    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = set_mixed_precision(args, accelerator)

    # Move transformer, vae and text_encoder to device and cast to weight_dtype
    ema_model.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers(transformer)

    # lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", algorithm_type="dpmsolver++", use_karras_sigmas=True)
        # noise_scheduler = DDIMScheduler(
        #     num_train_timesteps=args.sample_steps,
        #     beta_schedule=args.ddpm_beta_schedule,
        #     prediction_type=args.prediction_type,
        # )
        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=args.sample_steps,
            shift=1.0,
        )
    else:
        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=args.sample_steps,
            shift=1.0,
        )
        # noise_scheduler = DDIMScheduler(num_train_timesteps=args.sample_steps, beta_schedule=args.ddpm_beta_schedule)
        # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", algorithm_type="dpmsolver++", use_karras_sigmas=True)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model  

    # Preprocessing the datasets and DataLoaders creation.
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
        # examples["input_ids"] = tokenize_captions(tokenizer, examples, caption_column)
        # examples["input_ids"] = tokenize_captions2(tokenizer, examples, class_column, PROMPT_TEMPLATE)
        examples["class_labels"] = examples[class_column]
        examples["latents"] = examples[latents_column]
        return examples


    class_set_plain = dataset['sample_level'].features[class_column].names
    selected_class_set = selected_hier_mapping[args.dataset_name]
    if args.dataset_name == "./local_datasets/Donghyun99/Stanford-Cars_latents":
        selected_class_labels = [class_set_plain.index(cls.replace("_", " ")) for cls in selected_class_set]
    else:
        selected_class_labels = [class_set_plain.index(cls) for cls in selected_class_set]
    
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['sample_level'] = dataset['sample_level'].shuffle(seed=args.seed)
            # idx = torch.tensor([(label in [1, 2, 3]) for label in dataset['sample_level']['label']])
            idx = torch.tensor([(label in selected_class_labels) for label in dataset['sample_level']['label']])
            idx = torch.where(idx)[0]
            # Resample idx to match args.max_train_samples size
            if len(idx) < args.max_train_samples:
                # Calculate how many times we need to repeat the indices
                repeat_factor = args.max_train_samples // len(idx) + 1
                # Repeat the indices and then select the required number
                idx = idx.repeat(repeat_factor)[:args.max_train_samples]
            dataset['sample_level'] = dataset['sample_level'].select(idx)
            # if args.max_train_samples is not None:
                # dataset['sample_level'] = dataset['sample_level'].select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset['sample_level'].with_transform(preprocess_train)
        # print the label of the first 10 samples

    logger.info(f"Dataset size: {train_dataset.num_rows}")
    train_labels = [class_set_plain[label] for label in dataset['sample_level']['label']]
    unique_train_labels = list(set(train_labels))
    logger.info(f"Label of the samples: {unique_train_labels}")

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        latents_list = [torch.tensor(example['latents']) for example in examples]   
        latents = torch.stack(latents_list)
        # input_ids = torch.stack([example["input_ids"] for example in examples])
        class_labels = torch.tensor([example["class_labels"] for example in examples])
        class_names = [class_set_plain[label] for label in class_labels]
        class_names = [name.replace(" ", "_") for name in class_names]
        class_emb_idx = torch.tensor([class_set_emb.index(name) for name in class_names])
        cond_embeddings = class_embeddings[class_emb_idx]
        return {"pixel_values": pixel_values, "class_labels": class_labels, "cond_embeddings": cond_embeddings, "latents": latents}

    # DataLoaders creation:
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
    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    transformer, ema_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, ema_model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    logger.info(f"RANK={accelerator.process_index} LOCAL_RANK={accelerator.local_process_index} "
                f"DEVICE={accelerator.device} WORLD_SIZE={accelerator.num_processes}",
                extra={"main_process_only": False})
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # max_train_steps = args.num_epochs * num_update_steps_per_epoch
    max_train_steps = args.max_train_steps
    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset.num_rows}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    first_epoch, resume_step = resume_training_if_needed(args, accelerator, num_update_steps_per_epoch)

    # Train!
    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_main_process, mininterval=1)
    transformer.train()
    for epoch in range(first_epoch, num_epochs):
        train_loss = 0.0

        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                latents = batch["latents"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                class_labels = batch["class_labels"].to(accelerator.device, non_blocking=True)
                if torch.rand(1).item() < args.guidance_dropout_prob:
                    class_labels = torch.zeros_like(class_labels)

                # Sample noise that we'll add to the images
                noise = torch.randn(latents.shape, dtype=weight_dtype, device=latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(1, noise_scheduler.config.num_train_timesteps + 1, (bsz,), device=latents.device).float()

                noisy_latents = noise_scheduler.scale_noise(latents, timesteps, noise)

                # Get the target for loss depending on the scheduler type
                # Flow Matching schedulers predict velocity field directly, not epsilon or v_prediction
                if isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
                    # For Flow Matching: target is the velocity field v = x_1 - x_0
                    # where x_0 is noise and x_1 is the real data (latents)
                    target = latents - noise
                else:
                    # For traditional diffusion schedulers (DDPM, DDIM, etc.)
                    if args.prediction_type is not None:
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = transformer(noisy_latents, timesteps, class_labels, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr_compatible(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    # Flow Matching doesn't use prediction_type, so we use a simple weighting
                    if isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler):
                        # For Flow Matching, use uniform weighting (or adjust as needed)
                        pass  # Keep mse_loss_weights as computed from SNR
                    elif hasattr(noise_scheduler.config, 'prediction_type'):
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

                # Backpropagate BUGGY.
                # if accelerator.sync_gradients:
                    # accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(transformer.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            ckpt_limit(args)

                        # Get the wandb run ID if wandb is being used
                        wandb_run_id = ""
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                wandb_run_id = tracker.run.id
                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-{args.emb_type}-{wandb_run_id}")
                        accelerator.save_state(save_path)

                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs != 0:
                # create pipeline

                pipeline = DiTPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                # Replace scheduler with the one used during training to ensure consistency
                # This ensures validation uses the same scheduler configuration as training
                pipeline.scheduler = noise_scheduler
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

                    vae_dtype = pipeline.vae.dtype
                    generated_latents = generated_latents.to(vae_dtype)

                    # This decode is based on the modified DiTPipeline implementation
                    # that outputs the latents along with the images
                    generated_images = pipeline.vae.decode(
                        generated_latents / pipeline.vae.config.scaling_factor,
                        return_dict=False,
                        generator=None,
                    )[0]

                    reference_latents = latents[: args.num_validation_images].to(vae_dtype)
                    images_decode = pipeline.vae.decode(
                        reference_latents / pipeline.vae.config.scaling_factor,
                        return_dict=False,
                        generator=None,
                    )[0]

                    original_latents = reference_latents.detach().float().cpu()
                    generated_latents_cpu = generated_latents.detach().float().cpu()

                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            original_labels = (
                                batch["class_labels"][: args.num_validation_images]
                                .detach()
                                .cpu()
                                .tolist()
                            )

                            num_generated = min(len(generated_labels), generated_images.shape[0])
                            num_original = min(len(original_labels), images_decode.shape[0])

                            generated_captions = [
                                class_set_emb[idx] for idx in generated_labels[:num_generated]
                            ]
                            original_captions = [
                                class_set_plain[idx] for idx in original_labels[:num_original]
                            ]

                            tracker.log(
                                {
                                    "validation/original_images": _prepare_wandb_images(
                                        images_decode[:num_original], original_captions
                                    ),
                                    "validation/generated_images": _prepare_wandb_images(
                                        generated_images[:num_generated], generated_captions
                                    ),
                                    "validation/original_latents": wandb.Histogram(
                                        original_latents.numpy().flatten()
                                    ),
                                    "validation/generated_latents": wandb.Histogram(
                                        generated_latents_cpu.numpy().flatten()
                                    ),
                                },
                                step=global_step,
                            )

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = transformer.to(torch.float32)
        unwrapped_transformer = unwrap_model(transformer)
        
        # Save full model weights
        unwrapped_transformer.save_pretrained(
            os.path.join(args.output_dir, "transformer"), safe_serialization=True
        )

        # Final inference
        # Load previous pipeline
        if args.validation_prompt is not None:
            pipeline = DiTPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )

            pipeline.transformer = unwrapped_transformer
            # Replace scheduler with the one used during training to ensure consistency
            pipeline.scheduler = noise_scheduler

            # run inference
            with torch.no_grad():
                generated_latents_object, generated_labels = log_validation(
                    pipeline,
                    args,
                    accelerator,
                    epoch,
                    class_embeddings=class_embeddings,
                    class_set=class_set_emb,
                    unique_train_labels=unique_train_labels,
                    is_final_validation=True,
                )
                generated_latents = generated_latents_object[0].to(pipeline.vae.dtype)
                generated_images = pipeline.vae.decode(
                    generated_latents / pipeline.vae.config.scaling_factor,
                    return_dict=False,
                    generator=None,
                )[0]

                generated_latents_cpu = generated_latents.detach().float().cpu()

                num_generated = min(len(generated_labels), generated_images.shape[0])
                generated_captions = [
                    class_set_emb[idx] for idx in generated_labels[:num_generated]
                ]

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation/final_generated_images": _prepare_wandb_images(
                                    generated_images[:num_generated], generated_captions
                                ),
                                "validation/final_generated_latents": wandb.Histogram(
                                    generated_latents_cpu.numpy().flatten()
                                ),
                            },
                            step=global_step,
                        )

            del pipeline
            torch.cuda.empty_cache()


    accelerator.end_training()


def resume_training_if_needed(args, accelerator, num_update_steps_per_epoch):


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    else:
        first_epoch = 0
        resume_step = 0
    return first_epoch,resume_step

    

def set_mixed_precision(args, accelerator):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    return weight_dtype

def create_repository(args, accelerator):
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

def set_verbose_level(accelerator):
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


if __name__ == "__main__":
    main()