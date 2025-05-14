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
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr, cast_training_params, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import wandb
from utils import ckpt_limit, enable_xformers, log_validation, tokenize_captions2
from hier_util import HierUtil


logger = get_logger(__name__, log_level="INFO")


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
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


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
    



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training tiny model script.")
    parser.add_argument("--dataset_name", type=str, default=None, required=True, help="The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that HF Datasets can understand.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The config of the Dataset, leave as None if there's only one config.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
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
    image_column = 'image'
    class_column = 'label'
    latents_column = 'latents'
    
    class_set = dataset['class_level_hyp']['objects']
    # class_embeddings = torch.randn(len(class_set), 300)
    # for i, name in enumerate(class_set):
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

    if False:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
    else:   
        unet = UNet2DConditionModel(
            sample_size=args.resolution_latent,
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
            only_cross_attention=False, # 是否只使用交叉注意力, 而不使用自注意力
            cross_attention_dim=class_embeddings.shape[1], # 交叉注意力维度, 此处设置为条件向量的维度
            projection_class_embeddings_input_dim=None, # 条件向量维度
            # class_embed_type="simple_projection", # 条件向量类型, 可选值为"simple"或"projection"
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
    
    # `accelerate` 0.16.0 will have better support for customized saving
    registor_new_accelerate(args, accelerator, ema_model)

    # freeze parameters of models to save more memory

    unet.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = set_mixed_precision(args, accelerator)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    ema_model.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers(unet)

    # lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", algorithm_type="dpmsolver++", use_karras_sigmas=True)
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.sample_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.sample_steps, beta_schedule=args.ddpm_beta_schedule)
        # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", algorithm_type="dpmsolver++", use_karras_sigmas=True)

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
        cond_embeddings = class_embeddings[class_labels]
        return {"pixel_values": pixel_values, "class_labels": class_labels, "cond_embeddings": cond_embeddings, "latents": latents}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

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
    for epoch in range(first_epoch, num_epochs):
        unet.train()
        train_loss = 0.0

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process, mininterval=1)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                latents = batch["latents"].to(weight_dtype)

                # Sample noise that we'll add to the images
                noise = torch.randn(latents.shape, dtype=weight_dtype, device=latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

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

                encoder_hidden_states = batch["cond_embeddings"][:, None, :] # create a new axis corresponding to sequence length
                if torch.rand(1).item() < args.guidance_dropout_prob:
                    encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
                # encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

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

                # Backpropagate BUGGY.
                # if accelerator.sync_gradients:
                    # accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(unet.parameters())
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
                with torch.no_grad():
                    generated_latents_object, generated_labels = log_validation(pipeline, args, accelerator, epoch, class_embeddings=class_embeddings, class_set=class_set_plain, unique_train_labels=unique_train_labels)
                    generated_latents = generated_latents_object[0]
                    # generated_latents = generated_latents * pipeline.vae.config.scaling_factor

                    generated_images = pipeline.vae.decode(generated_latents.to(torch.bfloat16) / pipeline.vae.config.scaling_factor, 
                                                       return_dict=False, generator=None)[0]
                    
                    images_decode = pipeline.vae.decode(latents[:args.num_validation_images] / pipeline.vae.config.scaling_factor, return_dict=False, generator=None)[0]

                    for tracker in accelerator.trackers:
                        original_labels = batch["class_labels"]

                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    'original_image': [
                                        wandb.Image(image, caption=f"{class_set_plain[original_labels[i]]}") for i, image in enumerate(images_decode)
                                    ],
                                    'generated_image': [
                                        wandb.Image(image, caption=f"{class_set_plain[generated_labels[i]]}") for i, image in enumerate(generated_images)
                                    ],
                                    'latents_histogram': wandb.Histogram(latents.detach().float().cpu().numpy().flatten()),
                                    'generated_latents_histogram': wandb.Histogram(generated_latents.detach().float().cpu().numpy().flatten())
                                }
                            )

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = unwrap_model(unet)
        
        # Save full model weights
        # unet_state_dict = convert_state_dict_to_diffusers(unwrapped_unet.state_dict())
        StableDiffusionPipeline.save_pretrained(
            save_directory=args.output_dir,
            unet=unwrapped_unet,
            safe_serialization=True,
        )

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

            pipeline.unet = unwrapped_unet

            # run inference
            images = log_validation(pipeline, args, accelerator, epoch, class_embeddings=class_embeddings, class_set=class_set, is_final_validation=True)


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