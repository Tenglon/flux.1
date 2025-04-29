from datasets import load_from_disk
from diffusers import AutoencoderKL
import os
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator, InitProcessGroupKwargs
from huggingface_hub import create_repo, upload_folder
import torch
import wandb
from torchvision import transforms
import datasets
import diffusers
import logging
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.utils.torch_utils import is_compiled_module
from utils import log_validation
logger = get_logger(__name__, log_level="INFO")

class args:
    dataset_name = "./local_datasets/Donghyun99/CUB-200-2011_latents"
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    output_dir = "./check_output"
    logging_dir = "logs"
    revision = "main"
    train_batch_size = 8
    dataloader_num_workers = 1
    variant = None
    resolution = 256
    center_crop = False
    random_flip = False
    max_train_samples = None
    seed = 42
    num_validation_images = 8
    validation_prompt = None
    guidance_scale = 0

def main():
    dataset = load_from_disk(args.dataset_name)
    image_column = 'image'
    class_column = 'label'
    latents_column = 'latents'
    
    train_dataset = dataset['sample_level']
    emb_dataset = dataset['class_level_hyp']
    class_embeddings = torch.tensor(emb_dataset['embeddings'])
    
    class_set = train_dataset.features[class_column].names

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="wandb",
        project_config=accelerator_project_config,
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

    vae.to(accelerator.device)

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

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['sample_level'] = dataset['sample_level'].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset['sample_level'].with_transform(preprocess_train)  

    logger.info(f"Dataset size: {train_dataset.num_rows}")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

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

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model  

    for i, batch in enumerate(train_dataloader):
        
        images = batch['pixel_values']
        latents = batch['latents']
        cond_embeddings = batch['cond_embeddings']

        if accelerator.is_main_process:

            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=vae.dtype,
                safety_checker=None
            )
            images_decode = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False, generator=None)[0]

            for tracker in accelerator.trackers:

                if tracker.name == "wandb":
                    tracker.log(
                        {
                            'original_image': [
                                wandb.Image(image, caption=f"{i}: the original image") for i, image in enumerate(images)
                            ],
                            'decoded_image': [
                                wandb.Image(image, caption=f"{i}: the decoded image") for i, image in enumerate(images_decode)
                            ]
                        }
                    )
        break

def create_repository(args, accelerator):
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

def set_verbose_level(accelerator):
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


if __name__ == "__main__":
    main()