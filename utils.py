from contextlib import nullcontext
import os
import random
import shutil
from diffusers.utils import is_xformers_available
import numpy as np
from packaging import version
from accelerate.logging import get_logger
import torch
import wandb

logger = get_logger(__name__, log_level="INFO")



def log_validation(pipeline, args, accelerator, epoch, is_final_validation=False, class_embeddings=None, class_set=None):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)
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
    prompt_embeds = cond_embeddings[:, None, :] # [batch_size, 1, 768], where 1 is the sequence length
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    classidx2name = {i: name for i, name in enumerate(class_set)}
    prompt_labels = [classidx2name[i] for i in prompt_idxs]
    prompt_labels = [args.validation_prompt.format(prompt_label) for prompt_label in prompt_labels]

    with autocast_ctx:
        generated = pipeline(prompt = None, 
                             guidance_scale=args.guidance_scale, 
                             num_inference_steps=300, 
                             generator=generator, 
                             prompt_embeds=prompt_embeds, 
                             negative_prompt_embeds=negative_prompt_embeds)
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


def ckpt_limit(args):
    checkpoints = os.listdir(args.output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= args.checkpoints_total_limit:
        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)

def enable_xformers(unet):
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")


def restum_from_ckpt(args, accelerator, num_update_steps_per_epoch):
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
            # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        args.resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    return global_step,first_epoch


# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(tokenizer, examples, caption_column, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def tokenize_captions2(tokenizer, examples, class_column, template: str, is_train=True):
    captions = []
    for class_name in examples[class_column]:
        caption = template.format(class_name)
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Class column `{class_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids