import os
import torch
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import DDPMScheduler
from tqdm import tqdm
# Set CUDA_VISIBLE_DEVICES to 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_epochs = 200
# Data transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust based on model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformations using with_transform
def transform_fn(examples):
    examples["image"] = [transform(img) for img in examples["image"] if img.mode == "RGB"]
    return examples


# Load CUB dataset using HuggingFace datasets
dataset = load_dataset("Donghyun99/CUB-200-2011")
train_dataset = dataset["train"].with_transform(transform_fn)
val_dataset = dataset["test"].with_transform(transform_fn)

# Get the corresponding loader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Test the loader
for batch in train_loader:

    images, labels = batch['image'], batch['label']
    print(images.shape, labels.shape)
    break

model_id = "runwayml/stable-diffusion-v1-5"  # or "CompVis/stable-diffusion-v1-4", etc.
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to("cuda")

# Extract components
unet = pipe.unet
vae = pipe.vae
# We won't use pipe.text_encoder if we do purely class-based conditioning

import torch.nn as nn

num_classes = 200  # example
embed_dim = 768   # must match the U-Net cross-attention dim for SD v1.x

class_embedding = nn.Embedding(num_classes, embed_dim).to("cuda")

for name, param in unet.named_parameters():
    # If "attn2" is in the parameter name, unfreeze it, else freeze
    if "attn2" in name:
        # print(f"Unfreezing {name}")
        param.requires_grad = True
    else:
        # print(f"Freezing {name}")
        param.requires_grad = False

# Also keep the new class embedding trainable
for param in class_embedding.parameters():
    print(f"Unfreezing class embedding")
    param.requires_grad = True


noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085, 
    beta_end=0.012
)

import torch.nn.functional as F
optimizer = torch.optim.AdamW(
    list(unet.parameters()) + list(class_embedding.parameters()),
    lr=1e-5
)

unet.train()
for epoch in range(num_epochs):
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = batch['image'], batch['label']
        images = images.to("cuda")      # (B, 4, H, W)
        labels = labels.to("cuda")      # (B,)

        latents = vae.encode(images).latent_dist.sample() * 0.18215
        
        # Sample a random timestep
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, 
            (latents.shape[0],), device=latents.device
        ).long()
        
        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get class embeddings and reshape for cross-attention
        cond_emb = class_embedding(labels)  # (B, 768)
        # Cross-attention in SD expects (B, sequence_len, cross_attention_dim).
        # We have one "token" per label => sequence_len = 1
        cond_emb = cond_emb.unsqueeze(1)  # (B, 1, 768)

        # Predict noise in the latent space
        noise_pred = unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=cond_emb
        ).sample
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        if i % 100 == 0:
            print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

